from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from genrouter.backends.http_client import exponential_retry_delay, request_with_retry


DEFAULT_IMAGE_DIR = "/tmp/genrouter_images"
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
BLOCKED_HOST_TOKENS = ("youtube.com", "youtu.be")
BLOCKED_SUFFIXES = (".pdf",)
LOW_QUALITY_HOST_TOKENS = (
    "medium.com",
    "substack.com",
    "blog.",
    "news.",
    "linkedin.com",
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "zhihu.com",
    "bilibili.com",
    "weibo.com",
)
PREFERRED_HOST_TOKENS = (
    "wikipedia.org",
    "wikimedia.org",
    ".gov",
    ".edu",
    ".org",
)


class SearchRequestError(RuntimeError):
    """Raised when a configured search backend request fails."""


@dataclass(frozen=True)
class SearchBackendSpec:
    name: str
    base_url: str
    api_key_env: str
    default_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "SearchBackendSpec":
        default_params = dict(data.get("default_params", {}) or {})
        return cls(
            name=str(data["default"]),
            base_url=str(data["base_url"]),
            api_key_env=str(data["api_key_env"]),
            default_params=default_params,
        )


@dataclass
class SerperSearchBackend:
    """Serper-compatible text/image search backend.

    This follows the GenEvolve/SCOPE pattern: POST `{q, num}` to Serper's
    `/search` and `/images`, parse `organic`/`images`, optionally download
    image references, and keep source metadata for trace inspection.
    """

    spec: SearchBackendSpec
    timeout: float = 30.0
    max_retries: int = 3

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def cost_per_query(self) -> float:
        return float(self.spec.default_params.get("cost_per_query", 0.0))

    @property
    def base_url(self) -> str:
        return self.spec.base_url.rstrip("/")

    @property
    def download_images(self) -> bool:
        return bool(self.spec.default_params.get("download_images", False))

    @property
    def download_dir(self) -> Path:
        raw = str(self.spec.default_params.get("download_dir", DEFAULT_IMAGE_DIR))
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_image_bytes(self) -> int:
        return int(self.spec.default_params.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES))

    def text(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = (query or "").strip()
        limit = max(0, int(top_k))
        if not query or limit == 0:
            return []
        payload = self._post("/search", {"q": query, "num": limit})
        organic = _result_list(payload, "organic", "/search")

        candidates: list[dict[str, Any]] = []
        for item in organic:
            url = str(item.get("link") or "").strip()
            if not url or _should_skip_url(url):
                continue
            candidates.append(
                {
                    "kind": "text",
                    "query": query,
                    "title": str(item.get("title") or "").strip() or url,
                    "snippet": str(item.get("snippet") or "").strip(),
                    "source_url": url,
                    "source": _source(url),
                }
            )
        candidates.sort(key=lambda item: _score_evidence(item, query), reverse=True)
        return candidates[:limit]

    def image(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        limit = max(0, int(top_k))
        if not query or limit == 0:
            return []
        payload = self._post("/images", {"q": query, "num": limit})
        raw_results = _result_list(payload, "images", "/images")

        results: list[dict[str, Any]] = []
        download_failures = 0
        for raw in raw_results:
            if len(results) >= limit:
                break
            image_url = str(raw.get("imageUrl") or "").strip()
            if not image_url:
                continue
            thumbnail_url = str(raw.get("thumbnailUrl") or "").strip()
            page_url = str(raw.get("link") or "").strip()
            local_path = ""
            downloaded_image_url = ""
            if self.download_images:
                for candidate_url in _unique_urls([image_url, thumbnail_url]):
                    local_path = self._download(candidate_url)
                    if local_path:
                        downloaded_image_url = candidate_url
                        break
            if self.download_images and not local_path:
                download_failures += 1
                continue
            item = {
                "kind": "image",
                "query": query,
                "title": str(raw.get("title") or "").strip() or image_url,
                "snippet": str(raw.get("source") or "").strip(),
                "source": _source(page_url),
                "source_url": page_url,
                "image_url": image_url,
                "local_path": local_path,
            }
            if downloaded_image_url:
                item["downloaded_image_url"] = downloaded_image_url
            results.append(item)
        if self.download_images and download_failures and not results:
            raise SearchRequestError(f"Failed to download image references for query: {query}")
        return results

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._url(path)
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self._api_key(),
        }

        def request() -> dict[str, Any]:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                raise SearchRequestError(f"Search {path} response must be a JSON object")
            return parsed

        return request_with_retry(
            request,
            max_retries=self.max_retries,
            label=f"search {path}",
            error_cls=SearchRequestError,
            retry_delay=exponential_retry_delay,
            sleep=time.sleep,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _api_key(self) -> str:
        env_name = self.spec.api_key_env
        api_key = (os.environ.get(env_name) or "").strip()
        if not api_key:
            raise SearchRequestError(f"Set {env_name} before using search backend '{self.name}' in real mode")
        return api_key

    def _download(self, image_url: str) -> str:
        cached = self._cached_path(image_url)
        if cached is not None:
            return str(cached)
        target = self._local_path_for(image_url)

        def request():
            response = requests.get(image_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            return response

        try:
            response = request_with_retry(
                request,
                max_retries=self.max_retries,
                label="image download",
                error_cls=SearchRequestError,
                retry_delay=exponential_retry_delay,
                sleep=time.sleep,
            )
        except SearchRequestError:
            return ""

        content = _read_image_content(response, self.max_image_bytes)
        if not _looks_like_image(content):
            return ""
        content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        suffix = _extension_from_content_type(content_type) or target.suffix
        target = target.with_suffix(suffix)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(target)
        return str(target)

    def _cached_path(self, image_url: str) -> Path | None:
        stem = self._local_path_for(image_url).stem
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            path = self.download_dir / f"{stem}{suffix}"
            if path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def _local_path_for(self, image_url: str) -> Path:
        digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:24]
        suffix = ".jpg"
        parsed = urlparse(image_url)
        ext = Path(parsed.path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            suffix = ext
        return self.download_dir / f"{digest}{suffix}"


def build_search_backend(config: dict[str, Any]):
    spec = SearchBackendSpec.from_config(dict(config))
    timeout = float(spec.default_params.get("timeout", 30.0))
    max_retries = int(spec.default_params.get("max_retries", 3))
    return SerperSearchBackend(spec=spec, timeout=timeout, max_retries=max_retries)


def _result_list(payload: dict[str, Any], field: str, path: str) -> list[dict[str, Any]]:
    if field not in payload:
        raise SearchRequestError(f"Search {path} response missing {field}")
    value = payload[field]
    if not isinstance(value, list):
        raise SearchRequestError(f"Search {path} {field} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise SearchRequestError(f"Search {path} {field} items must be objects")
    return value


def _unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _read_image_content(response: Any, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        return b""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            return b""
        chunks.append(chunk)
    return b"".join(chunks)


def _source(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.removeprefix("www.")


def _extension_from_content_type(content_type: str) -> str:
    if content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    if content_type == "image/gif":
        return ".gif"
    return ""


def _looks_like_image(content: bytes) -> bool:
    if content.startswith(b"\xff\xd8\xff"):
        return True
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return True
    if content.startswith((b"GIF87a", b"GIF89a")):
        return True
    return False


def _should_skip_url(url: str) -> bool:
    lowered = url.casefold()
    if any(token in lowered for token in BLOCKED_HOST_TOKENS):
        return True
    path = urlparse(lowered).path
    if any(path.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        return True
    return False


def _score_evidence(item: dict[str, Any], query: str) -> int:
    score = 0
    domain = urlparse(str(item.get("source_url", ""))).netloc.casefold()
    title = str(item.get("title", "")).casefold()
    snippet = str(item.get("snippet", "")).casefold()
    if any(token in domain for token in PREFERRED_HOST_TOKENS):
        score += 4
    if any(token in domain for token in LOW_QUALITY_HOST_TOKENS):
        score -= 4
    if "official" in title or "official" in snippet:
        score += 2
    for token in _query_tokens(query):
        if token in domain:
            score += 2
        if token in title:
            score += 1
        if token in snippet:
            score += 1
    return score


def _query_tokens(query: str) -> list[str]:
    stopwords = {"what", "does", "look", "like", "the", "and", "from", "with", "official", "visual", "reference"}
    normalized = "".join(ch if ch.isalnum() else " " for ch in query.casefold())
    return [token for token in normalized.split() if len(token) >= 3 and token not in stopwords]
