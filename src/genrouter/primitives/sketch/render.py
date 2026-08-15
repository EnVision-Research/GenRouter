"""Rasterize sketch code to a PNG, returning the backend used.

Rendering is the one inherently-code part of the primitive. Order:
  svg      -> headless Chrome ("browser_svg"), else Pillow SVG interpreter ("pillow_svg")
  html_css -> headless Chrome ("browser_html_css"), else Pillow HTML interpreter ("pillow_html_css")
  threejs  -> headless Chrome + WebGL ("browser_threejs"); if no GPU/blank, raises so the
              caller's repair loop can react.

Headless Chrome is best-effort and degrades to the Pillow interpreters when it
is absent or produces a blank frame.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

THREE_CDN_VERSION = "0.160.0"
_THREE_CDN = f"https://esm.sh/three@{THREE_CDN_VERSION}"
_RENDER_TIMEOUT_SECONDS = 25
_UNIFORM_FRACTION = 0.985

_BROWSER_BINARIES = (
    "chrome-headless-shell",
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)
_browser_failed = False


def render_sketch(
    sketch_type: str,
    code: str,
    output_path: Path,
    width: int,
    height: int,
    records: list[dict[str, Any]] | None = None,
) -> str:
    if sketch_type == "svg":
        if "<svg" not in code:
            raise ValueError("SVG sketch code must contain an <svg> element")
        if _browser_screenshot(_wrap_html(code, width, height), output_path, width, height) and _has_content(output_path):
            return "browser_svg"
        _render_svg_pillow(code, output_path, width, height)
        return "pillow_svg"
    if sketch_type == "html_css":
        document = code if _is_document(code) else _wrap_html(code, width, height)
        if _browser_screenshot(document, output_path, width, height) and _has_content(output_path):
            return "browser_html_css"
        _render_html_pillow(code, output_path, width, height)
        return "pillow_html_css"
    if sketch_type == "threejs":
        document = code if _is_document(code) else _threejs_document(code, width, height)
        if _browser_screenshot(document, output_path, width, height) and _has_content(output_path):
            return "browser_threejs"
        raise ValueError(
            "Three.js sketch could not be rendered (headless WebGL unavailable or produced a blank frame)"
        )
    raise ValueError(f"Unsupported sketch_type: {sketch_type}")


# -- headless Chrome ---------------------------------------------------------

def browser_executable() -> str | None:
    if not _browser_rendering_enabled() or _browser_failed:
        return None
    for name in _BROWSER_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _browser_screenshot(document: str, output_path: Path, width: int, height: int) -> bool:
    global _browser_failed

    browser = browser_executable()
    if not browser:
        return False
    html_path = output_path.with_suffix(".render.html")
    try:
        html_path.write_text(document, encoding="utf-8")
        completed = subprocess.run(
            [
                browser, "--headless=new", "--hide-scrollbars", "--no-sandbox",
                "--virtual-time-budget=4000", f"--window-size={width},{height}",
                f"--screenshot={output_path}", html_path.resolve().as_uri(),
            ],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=_RENDER_TIMEOUT_SECONDS,
        )
    except Exception:
        _browser_failed = True
        return False
    if completed.returncode != 0:
        _browser_failed = True
        return False
    return output_path.is_file()


def _browser_rendering_enabled() -> bool:
    return os.environ.get("GENROUTER_BROWSER_RENDERING", "1").strip().casefold() not in {"0", "false", "no"}


def _is_document(code: str) -> bool:
    head = code.lstrip()[:256].casefold()
    return head.startswith("<!doctype html") or head.startswith("<html")


def _wrap_html(code: str, width: int, height: int) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<style>html,body{margin:0;padding:0;overflow:hidden;}*{box-sizing:border-box;}</style>"
        f'</head><body><div style="width:{width}px;height:{height}px;overflow:hidden;">{code}</div></body></html>'
    )


def _threejs_document(code: str, width: int, height: int) -> str:
    """Wrap Three.js module code with a CDN import map and a one-frame render harness."""
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<style>html,body{margin:0;padding:0;overflow:hidden;background:#fff;}</style>"
        f'<script type="importmap">{{"imports":{{"three":"{_THREE_CDN}","three/":"{_THREE_CDN}/"}}}}</script>'
        f'</head><body><canvas id="scene" width="{width}" height="{height}"></canvas>'
        '<script type="module">\nimport * as THREE from \'three\';\nglobalThis.THREE = THREE;\n'
        f"const __W={width}, __H={height};\nlet scene, camera, renderer;\n"
        f"try {{\n{code}\n}} catch (e) {{ document.title='err:'+e; }}\n"
        "scene = (typeof scene!=='undefined' && scene) || globalThis.scene;\n"
        "if (scene) {\n"
        "  if (typeof camera==='undefined' || !camera) { camera=new THREE.PerspectiveCamera(50,__W/__H,0.1,1000); camera.position.set(0,0,6); }\n"
        "  const canvas=document.getElementById('scene');\n"
        "  if (typeof renderer==='undefined' || !renderer) { renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true}); renderer.setClearColor(0xffffff,1); }\n"
        "  renderer.setSize(__W,__H,false); renderer.render(scene,camera);\n"
        "}\n</script></body></html>"
    )


def _has_content(path: Path) -> bool:
    """Reject an effectively-uniform (blank) screenshot."""
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return False
    total = image.width * image.height
    colors = image.getcolors(maxcolors=total) or []
    if not colors:
        return True
    return max(count for count, _ in colors) < _UNIFORM_FRACTION * total


# -- Pillow SVG fallback -----------------------------------------------------

def _render_svg_pillow(svg: str, output_path: Path, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    root = ET.fromstring(svg)
    paint_map = _svg_gradient_map(root)
    for node in root.iter():
        tag = _tag(node.tag)
        style = _svg_style(node, paint_map)
        if tag == "rect":
            x, y, w, h = _n(node.get("x")), _n(node.get("y")), _n(node.get("width")), _n(node.get("height"))
            draw.rectangle([x, y, x + w, y + h], fill=style["fill"], outline=style["stroke"], width=style["sw"])
        elif tag == "circle":
            cx, cy, r = _n(node.get("cx")), _n(node.get("cy")), _n(node.get("r"))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=style["fill"], outline=style["stroke"], width=style["sw"])
        elif tag == "ellipse":
            cx, cy, rx, ry = _n(node.get("cx")), _n(node.get("cy")), _n(node.get("rx")), _n(node.get("ry"))
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=style["fill"], outline=style["stroke"], width=style["sw"])
        elif tag == "line":
            xy = [_n(node.get("x1")), _n(node.get("y1")), _n(node.get("x2")), _n(node.get("y2"))]
            draw.line(xy, fill=style["stroke"] or "black", width=style["sw"])
            if str(node.get("marker-end") or "").strip():
                _arrow_head(draw, xy, style["stroke"] or "black", style["sw"])
        elif tag in {"polygon", "polyline"}:
            pts = _points(node.get("points", ""))
            if len(pts) >= 2 and tag == "polygon":
                draw.polygon(pts, fill=style["fill"], outline=style["stroke"])
            elif len(pts) >= 2:
                draw.line(pts, fill=style["stroke"] or "black", width=style["sw"])
        elif tag == "text":
            x, y = _n(node.get("x")), _n(node.get("y"))
            text = "".join(node.itertext())
            font = _font(int(_n(node.get("font-size"), 12.0)))
            anchor = str(node.get("text-anchor") or "").strip()
            if anchor in {"middle", "center"}:
                box = draw.textbbox((0, 0), text, font=font)
                x -= (box[2] - box[0]) / 2.0
            elif anchor == "end":
                box = draw.textbbox((0, 0), text, font=font)
                x -= box[2] - box[0]
            draw.text((x, y), text, fill=style["fill"] or style["stroke"] or "black", font=font)
    image.save(output_path, format="PNG")


def _svg_gradient_map(root: ET.Element) -> dict[str, str]:
    paints: dict[str, str] = {}
    for node in root.iter():
        if _tag(node.tag) not in {"linearGradient", "radialGradient"}:
            continue
        gid = str(node.get("id") or "").strip()
        stops = [str(c.get("stop-color")).strip() for c in node if _tag(c.tag) == "stop" and c.get("stop-color")]
        if gid and stops:
            paints[gid] = stops[len(stops) // 2]
    return paints


def _svg_style(node: ET.Element, paint_map: dict[str, str]) -> dict[str, Any]:
    data: dict[str, str] = {}
    for chunk in (node.get("style") or "").split(";"):
        if ":" in chunk:
            key, value = chunk.split(":", 1)
            data[key.strip()] = value.strip()
    for key in ("fill", "stroke", "stroke-width"):
        if node.get(key) is not None:
            data[key] = str(node.get(key))
    return {
        "fill": _paint(data.get("fill"), paint_map),
        "stroke": _paint(data.get("stroke"), paint_map),
        "sw": max(1, int(_n(data.get("stroke-width"), 1.0))),
    }


# -- Pillow HTML fallback ----------------------------------------------------

def _render_html_pillow(code: str, output_path: Path, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for node in _HtmlNodes.parse(code):
        style = node["style"]
        x, y = _n(style.get("left") or node["attrs"].get("x"), 0.0), _n(style.get("top") or node["attrs"].get("y"), 0.0)
        w, h = _n(style.get("width"), 0.0), _n(style.get("height"), 0.0)
        fill = _paint(style.get("background") or style.get("background-color"), {})
        stroke = _border_color(style.get("border"))
        if w > 0 and h > 0 and (fill or stroke):
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=stroke)
        text = " ".join(str(node.get("text") or "").split())
        if not text:
            continue
        color = _paint(style.get("color"), {}) or "black"
        font = _font(int(_n(style.get("font-size"), _default_font_size(node["tag"]))))
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        tx = x + _n(style.get("padding-left"), _n(style.get("padding"), 0.0))
        align = str(style.get("text-align") or "").strip().lower()
        if align in {"center", "middle"} and w > 0:
            tx = x + max(0.0, (w - tw) / 2.0)
        elif align == "right" and w > 0:
            tx = x + max(0.0, w - tw)
        line_height = _n(style.get("line-height"), 0.0)
        ty = y + max(0.0, (line_height - th) / 2.0) if line_height > 0 else (y + max(0.0, (h - th) / 2.0) if h > 0 else y)
        draw.text((tx, ty), text, fill=color, font=font)
    image.save(output_path, format="PNG")


_HTML_DRAWABLE = {
    "div", "span", "p", "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer",
    "section", "article", "nav", "aside", "main", "label", "li", "button",
}


class _HtmlNodes(HTMLParser):
    """Flatten an HTML fragment into drawable {tag, attrs, style, text} nodes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []

    @classmethod
    def parse(cls, code: str) -> list[dict[str, Any]]:
        parser = cls()
        parser.feed(code)
        return [n for n in parser.out if n["tag"] in _HTML_DRAWABLE]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.casefold(): str(v or "") for k, v in attrs}
        node = {"tag": tag.casefold(), "attrs": attr, "style": _style_dict(attr.get("style", "")), "text": ""}
        self.out.append(node)
        self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        name = tag.casefold()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i]["tag"] == name:
                del self._stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1]["text"] = str(self._stack[-1].get("text") or "") + data


def _style_dict(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in style.split(";"):
        if ":" in chunk:
            key, value = chunk.split(":", 1)
            out[key.strip().casefold()] = value.strip()
    return out


# -- small shared helpers ----------------------------------------------------

def _tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _paint(value: str | None, paint_map: dict[str, str]) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text or text.casefold() in {"none", "transparent"}:
        return None
    match = re.match(r"url\(#([^)]+)\)", text)
    if match:
        return paint_map.get(match.group(1))
    return text


def _border_color(value: str | None) -> str | None:
    if not value:
        return None
    for part in reversed(str(value).split()):
        if part.casefold() not in {"solid", "dashed", "dotted", "none"} and not re.match(r"^\d", part):
            color = _paint(part, {})
            if color:
                return color
    return None


def _default_font_size(tag: str) -> float:
    return {"h1": 32.0, "h2": 24.0, "h3": 20.0}.get(tag, 16.0)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, max(1, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _arrow_head(draw: ImageDraw.ImageDraw, xy: list[float], fill: str, width: int) -> None:
    x1, y1, x2, y2 = xy
    angle = math.atan2(y2 - y1, x2 - x1)
    size = max(8.0, float(width) * 2.8)
    spread = math.radians(28)
    tip = (x2 + math.cos(angle) * size * 0.45, y2 + math.sin(angle) * size * 0.45)
    points = [tip]
    for d in (angle + math.pi - spread, angle + math.pi + spread):
        points.append((tip[0] + math.cos(d) * size, tip[1] + math.sin(d) * size))
    draw.polygon(points, fill=fill)


def _n(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def _points(value: str) -> list[tuple[float, float]]:
    nums = [_n(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
