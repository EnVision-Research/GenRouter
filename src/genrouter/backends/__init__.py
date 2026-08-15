"""Backend interfaces and real implementations."""

from genrouter.backends.chat import (
    ChatBackendSpec,
    ChatCompletionBackend,
    ChatRequestError,
    LocalTransformersChatBackend,
    build_llm_backend,
    build_mllm_backend,
    build_signature_llm_backend,
)
from genrouter.backends.embedding import (
    EmbeddingBackendSpec,
    EmbeddingRequestError,
    OpenAIEmbeddingBackend,
    build_embedding_backend,
)
from genrouter.backends.generator import (
    GeneratorRequestError,
    HttpGeneratorBackend,
    ModelScopeImageGeneratorBackend,
    ReferenceApiGeneratorBackend,
    build_generator_backend,
)
from genrouter.backends.scorer import MLLMScorerBackend, WiseScorerBackend, build_scorer_backend

__all__ = [
    "ChatBackendSpec",
    "ChatCompletionBackend",
    "ChatRequestError",
    "EmbeddingBackendSpec",
    "EmbeddingRequestError",
    "GeneratorRequestError",
    "HttpGeneratorBackend",
    "LocalTransformersChatBackend",
    "ModelScopeImageGeneratorBackend",
    "MLLMScorerBackend",
    "OpenAIEmbeddingBackend",
    "WiseScorerBackend",
    "ReferenceApiGeneratorBackend",
    "build_generator_backend",
    "build_embedding_backend",
    "build_llm_backend",
    "build_mllm_backend",
    "build_scorer_backend",
    "build_signature_llm_backend",
]
