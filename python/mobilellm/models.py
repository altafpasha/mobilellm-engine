"""
Domain data models, request/response schemas, and OpenAI wire specifications.
"""

import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class BackendType(IntEnum):
    """Compute backend targets matching native C-ABI."""

    AUTO = 0
    CPU_NEON = 1
    VULKAN = 2
    VENDOR_NPU = 3
    MOCK = 99


@dataclass
class SamplingParams:
    """Hyperparameters governing autoregressive token sampling."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0.0, 1.0]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p must be in [0.0, 1.0]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
        }


@dataclass
class ModelConfig:
    """Configuration for loading model weights into native memory."""

    model_path: str
    preferred_backend: BackendType = BackendType.AUTO
    context_length: int = 2048
    n_threads: int = field(default_factory=lambda: max(1, os.cpu_count() or 4))
    use_mmap: bool = True
    memory_budget_mb: int = 0

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path cannot be empty")
        if self.context_length < 16:
            raise ValueError("context_length must be >= 16")
        if self.n_threads < 1:
            raise ValueError("n_threads must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "preferred_backend": int(self.preferred_backend),
            "context_length": self.context_length,
            "n_threads": self.n_threads,
            "use_mmap": self.use_mmap,
            "memory_budget_mb": self.memory_budget_mb,
        }


@dataclass
class SessionConfig:
    """Session configuration for paged KV-cache allocation."""

    max_context_tokens: int = 2048
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.max_context_tokens < 16:
            raise ValueError("max_context_tokens must be >= 16")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
        }


class TokenStr(str):
    """String token yielded by streaming generation with embedded metadata."""

    token_id: int
    is_special: bool
    logprob: float

    def __new__(
        cls,
        text: str,
        token_id: int = 0,
        is_special: bool = False,
        logprob: float = 0.0,
    ) -> "TokenStr":
        s = super().__new__(cls, text)
        s.token_id = token_id
        s.is_special = is_special
        s.logprob = logprob
        return s

    @property
    def text(self) -> str:
        return str(self)


@dataclass
class TokenChunk:
    """Single emitted token in streaming generation."""

    text: str
    token_id: int
    is_special: bool = False
    logprob: float = 0.0

    def __str__(self) -> str:
        return self.text

    def to_token_str(self) -> TokenStr:
        return TokenStr(self.text, self.token_id, self.is_special, self.logprob)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_id": self.token_id,
            "is_special": self.is_special,
            "logprob": self.logprob,
        }


@dataclass
class GenerationMetrics:
    """Telemetry telemetry reported at the conclusion of generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    peak_rss_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "tokens_per_second": self.tokens_per_second,
            "peak_rss_mb": self.peak_rss_mb,
        }


@dataclass
class GenerateResponse:
    """Unified response for single-turn or multi-turn generation."""

    text: str
    model: str
    finish_reason: str = "stop"
    metrics: GenerationMetrics | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "session_id": self.session_id,
        }


@dataclass
class ModelInfo:
    """Metadata regarding a model known or loaded in the runtime."""

    id: str
    path: str
    architecture: str = "unknown"
    quantization: str = "unknown"
    context_length: int = 2048
    file_size_bytes: int = 0
    valid: bool = True
    loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "file_size_bytes": self.file_size_bytes,
            "valid": self.valid,
            "loaded": self.loaded,
        }


@dataclass
class RuntimeHealth:
    """Runtime health and diagnostic telemetry."""

    status: str = "ok"
    version: str = "0.1.0-dev"
    uptime_seconds: float = 0.0
    backend: str = "CPU_NEON"
    memory_used_mb: float = 0.0
    memory_budget_mb: float = 0.0
    models_loaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "backend": self.backend,
            "memory_used_mb": self.memory_used_mb,
            "memory_budget_mb": self.memory_budget_mb,
            "models_loaded": self.models_loaded,
        }


@dataclass
class EngineStatus:
    """Engine operational status and hardware capabilities."""

    status: str
    version: str
    hardware: dict[str, Any]
    models: list[str] = field(default_factory=list)
    active_sessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "hardware": self.hardware,
            "models": self.models,
            "active_sessions": self.active_sessions,
        }


# --- OpenAI Compatible Wire Schemas ---


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    stream: bool = False
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatCompletionRequest":
        raw_msgs = data.get("messages", [])
        messages = [
            ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in raw_msgs
        ]
        stop = data.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        return cls(
            model=data.get("model", "default"),
            messages=messages,
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            top_k=int(data.get("top_k", 40)),
            repetition_penalty=float(data.get("repetition_penalty", 1.1)),
            max_tokens=int(data.get("max_tokens", 512)),
            stream=bool(data.get("stream", False)),
            stop=list(stop),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "stop": self.stop,
        }


@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionResponse:
    id: str
    choices: list[ChatCompletionChoice]
    model: str = "default"
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage": self.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


@dataclass
class ChatCompletionChunkChoice:
    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "delta": self.delta,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionChunk:
    id: str
    choices: list[ChatCompletionChunkChoice]
    model: str = "default"
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
        }


@dataclass
class ModelCard:
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "mobilellm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }
