"""
Domain data models, request/response schemas, and OpenAI wire specifications.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import os
import time
from typing import Any, Dict, List, Optional


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


@dataclass
class SessionConfig:
    """Session configuration for paged KV-cache allocation."""
    max_context_tokens: int = 2048
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.max_context_tokens < 16:
            raise ValueError("max_context_tokens must be >= 16")


@dataclass
class TokenChunk:
    """Single emitted token in streaming generation."""
    text: str
    token_id: int
    is_special: bool = False
    logprob: float = 0.0


@dataclass
class GenerationMetrics:
    """Telemetry telemetry reported at the conclusion of generation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    peak_rss_mb: float = 0.0


# --- OpenAI Compatible Wire Schemas ---


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    stream: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletionRequest":
        raw_msgs = data.get("messages", [])
        messages = [
            ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", "")
            )
            for m in raw_msgs
        ]
        return cls(
            model=data.get("model", "default"),
            messages=messages,
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            top_k=int(data.get("top_k", 40)),
            repetition_penalty=float(data.get("repetition_penalty", 1.1)),
            max_tokens=int(data.get("max_tokens", 512)),
            stream=bool(data.get("stream", False)),
        )


@dataclass
class ModelCard:
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "mobilellm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }

