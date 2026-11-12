"""
MobileLLM Python SDK — Native Edge AI Inference & Local Loopback OpenAI Server.
"""

from mobilellm.bindings.native_client import NativeClient, NativeSession
from mobilellm.client import AsyncMobileLLMClient
from mobilellm.engine import AsyncEngine, Engine
from mobilellm.exceptions import (
    AuthenticationError,
    BackendNotAvailableError,
    ContextExhaustedError,
    InferenceCancelledError,
    InvalidRequestError,
    MobileLLMConnectionError,
    MobileLLMError,
    MobileLLMTimeoutError,
    ModelNotFoundError,
    OutOfMemoryError,
    SecurityViolationError,
    ServerRuntimeError,
    SessionNotFoundError,
)
from mobilellm.model import AsyncModel, Model
from mobilellm.models import (
    BackendType,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EngineStatus,
    GenerateResponse,
    GenerationMetrics,
    ModelCard,
    ModelConfig,
    ModelInfo,
    RuntimeHealth,
    SamplingParams,
    SessionConfig,
    TokenChunk,
    TokenStr,
)
from mobilellm.session import AsyncSession, Session
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport
from mobilellm.transport.http import AsyncHTTPTransport, HTTPTransport
from mobilellm.transport.mock import AsyncMockTransport, MockTransport
from mobilellm.transport.native import AsyncNativeTransport, NativeTransport

__version__ = "0.1.0"

__all__ = [
    # Top-level Engine & Models
    "Engine",
    "AsyncEngine",
    "Model",
    "AsyncModel",
    "Session",
    "AsyncSession",
    "NativeClient",
    "NativeSession",
    "AsyncMobileLLMClient",
    # Transports
    "BaseTransport",
    "AsyncBaseTransport",
    "HTTPTransport",
    "AsyncHTTPTransport",
    "NativeTransport",
    "AsyncNativeTransport",
    "MockTransport",
    "AsyncMockTransport",
    # Configuration and Data Types
    "BackendType",
    "ModelConfig",
    "SessionConfig",
    "SamplingParams",
    "TokenChunk",
    "TokenStr",
    "GenerateResponse",
    "GenerationMetrics",
    "ModelInfo",
    "RuntimeHealth",
    "EngineStatus",
    # OpenAI Schemas
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ModelCard",
    # Exceptions
    "MobileLLMError",
    "ModelNotFoundError",
    "SessionNotFoundError",
    "OutOfMemoryError",
    "ContextExhaustedError",
    "BackendNotAvailableError",
    "InferenceCancelledError",
    "AuthenticationError",
    "SecurityViolationError",
    "MobileLLMConnectionError",
    "MobileLLMTimeoutError",
    "InvalidRequestError",
    "ServerRuntimeError",
    "__version__",
]
