"""
MobileLLM Exception hierarchy and C-ABI error code mapping.
Authoritative source of truth matching runtime/include/mobilellm/types.h StatusCode enum.
"""

from typing import Any


class MobileLLMError(Exception):
    """Base exception for all MobileLLM runtime and SDK errors."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[MobileLLM Error {self.code}] {self.message}"


# --- C-ABI Status Code Exceptions (Codes 1 - 12, 99) ---


class InvalidArgumentError(MobileLLMError):
    """Raised when client passes invalid parameters or malformed payloads (Code 1)."""


class ModelNotFoundError(MobileLLMError):
    """Raised when the specified model file does not exist or fails to open (Code 2)."""


class UnsupportedFormatError(MobileLLMError):
    """Raised when model file is truncated, corrupt, or unsupported format/magic (Code 3)."""


class OutOfMemoryError(MobileLLMError):
    """Raised when memory budget or system RAM is exhausted (Code 4)."""


class BackendInitializationError(MobileLLMError):
    """Raised when compute backend (e.g. Vulkan, CPU/NEON) fails to initialize (Code 5)."""


class BackendExecutionError(MobileLLMError):
    """Raised when backend execution or inference decode fails (Code 6)."""


class SessionNotFoundError(MobileLLMError):
    """Raised when referencing an invalid or expired inference session handle (Code 7)."""


class ModelNotLoadedError(MobileLLMError):
    """Raised when attempting an operation on an unloaded model handle (Code 8)."""


class InferenceCancelledError(MobileLLMError):
    """Raised when an active token generation was cooperatively cancelled (Code 9)."""


class ContextExhaustedError(MobileLLMError):
    """Raised when prompt and generation exceed maximum context token window (Code 10)."""


class MobileLLMTimeoutError(MobileLLMError):
    """Raised when an operation times out (Code 11)."""


class NotImplementedMobileLLMError(MobileLLMError):
    """Raised when an operation is not implemented by the backend or runtime (Code 12)."""


class ServerRuntimeError(MobileLLMError):
    """Raised when MobileLLM server/runtime encounters an unhandled internal error (Code 99)."""


# --- Non-C-ABI SDK & Security Exceptions ---


class AuthenticationError(MobileLLMError):
    """Raised when API key is missing or invalid on authenticated endpoints."""


class SecurityViolationError(MobileLLMError):
    """Raised when attempting forbidden operations (e.g. non-loopback bind without key, path traversal)."""


class MobileLLMConnectionError(MobileLLMError):
    """Raised when network connection to MobileLLM daemon fails."""


# --- Aliases for Backward Compatibility ---
InvalidRequestError = InvalidArgumentError
InvalidModelError = UnsupportedFormatError
BackendNotAvailableError = BackendInitializationError
ContextExceededError = ContextExhaustedError
CancelledError = InferenceCancelledError
ConnectionError = MobileLLMConnectionError  # type: ignore[assignment]
TimeoutError = MobileLLMTimeoutError  # type: ignore[assignment]
InternalRuntimeError = ServerRuntimeError
MobileLLMNotImplementedError = NotImplementedMobileLLMError


# Authoritative mapping from C-ABI StatusCode to Exception type
NATIVE_STATUS_CODE_MAP: dict[int, type[MobileLLMError]] = {
    1: InvalidArgumentError,
    2: ModelNotFoundError,
    3: UnsupportedFormatError,
    4: OutOfMemoryError,
    5: BackendInitializationError,
    6: BackendExecutionError,
    7: SessionNotFoundError,
    8: ModelNotLoadedError,
    9: InferenceCancelledError,
    10: ContextExhaustedError,
    11: MobileLLMTimeoutError,
    12: NotImplementedMobileLLMError,
    99: ServerRuntimeError,
}


def map_error_code(code: int, message: str) -> MobileLLMError | None:
    """
    Maps native C-ABI status codes to typed Python exceptions according to C++ StatusCode enum:
    0  = OK (returns None)
    1  = INVALID_ARGUMENT -> InvalidArgumentError
    2  = FILE_NOT_FOUND -> ModelNotFoundError
    3  = UNSUPPORTED_FORMAT -> UnsupportedFormatError
    4  = OUT_OF_MEMORY -> OutOfMemoryError
    5  = BACKEND_INITIALIZATION_FAILED -> BackendInitializationError
    6  = BACKEND_EXECUTION_ERROR -> BackendExecutionError
    7  = SESSION_NOT_FOUND -> SessionNotFoundError
    8  = MODEL_NOT_LOADED -> ModelNotLoadedError
    9  = CANCELLED -> InferenceCancelledError
    10 = CONTEXT_EXCEEDED -> ContextExhaustedError
    11 = TIMEOUT -> MobileLLMTimeoutError
    12 = NOT_IMPLEMENTED -> NotImplementedMobileLLMError
    99 = INTERNAL_ERROR -> ServerRuntimeError
    other -> MobileLLMError
    """
    if code == 0:
        return None
    exc_cls = NATIVE_STATUS_CODE_MAP.get(code, MobileLLMError)
    return exc_cls(message, code)


def map_http_status(status_code: int, detail: str | dict[str, Any]) -> MobileLLMError:
    """
    Maps HTTP response status codes to typed SDK exceptions.
    """
    msg = (
        detail
        if isinstance(detail, str)
        else str(detail.get("error", detail.get("detail", detail)))
    )
    if status_code == 400:
        return InvalidRequestError(msg, status_code)
    elif status_code == 401:
        return AuthenticationError(msg, status_code)
    elif status_code == 403:
        return SecurityViolationError(msg, status_code)
    elif status_code == 404:
        if "session" in msg.lower():
            return SessionNotFoundError(msg, status_code)
        return ModelNotFoundError(msg, status_code)
    elif status_code in (408, 504):
        return MobileLLMTimeoutError(msg, status_code)
    elif status_code == 499:
        return InferenceCancelledError(msg, status_code)
    elif status_code == 503 and "memory" in msg.lower():
        return OutOfMemoryError(msg, status_code)
    elif status_code >= 500:
        return ServerRuntimeError(msg, status_code)
    return MobileLLMError(msg, status_code)
