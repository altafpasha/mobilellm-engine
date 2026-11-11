"""
MobileLLM Exception hierarchy and C-ABI error code mapping.
"""

from typing import Optional


class MobileLLMError(Exception):
    """Base exception for all MobileLLM runtime and SDK errors."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[MobileLLM Error {self.code}] {self.message}"


class ModelNotFoundError(MobileLLMError):
    """Raised when the specified model file does not exist or fails to open."""
    pass


class SessionNotFoundError(MobileLLMError):
    """Raised when referencing an invalid or expired inference session handle."""
    pass


class OutOfMemoryError(MobileLLMError):
    """Raised when memory budget or system RAM is exhausted."""
    pass


class ContextExhaustedError(MobileLLMError):
    """Raised when prompt and generation exceed maximum context token window."""
    pass


class BackendNotAvailableError(MobileLLMError):
    """Raised when requested compute backend (e.g. Vulkan) is not available."""
    pass


class InferenceCancelledError(MobileLLMError):
    """Raised when an active token generation was cooperatively cancelled."""
    pass


class AuthenticationError(MobileLLMError):
    """Raised when API key is missing or invalid on authenticated endpoints."""
    pass


class SecurityViolationError(MobileLLMError):
    """Raised when attempting forbidden operations (e.g. non-loopback bind without key)."""
    pass


def map_error_code(code: int, message: str) -> Optional[MobileLLMError]:
    """
    Maps native C-ABI status codes to typed Python exceptions.
    0 = OK (returns None)
    """
    if code == 0:
        return None
    elif code == 1:
        return ModelNotFoundError(message, code)
    elif code == 2:
        return SessionNotFoundError(message, code)
    elif code == 3:
        return OutOfMemoryError(message, code)
    elif code == 4:
        return ContextExhaustedError(message, code)
    elif code == 5:
        return BackendNotAvailableError(message, code)
    elif code == 6:
        return InferenceCancelledError(message, code)
    elif code == 7:
        return AuthenticationError(message, code)
    elif code == 8:
        return SecurityViolationError(message, code)
    else:
        return MobileLLMError(message, code)

