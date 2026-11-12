"""
Unit tests for MobileLLM Exception hierarchy and error mapping.
Verifies synchronization with C++ StatusCode definitions (Codes 1 - 12, 99).
"""

from mobilellm.exceptions import (
    AuthenticationError,
    BackendExecutionError,
    BackendInitializationError,
    BackendNotAvailableError,
    ContextExceededError,
    ContextExhaustedError,
    InferenceCancelledError,
    InvalidArgumentError,
    InvalidRequestError,
    MobileLLMConnectionError,
    MobileLLMError,
    MobileLLMTimeoutError,
    ModelNotFoundError,
    ModelNotLoadedError,
    NotImplementedMobileLLMError,
    OutOfMemoryError,
    SecurityViolationError,
    ServerRuntimeError,
    SessionNotFoundError,
    UnsupportedFormatError,
    map_error_code,
    map_http_status,
)


def test_exception_inheritance() -> None:
    assert issubclass(InvalidArgumentError, MobileLLMError)
    assert issubclass(ModelNotFoundError, MobileLLMError)
    assert issubclass(UnsupportedFormatError, MobileLLMError)
    assert issubclass(OutOfMemoryError, MobileLLMError)
    assert issubclass(BackendInitializationError, MobileLLMError)
    assert issubclass(BackendExecutionError, MobileLLMError)
    assert issubclass(SessionNotFoundError, MobileLLMError)
    assert issubclass(ModelNotLoadedError, MobileLLMError)
    assert issubclass(InferenceCancelledError, MobileLLMError)
    assert issubclass(ContextExhaustedError, MobileLLMError)
    assert issubclass(MobileLLMTimeoutError, MobileLLMError)
    assert issubclass(NotImplementedMobileLLMError, MobileLLMError)
    assert issubclass(ServerRuntimeError, MobileLLMError)
    assert issubclass(AuthenticationError, MobileLLMError)
    assert issubclass(SecurityViolationError, MobileLLMError)
    assert issubclass(MobileLLMConnectionError, MobileLLMError)


def test_aliases_backward_compatibility() -> None:
    assert InvalidRequestError is InvalidArgumentError
    assert BackendNotAvailableError is BackendInitializationError
    assert ContextExceededError is ContextExhaustedError


def test_error_str_formatting() -> None:
    err = ModelNotFoundError("File not found", code=2)
    assert "[MobileLLM Error 2] File not found" == str(err)
    assert err.code == 2
    assert err.message == "File not found"


def test_map_all_error_codes() -> None:
    # 0 = OK (returns None)
    assert map_error_code(0, "OK") is None

    # 1 = INVALID_ARGUMENT
    e1 = map_error_code(1, "Invalid parameter")
    assert isinstance(e1, InvalidArgumentError)
    assert e1.code == 1

    # 2 = FILE_NOT_FOUND
    e2 = map_error_code(2, "Model missing")
    assert isinstance(e2, ModelNotFoundError)
    assert e2.code == 2

    # 3 = UNSUPPORTED_FORMAT
    e3 = map_error_code(3, "Corrupt GGUF")
    assert isinstance(e3, UnsupportedFormatError)
    assert e3.code == 3

    # 4 = OUT_OF_MEMORY
    e4 = map_error_code(4, "RAM exhausted")
    assert isinstance(e4, OutOfMemoryError)
    assert e4.code == 4

    # 5 = BACKEND_INITIALIZATION_FAILED
    e5 = map_error_code(5, "Backend init failed")
    assert isinstance(e5, BackendInitializationError)
    assert e5.code == 5

    # 6 = BACKEND_EXECUTION_ERROR
    e6 = map_error_code(6, "Decode error")
    assert isinstance(e6, BackendExecutionError)
    assert e6.code == 6

    # 7 = SESSION_NOT_FOUND
    e7 = map_error_code(7, "Session expired")
    assert isinstance(e7, SessionNotFoundError)
    assert e7.code == 7

    # 8 = MODEL_NOT_LOADED
    e8 = map_error_code(8, "Model unloaded")
    assert isinstance(e8, ModelNotLoadedError)
    assert e8.code == 8

    # 9 = CANCELLED
    e9 = map_error_code(9, "Generation cancelled")
    assert isinstance(e9, InferenceCancelledError)
    assert e9.code == 9

    # 10 = CONTEXT_EXCEEDED
    e10 = map_error_code(10, "Context window full")
    assert isinstance(e10, ContextExhaustedError)
    assert e10.code == 10

    # 11 = TIMEOUT
    e11 = map_error_code(11, "Operation timed out")
    assert isinstance(e11, MobileLLMTimeoutError)
    assert e11.code == 11

    # 12 = NOT_IMPLEMENTED
    e12 = map_error_code(12, "Feature not implemented")
    assert isinstance(e12, NotImplementedMobileLLMError)
    assert e12.code == 12

    # 99 = INTERNAL_ERROR
    e99 = map_error_code(99, "Internal crash")
    assert isinstance(e99, ServerRuntimeError)
    assert e99.code == 99

    # Unknown codes map safely to base MobileLLMError
    e_unk = map_error_code(999, "Unknown error")
    assert type(e_unk) is MobileLLMError
    assert e_unk.code == 999


def test_map_http_status() -> None:
    assert isinstance(map_http_status(400, "Bad params"), InvalidRequestError)
    assert isinstance(map_http_status(401, "Unauthorized"), AuthenticationError)
    assert isinstance(map_http_status(403, "Forbidden"), SecurityViolationError)
    assert isinstance(map_http_status(404, "Model not found"), ModelNotFoundError)
    assert isinstance(map_http_status(404, "Session not found"), SessionNotFoundError)
    assert isinstance(map_http_status(408, "Timed out"), MobileLLMTimeoutError)
    assert isinstance(map_http_status(504, "Gateway timeout"), MobileLLMTimeoutError)
    assert isinstance(map_http_status(499, "Client closed"), InferenceCancelledError)
    assert isinstance(map_http_status(503, "Low memory budget"), OutOfMemoryError)
    assert isinstance(map_http_status(500, "Crash"), ServerRuntimeError)
