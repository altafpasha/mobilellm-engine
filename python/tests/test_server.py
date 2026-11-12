"""
Unit and integration tests for local loopback HTTP server and security policies.
"""

import io
import socket
import threading
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mobilellm.engine import Engine
from mobilellm.exceptions import (
    AuthenticationError,
    SecurityViolationError,
)
from mobilellm.models import SamplingParams
from mobilellm.server.app import MobileLLMRequestHandler, MobileLLMServer, start_server
from mobilellm.server.security import (
    is_loopback_address,
    resolve_and_validate_model_path,
    validate_auth_token,
    validate_bind_security,
)


def check_loopback_socket_allowed() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.listen(1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
                c.settimeout(0.5)
                c.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_loopback_detection() -> None:
    assert is_loopback_address("127.0.0.1")
    assert is_loopback_address("localhost")
    assert is_loopback_address("::1")
    assert not is_loopback_address("0.0.0.0")
    assert not is_loopback_address("192.168.1.100")
    assert not is_loopback_address("10.0.0.1")


def test_security_violation_policies() -> None:
    # 1. 0.0.0.0 without allow_remote
    with pytest.raises(SecurityViolationError, match="forbidden by default"):
        validate_bind_security("0.0.0.0", allow_remote=False)

    # 2. 0.0.0.0 with allow_remote but without api_key
    with pytest.raises(
        SecurityViolationError, match="strictly requires configuring a non-empty --api-key"
    ):
        validate_bind_security("0.0.0.0", allow_remote=True, api_key=None)

    with pytest.raises(
        SecurityViolationError, match="strictly requires configuring a non-empty --api-key"
    ):
        validate_bind_security("0.0.0.0", allow_remote=True, api_key="   ")

    # 3. 0.0.0.0 with allow_remote AND api_key passes
    validate_bind_security("0.0.0.0", allow_remote=True, api_key="secret123")

    # 4. 127.0.0.1 passes without extra flags
    validate_bind_security("127.0.0.1")


def test_bearer_token_validation() -> None:
    # No secret configured -> open
    assert validate_auth_token(None, None)
    assert validate_auth_token("Bearer foo", None)

    # Secret configured
    secret = "sk-mobilellm-test"
    assert validate_auth_token(f"Bearer {secret}", secret)
    assert validate_auth_token(f"bearer {secret}", secret)
    assert not validate_auth_token("Bearer wrong-key", secret)
    assert not validate_auth_token(None, secret)
    assert not validate_auth_token("", secret)


class MockSocket:
    """Simulates a client socket for testing the HTTP request handler in-memory."""

    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode: str, *args: object, **kwargs: object) -> io.BytesIO:
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, b: bytes) -> None:
        self.wfile.write(b)


def test_request_handler_in_memory() -> None:
    """Tests MobileLLMRequestHandler logic in-memory without requiring TCP sockets."""
    mock_server = MagicMock()
    mock_server.api_key = "test-key"
    mock_server.start_time = time.time()
    mock_server.transport = MagicMock()
    mock_server.transport.health.return_value = MagicMock(
        status="ok",
        version="0.1.0-dev",
        backend="ARM64",
        models_loaded=1,
    )
    mock_server.transport.detect_hardware.return_value = {"arch": "ARM64"}

    # 1. Unauthenticated request -> 401
    sock = MockSocket(b"GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    MobileLLMRequestHandler(sock, ("127.0.0.1", 12345), mock_server)  # type: ignore[arg-type]
    sock.wfile.seek(0)
    out = sock.wfile.read().decode("utf-8", errors="replace")
    assert "401 Unauthorized" in out

    # 2. Authenticated GET /v1/health -> 200
    sock = MockSocket(
        b"GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer test-key\r\n\r\n"
    )
    MobileLLMRequestHandler(sock, ("127.0.0.1", 12345), mock_server)  # type: ignore[arg-type]
    sock.wfile.seek(0)
    out = sock.wfile.read().decode("utf-8", errors="replace")
    assert "200 OK" in out
    assert '"status": "ok"' in out


def test_path_traversal_validation(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    valid_model = models_dir / "valid.gguf"
    valid_model.write_bytes(b"dummy")

    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "passwords.txt"
    secret_file.write_text("classified")

    allowed = [models_dir]

    # 1. Valid model inside allowed dir passes
    p = resolve_and_validate_model_path(valid_model, allowed_dirs=allowed)
    assert p == valid_model.resolve()

    # Relative to allowed dir
    p_rel = resolve_and_validate_model_path("valid.gguf", allowed_dirs=allowed)
    assert p_rel == valid_model.resolve()

    # 2. Directory traversal attempt rejected
    with pytest.raises(SecurityViolationError, match="outside allowed directories"):
        resolve_and_validate_model_path("../secret/passwords.txt", allowed_dirs=allowed)

    with pytest.raises(SecurityViolationError, match="outside allowed directories"):
        resolve_and_validate_model_path(
            models_dir / "../secret/passwords.txt", allowed_dirs=allowed
        )

    # 3. Absolute path outside allowed dir rejected
    with pytest.raises(SecurityViolationError, match="outside allowed directories"):
        resolve_and_validate_model_path(secret_file, allowed_dirs=allowed)

    # 4. Null byte injection rejected
    with pytest.raises(SecurityViolationError, match="Null bytes forbidden"):
        resolve_and_validate_model_path("valid.gguf\0.evil", allowed_dirs=allowed)

    # 5. Symlink escape rejected
    symlink_model = models_dir / "escape.gguf"
    try:
        symlink_model.symlink_to(secret_file)
        with pytest.raises(
            SecurityViolationError, match="(outside allowed directories|Symlink escape rejected)"
        ):
            resolve_and_validate_model_path(symlink_model, allowed_dirs=allowed)
    except OSError:
        pass


def test_server_blocks_model_traversal_via_http(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    valid_model = models_dir / "valid.gguf"
    valid_model.write_bytes(b"GGUF")

    mock_server = MagicMock()
    mock_server.api_key = "test-key"
    mock_server.start_time = time.time()
    mock_server.allowed_model_dirs = [models_dir]
    mock_server.transport = MagicMock()
    mock_server.transport.load_model.return_value = MagicMock(
        id="valid.gguf",
        architecture="llama",
        quantization="Q4_0",
        context_length=2048,
    )

    # 1. Traversal attempt -> 403 Forbidden
    payload = b'{"model_path": "../../etc/passwd"}'
    req = (
        b"POST /v1/models/load HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer test-key\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
    )
    sock = MockSocket(req)
    MobileLLMRequestHandler(sock, ("127.0.0.1", 12345), mock_server)  # type: ignore[arg-type]
    sock.wfile.seek(0)
    out = sock.wfile.read().decode("utf-8", errors="replace")
    assert "403 Forbidden" in out
    assert "Security violation" in out
    mock_server.transport.load_model.assert_not_called()

    # 2. Legitimate model under allowed dir -> 200 OK
    payload_valid = b'{"model_path": "valid.gguf"}'
    req_valid = (
        b"POST /v1/models/load HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer test-key\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload_valid)).encode() + b"\r\n\r\n" + payload_valid
    )
    sock2 = MockSocket(req_valid)
    MobileLLMRequestHandler(sock2, ("127.0.0.1", 12345), mock_server)  # type: ignore[arg-type]
    sock2.wfile.seek(0)
    out2 = sock2.wfile.read().decode("utf-8", errors="replace")
    assert "200 OK" in out2
    assert '"status": "loaded"' in out2
    mock_server.transport.load_model.assert_called_once()


@pytest.fixture
def running_server() -> Generator[MobileLLMServer, None, None]:
    if not check_loopback_socket_allowed():
        pytest.skip("Loopback TCP socket connection not permitted in current sandbox environment")

    port = find_free_port()
    server = start_server(
        host="127.0.0.1",
        port=port,
        api_key="test-secret-key",
        initial_model="models/smollm2-135m-instruct-q4_k_m.gguf",
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)
    yield server
    server.shutdown_server()


def test_http_transport_against_live_server(running_server: MobileLLMServer) -> None:
    endpoint = f"http://127.0.0.1:{running_server.port}"

    # Test 1: Unauthenticated request should fail with AuthenticationError
    unauth_engine = Engine(endpoint=endpoint, api_key=None)
    with pytest.raises(AuthenticationError):
        unauth_engine.connect()

    # Test 2: Authenticated connection succeeds
    auth_engine = Engine(endpoint=endpoint, api_key="test-secret-key")
    auth_engine.connect()
    health = auth_engine.health()
    assert health.status == "ok"

    # Test 3: List models
    models = auth_engine.list_models()
    assert len(models) >= 1
    assert "smollm2" in models[0].id.lower()

    # Test 4: Unary generation
    model = auth_engine.load_model("models/smollm2-135m-instruct-q4_k_m.gguf")
    resp = model.generate("What is 2+2?", SamplingParams(max_tokens=4))
    assert len(resp.text) > 0

    # Test 5: SSE Streaming generation
    tokens = list(model.generate_stream("What is 3+3?", SamplingParams(max_tokens=4)))
    assert len(tokens) > 0

    auth_engine.close()
