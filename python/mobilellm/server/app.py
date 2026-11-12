"""
Standalone OpenAI-compatible loopback HTTP and SSE streaming server for MobileLLM.
"""

import json
import logging
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from mobilellm.exceptions import SecurityViolationError
from mobilellm.models import (
    BackendType,
    ChatCompletionRequest,
    ModelConfig,
    SamplingParams,
    SessionConfig,
)
from mobilellm.server.security import (
    resolve_and_validate_model_path,
    validate_auth_token,
    validate_bind_security,
)
from mobilellm.transport.native import NativeTransport

logger = logging.getLogger("mobilellm.server")


class MobileLLMRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing OpenAI REST and SSE wire specifications."""

    server: "MobileLLMServer"

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "code": status}})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def _check_auth(self) -> bool:
        if not self.server.api_key:
            return True
        auth_header = self.headers.get("Authorization")
        if not validate_auth_token(auth_header, self.server.api_key):
            self._send_error(401, "Unauthorized: Invalid or missing API key Bearer token.")
            return False
        return True

    def _parse_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        if not self._check_auth():
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/v1/health", "/health"):
            health = self.server.transport.health()
            uptime = time.time() - self.server.start_time
            self._send_json(
                200,
                {
                    "status": health.status,
                    "version": health.version,
                    "uptime_seconds": uptime,
                    "backend": health.backend,
                    "models_loaded": health.models_loaded,
                    "hardware": self.server.transport.detect_hardware(),
                },
            )
        elif path in ("/v1/models", "/models"):
            models = self.server.transport.list_models()
            data = [
                {
                    "id": m.id,
                    "object": "model",
                    "created": int(self.server.start_time),
                    "owned_by": "mobilellm",
                    "path": m.path,
                    "architecture": m.architecture,
                    "quantization": m.quantization,
                    "context_length": m.context_length,
                }
                for m in models
            ]
            self._send_json(200, {"object": "list", "data": data})
        elif path in ("/v1/metrics", "/metrics"):
            status = self.server.transport.status()
            self._send_json(
                200,
                {
                    "models_count": len(status.models),
                    "active_sessions": status.active_sessions,
                    "uptime_seconds": time.time() - self.server.start_time,
                },
            )
        else:
            self._send_error(404, f"Endpoint '{path}' not found")

    def do_POST(self) -> None:
        if not self._check_auth():
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._parse_body()

        if path in ("/v1/models/load", "/models/load"):
            model_path = body.get("model_path") or body.get("path")
            if not model_path:
                self._send_error(400, "Missing required field: model_path")
                return

            try:
                validated_path = resolve_and_validate_model_path(
                    model_path,
                    allowed_dirs=self.server.allowed_model_dirs,
                )
            except SecurityViolationError as e:
                self._send_error(403, f"Security violation: {e}")
                return
            except Exception as e:
                self._send_error(400, f"Invalid model path: {e}")
                return

            ctx_len = int(body.get("context_length", 2048))
            backend = int(body.get("backend", 0))
            n_threads = int(body.get("n_threads", 4))
            try:
                b_type = BackendType(backend) if backend in (0, 1, 2, 3, 99) else BackendType.AUTO
                cfg = ModelConfig(
                    model_path=str(validated_path),
                    context_length=ctx_len,
                    preferred_backend=b_type,
                    n_threads=n_threads,
                )
                info = self.server.transport.load_model(cfg)
                self._send_json(
                    200,
                    {
                        "id": info.id,
                        "object": "model",
                        "architecture": info.architecture,
                        "quantization": info.quantization,
                        "context_length": info.context_length,
                        "status": "loaded",
                    },
                )
            except Exception as e:
                self._send_error(500, f"Failed to load model: {e}")

        elif path in ("/v1/models/unload", "/models/unload"):
            model_id = body.get("model") or body.get("model_id") or body.get("id")
            if not model_id:
                self._send_error(400, "Missing required field: model")
                return
            try:
                self.server.transport.unload_model(model_id)
                self._send_json(200, {"id": model_id, "status": "unloaded"})
            except Exception as e:
                self._send_error(404, str(e))

        elif path in ("/v1/sessions", "/sessions"):
            model_id = body.get("model") or body.get("model_id", "default")
            max_ctx = int(body.get("max_context_tokens", 2048))
            try:
                s_id = self.server.transport.create_session(
                    model_id,
                    SessionConfig(max_context_tokens=max_ctx),
                )
                self._send_json(200, {"session_id": s_id, "model": model_id})
            except Exception as e:
                self._send_error(500, f"Failed to create session: {e}")

        elif path in (
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/completions",
            "/completions",
        ):
            self._handle_completions(path, body)

        elif path in ("/v1/cancel", "/cancel"):
            handle = body.get("handle")
            self.server.transport.cancel(handle)
            self._send_json(200, {"status": "cancelled"})

        else:
            self._send_error(404, f"Endpoint '{path}' not found")

    def do_DELETE(self) -> None:
        if not self._check_auth():
            return

        parsed = urllib.parse.urlparse(self.path)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        # Expect /v1/sessions/{id} or /sessions/{id}
        session_id = None
        if len(parts) >= 2 and parts[-2] == "sessions":
            session_id = parts[-1]

        if session_id:
            res = self.server.transport.destroy_session(session_id)
            if res:
                self._send_json(200, {"deleted": True, "id": session_id})
            else:
                self._send_error(404, f"Session '{session_id}' not found")
        else:
            self._send_error(404, "Invalid session endpoint")

    def _handle_completions(self, path: str, body: dict[str, Any]) -> None:
        model_id = body.get("model", "default")
        stream = bool(body.get("stream", False))

        # Check if model exists, or if there's only 1 loaded model use it
        loaded = self.server.transport.list_models()
        if not loaded:
            self._send_error(404, "No models loaded in MobileLLM runtime. Load a model first.")
            return

        target_model = model_id
        if model_id == "default" or model_id not in [m.id for m in loaded]:
            target_model = loaded[0].id

        # Extract prompt
        if "messages" in body:
            req = ChatCompletionRequest.from_dict(body)
            # Compose prompt from messages
            prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
            params = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                stop=req.stop,
            )
        else:
            prompt = str(body.get("prompt", ""))
            params = SamplingParams(
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 40)),
                repetition_penalty=float(body.get("repetition_penalty", 1.1)),
                max_tokens=int(body.get("max_tokens", 512)),
            )

        session_id = body.get("session_id")

        if not stream:
            # Unary JSON
            try:
                gen_resp = self.server.transport.generate(
                    prompt=prompt,
                    model_id=target_model,
                    params=params,
                    session_id=session_id,
                )
                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                usage = {
                    "prompt_tokens": gen_resp.metrics.prompt_tokens if gen_resp.metrics else 0,
                    "completion_tokens": gen_resp.metrics.completion_tokens
                    if gen_resp.metrics
                    else 0,
                    "total_tokens": (
                        (gen_resp.metrics.prompt_tokens + gen_resp.metrics.completion_tokens)
                        if gen_resp.metrics
                        else 0
                    ),
                    "ttft_ms": gen_resp.metrics.time_to_first_token_ms if gen_resp.metrics else 0.0,
                    "tps": gen_resp.metrics.tokens_per_second if gen_resp.metrics else 0.0,
                }
                res_obj = {
                    "id": cmpl_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": gen_resp.text},
                            "finish_reason": gen_resp.finish_reason,
                        }
                    ],
                    "usage": usage,
                }
                self._send_json(200, res_obj)
            except Exception as e:
                self._send_error(500, f"Generation failed: {e}")
        else:
            # Streaming SSE (text/event-stream)
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for chunk in self.server.transport.generate_stream(
                    prompt=prompt,
                    model_id=target_model,
                    params=params,
                    session_id=session_id,
                ):
                    chunk_data = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": target_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text, "role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
                    self.wfile.flush()

                # Send terminal stop chunk
                stop_data = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(stop_data)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                logger.error(f"SSE Streaming error: {e}")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard library access logs in production
        pass


class MobileLLMServer(ThreadingHTTPServer):
    """Multi-threaded MobileLLM HTTP daemon with strict loopback security enforcement."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        api_key: str | None = None,
        allow_remote: bool = False,
        lib_path: str | None = None,
        initial_model: str | None = None,
        allowed_model_dirs: Sequence[str | Path] | None = None,
    ):
        validate_bind_security(host, allow_remote=allow_remote, api_key=api_key)
        super().__init__((host, port), MobileLLMRequestHandler)

        self.host = host
        self.port = port
        self.api_key = api_key
        self.start_time = time.time()
        self.allowed_model_dirs = (
            [Path(d).resolve() for d in allowed_model_dirs]
            if allowed_model_dirs
            else None
        )
        self.transport = NativeTransport(lib_path=lib_path)
        self.transport.connect()

        if initial_model:
            validated = resolve_and_validate_model_path(
                initial_model, allowed_dirs=self.allowed_model_dirs
            )
            cfg = ModelConfig(model_path=str(validated))
            self.transport.load_model(cfg)

    def shutdown_server(self) -> None:
        self.transport.close()
        self.shutdown()
        self.server_close()


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    allow_remote: bool = False,
    lib_path: str | None = None,
    initial_model: str | None = None,
    allowed_model_dirs: Sequence[str | Path] | None = None,
) -> MobileLLMServer:
    """Instantiates and starts MobileLLMServer."""
    server = MobileLLMServer(
        host=host,
        port=port,
        api_key=api_key,
        allow_remote=allow_remote,
        lib_path=lib_path,
        initial_model=initial_model,
        allowed_model_dirs=allowed_model_dirs,
    )
    return server
