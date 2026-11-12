"""
Native in-process FFI transport binding directly to libmobilellm via C-ABI.
"""

import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from mobilellm.bindings.native_client import NativeClient, NativeSession
from mobilellm.exceptions import (
    ModelNotFoundError,
    SessionNotFoundError,
)
from mobilellm.models import (
    EngineStatus,
    GenerateResponse,
    GenerationMetrics,
    ModelConfig,
    ModelInfo,
    RuntimeHealth,
    SamplingParams,
    SessionConfig,
    TokenChunk,
)
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport

logger = logging.getLogger("mobilellm.transport.native")


class NativeTransport(BaseTransport):
    """Zero-network, direct in-process C-ABI transport linking libmobilellm."""

    def __init__(self, lib_path: str | None = None):
        self._lib_path = lib_path
        self._client: NativeClient | None = None
        self._models: dict[str, int] = {}  # model_id -> handle
        self._model_info: dict[str, ModelInfo] = {}
        self._sessions: dict[str, NativeSession] = {}  # session_id -> NativeSession
        self._default_sessions: dict[str, str] = {}  # model_id -> default session_id
        self._lock = threading.Lock()

    def _ensure_client(self) -> NativeClient:
        if self._client is None:
            self._client = NativeClient(self._lib_path)
        return self._client

    def connect(self) -> None:
        self._ensure_client()

    def close(self) -> None:
        with self._lock:
            for s in self._sessions.values():
                try:
                    s.close()
                except Exception:
                    pass
            self._sessions.clear()
            self._default_sessions.clear()

            if self._client is not None:
                self._client.close()
                self._client = None
            self._models.clear()
            self._model_info.clear()

    def health(self) -> RuntimeHealth:
        client = self._ensure_client()
        version = client.get_version()
        hw_raw = client.detect_hardware()
        backend = "CPU_NEON"
        try:
            hw = json.loads(hw_raw)
            backend = hw.get("arch", "CPU_NEON")
        except Exception:
            pass

        return RuntimeHealth(
            status="ok",
            version=version,
            uptime_seconds=0.0,
            backend=backend,
            memory_used_mb=0.0,
            memory_budget_mb=0.0,
            models_loaded=len(self._models),
        )

    def status(self) -> EngineStatus:
        client = self._ensure_client()
        version = client.get_version()
        hw_raw = client.detect_hardware()
        hw_dict: dict[str, Any] = {}
        try:
            hw_dict = json.loads(hw_raw)
        except Exception:
            hw_dict = {"raw": hw_raw}

        return EngineStatus(
            status="ok",
            version=version,
            hardware=hw_dict,
            models=list(self._models.keys()),
            active_sessions=len(self._sessions),
        )

    def detect_hardware(self) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            return json.loads(client.detect_hardware())
        except Exception:
            return {"device": "Host"}

    def list_models(self) -> list[ModelInfo]:
        return list(self._model_info.values())

    def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        client = self._ensure_client()
        if not os.path.isfile(config.model_path):
            raise ModelNotFoundError(f"Model file not found: {config.model_path}")

        m_id = Path(config.model_path).stem
        with self._lock:
            # Inspect metadata if supported
            meta = client.inspect_model(config.model_path)
            handle = client.load_model_handle(config)
            self._models[m_id] = handle

            info = ModelInfo(
                id=m_id,
                path=config.model_path,
                architecture=meta.get("architecture", "unknown"),
                quantization=meta.get("quantization", "unknown"),
                context_length=int(meta.get("context_length", config.context_length)),
                file_size_bytes=int(
                    meta.get("file_size_bytes", os.path.getsize(config.model_path))
                ),
                valid=True,
                loaded=True,
            )
            self._model_info[m_id] = info

            # Create initial default session
            s_cfg = session_config or SessionConfig(max_context_tokens=config.context_length)
            sess = client.create_session(handle, s_cfg)
            s_id = str(uuid.uuid4())
            self._sessions[s_id] = sess
            self._default_sessions[m_id] = s_id

            return info

    def unload_model(self, model_id: str) -> bool:
        client = self._ensure_client()
        with self._lock:
            handle = self._models.get(model_id)
            if handle is None:
                raise ModelNotFoundError(f"Model '{model_id}' is not loaded")

            # Close sessions attached to this model
            to_remove = [sid for sid, sess in self._sessions.items() if sess.model_handle == handle]
            for sid in to_remove:
                self._sessions[sid].close()
                del self._sessions[sid]

            if model_id in self._default_sessions:
                del self._default_sessions[model_id]

            client.unload_model(handle)
            del self._models[model_id]
            if model_id in self._model_info:
                del self._model_info[model_id]
            return True

    def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        client = self._ensure_client()
        with self._lock:
            handle = self._models.get(model_id)
            if handle is None:
                raise ModelNotFoundError(f"Model '{model_id}' is not loaded")

            s_cfg = config or SessionConfig()
            sess = client.create_session(handle, s_cfg)
            s_id = str(uuid.uuid4())
            self._sessions[s_id] = sess
            return s_id

    def destroy_session(self, session_id: str) -> bool:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            sess.close()
            del self._sessions[session_id]
            for m_id, ds_id in list(self._default_sessions.items()):
                if ds_id == session_id:
                    del self._default_sessions[m_id]
            return True

    def _get_session(self, model_id: str, session_id: str | None) -> NativeSession:
        if session_id:
            sess = self._sessions.get(session_id)
            if sess is None:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            return sess

        # Fall back to default session
        def_sid = self._default_sessions.get(model_id)
        if def_sid and def_sid in self._sessions:
            return self._sessions[def_sid]

        handle = self._models.get(model_id)
        if handle is None:
            raise ModelNotFoundError(f"Model '{model_id}' not loaded")

        client = self._ensure_client()
        sess = client.create_session(handle, SessionConfig())
        new_sid = str(uuid.uuid4())
        self._sessions[new_sid] = sess
        self._default_sessions[model_id] = new_sid
        return sess

    def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        sess = self._get_session(model_id, session_id)
        tokens: list[str] = []
        metrics_holder: list[GenerationMetrics | None] = [None]

        def _on_metrics(m: GenerationMetrics) -> None:
            metrics_holder[0] = m

        for chunk in sess.generate(prompt, params, on_metrics=_on_metrics):
            tokens.append(chunk.text)

        full_text = "".join(tokens)
        return GenerateResponse(
            text=full_text,
            model=model_id,
            finish_reason="stop",
            metrics=metrics_holder[0],
            session_id=session_id,
        )

    def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> Iterator[TokenChunk]:
        sess = self._get_session(model_id, session_id)
        yield from sess.generate(prompt, params)

    def cancel(self, gen_handle: str | None = None) -> None:
        with self._lock:
            for s in self._sessions.values():
                s.cancel()


class AsyncNativeTransport(AsyncBaseTransport):
    """Asynchronous wrapper for in-process NativeTransport executing on background threads."""

    def __init__(self, lib_path: str | None = None):
        self._sync = NativeTransport(lib_path)

    async def connect(self) -> None:
        await asyncio.to_thread(self._sync.connect)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def health(self) -> RuntimeHealth:
        return await asyncio.to_thread(self._sync.health)

    async def status(self) -> EngineStatus:
        return await asyncio.to_thread(self._sync.status)

    async def detect_hardware(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.detect_hardware)

    async def list_models(self) -> list[ModelInfo]:
        return await asyncio.to_thread(self._sync.list_models)

    async def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        return await asyncio.to_thread(self._sync.load_model, config, session_config)

    async def unload_model(self, model_id: str) -> bool:
        return await asyncio.to_thread(self._sync.unload_model, model_id)

    async def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        return await asyncio.to_thread(self._sync.create_session, model_id, config)

    async def destroy_session(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._sync.destroy_session, session_id)

    async def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        return await asyncio.to_thread(self._sync.generate, prompt, model_id, params, session_id)

    async def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[TokenChunk]:
        q: queue.Queue[Any | None] = queue.Queue()
        stop_event = threading.Event()

        def _producer() -> None:
            try:
                for chunk in self._sync.generate_stream(prompt, model_id, params, session_id):
                    if stop_event.is_set():
                        break
                    q.put(chunk)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        try:
            while True:
                item = await asyncio.to_thread(q.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            self._sync.cancel()

    async def cancel(self, gen_handle: str | None = None) -> None:
        await asyncio.to_thread(self._sync.cancel, gen_handle)
