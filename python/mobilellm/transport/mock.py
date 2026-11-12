"""
In-memory Mock Transports for deterministic unit and isolation testing.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from mobilellm.exceptions import (
    InferenceCancelledError,
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


class MockTransport(BaseTransport):
    """In-memory mock transport for sync unit testing."""

    def __init__(
        self,
        default_tokens: list[str] | None = None,
        simulate_error: Exception | None = None,
        latency_per_token: float = 0.0,
    ):
        self.default_tokens = default_tokens or ["Mock", " response", " from", " MobileLLM."]
        self.simulate_error = simulate_error
        self.latency_per_token = latency_per_token
        self.is_connected = False
        self.cancelled = False
        self.models: dict[str, ModelInfo] = {}
        self.sessions: dict[str, str] = {}  # session_id -> model_id

    def connect(self) -> None:
        if self.simulate_error:
            raise self.simulate_error
        self.is_connected = True

    def close(self) -> None:
        self.is_connected = False
        self.models.clear()
        self.sessions.clear()

    def health(self) -> RuntimeHealth:
        if self.simulate_error:
            raise self.simulate_error
        return RuntimeHealth(
            status="ok",
            version="0.1.0-mock",
            uptime_seconds=123.4,
            backend="MOCK",
            memory_used_mb=256.0,
            memory_budget_mb=1024.0,
            models_loaded=len(self.models),
        )

    def status(self) -> EngineStatus:
        if self.simulate_error:
            raise self.simulate_error
        return EngineStatus(
            status="ok",
            version="0.1.0-mock",
            hardware={"backend": "MOCK", "simulated": True},
            models=list(self.models.keys()),
            active_sessions=len(self.sessions),
        )

    def detect_hardware(self) -> dict[str, Any]:
        return {"backend": "MOCK", "cores": 8}

    def list_models(self) -> list[ModelInfo]:
        return list(self.models.values())

    def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        if self.simulate_error:
            raise self.simulate_error
        m_id = config.model_path.split("/")[-1].replace(".gguf", "")
        info = ModelInfo(
            id=m_id,
            path=config.model_path,
            architecture="mock-arch",
            quantization="Q4_K_M",
            context_length=config.context_length,
            file_size_bytes=1000000,
            valid=True,
            loaded=True,
        )
        self.models[m_id] = info
        # Default session
        s_id = str(uuid.uuid4())
        self.sessions[s_id] = m_id
        return info

    def unload_model(self, model_id: str) -> bool:
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model '{model_id}' not loaded")
        del self.models[model_id]
        to_del = [sid for sid, mid in self.sessions.items() if mid == model_id]
        for sid in to_del:
            del self.sessions[sid]
        return True

    def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model '{model_id}' not loaded")
        s_id = str(uuid.uuid4())
        self.sessions[s_id] = model_id
        return s_id

    def destroy_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        if self.simulate_error:
            raise self.simulate_error
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        if session_id and session_id not in self.sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        p = params or SamplingParams()
        tokens = self.default_tokens[: p.max_tokens]
        return GenerateResponse(
            text="".join(tokens),
            model=model_id,
            finish_reason="stop",
            metrics=GenerationMetrics(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(tokens),
                time_to_first_token_ms=10.0,
                tokens_per_second=50.0,
                peak_rss_mb=128.0,
            ),
            session_id=session_id,
        )

    def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> Iterator[TokenChunk]:
        if self.simulate_error:
            raise self.simulate_error
        if model_id not in self.models:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        if session_id and session_id not in self.sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        self.cancelled = False
        p = params or SamplingParams()
        tokens = self.default_tokens[: p.max_tokens]
        for idx, t in enumerate(tokens):
            if self.cancelled:
                raise InferenceCancelledError("Generation cancelled cooperatively")
            if self.latency_per_token > 0:
                time.sleep(self.latency_per_token)
            yield TokenChunk(text=t, token_id=100 + idx, is_special=False, logprob=-0.5)

    def cancel(self, gen_handle: str | None = None) -> None:
        self.cancelled = True


class AsyncMockTransport(AsyncBaseTransport):
    """In-memory mock transport for async unit testing."""

    def __init__(
        self,
        default_tokens: list[str] | None = None,
        simulate_error: Exception | None = None,
        latency_per_token: float = 0.0,
    ):
        self._sync = MockTransport(
            default_tokens=default_tokens,
            simulate_error=simulate_error,
            latency_per_token=latency_per_token,
        )

    async def connect(self) -> None:
        self._sync.connect()

    async def close(self) -> None:
        self._sync.close()

    async def health(self) -> RuntimeHealth:
        return self._sync.health()

    async def status(self) -> EngineStatus:
        return self._sync.status()

    async def detect_hardware(self) -> dict[str, Any]:
        return self._sync.detect_hardware()

    async def list_models(self) -> list[ModelInfo]:
        return self._sync.list_models()

    async def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        return self._sync.load_model(config, session_config)

    async def unload_model(self, model_id: str) -> bool:
        return self._sync.unload_model(model_id)

    async def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        return self._sync.create_session(model_id, config)

    async def destroy_session(self, session_id: str) -> bool:
        return self._sync.destroy_session(session_id)

    async def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        return self._sync.generate(prompt, model_id, params, session_id)

    async def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[TokenChunk]:
        for chunk in self._sync.generate_stream(prompt, model_id, params, session_id):
            if self._sync.latency_per_token > 0:
                await asyncio.sleep(self._sync.latency_per_token)
            yield chunk

    async def cancel(self, gen_handle: str | None = None) -> None:
        self._sync.cancel(gen_handle)
