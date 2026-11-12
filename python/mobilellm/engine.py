"""
MobileLLM Engine client providing synchronous and asynchronous developer interfaces.
"""

import logging
from typing import Any

from mobilellm.model import AsyncModel, Model
from mobilellm.models import (
    EngineStatus,
    ModelConfig,
    ModelInfo,
    RuntimeHealth,
    SessionConfig,
)
from mobilellm.session import AsyncSession, Session
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport
from mobilellm.transport.http import AsyncHTTPTransport, HTTPTransport
from mobilellm.transport.native import AsyncNativeTransport, NativeTransport

logger = logging.getLogger("mobilellm.engine")


class Engine:
    """
    Primary interface for connecting to and interacting with MobileLLM Engine.
    Supports local HTTP daemon, Android loopback via ADB, and direct in-process FFI.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        use_native: bool = False,
        lib_path: str | None = None,
        transport: BaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._custom_headers = headers

        if transport is not None:
            self._transport = transport
        elif use_native or (endpoint is not None and endpoint.startswith("native://")):
            self._transport = NativeTransport(lib_path=lib_path)
        elif endpoint is not None:
            self._transport = HTTPTransport(
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
                headers=headers,
            )
        else:
            # Default auto-discovery: try native if library available, else default loopback
            try:
                self._transport = NativeTransport(lib_path=lib_path)
            except Exception:
                self.endpoint = "http://127.0.0.1:8080"
                self._transport = HTTPTransport(
                    endpoint=self.endpoint,
                    api_key=api_key,
                    timeout=timeout,
                    headers=headers,
                )

        self._loaded_models: dict[str, Model] = {}
        self._is_connected = False

    def connect(self) -> "Engine":
        """Establishes connection to the engine or initializes runtime."""
        self._transport.connect()
        self._is_connected = True
        return self

    def health(self) -> RuntimeHealth:
        """Returns runtime health, memory, and uptime metrics."""
        return self._transport.health()

    def status(self) -> EngineStatus:
        """Returns current engine status, loaded models, and active sessions."""
        return self._transport.status()

    def capabilities(self) -> dict[str, Any]:
        """Returns hardware features (NEON, Vulkan, NPU)."""
        return self._transport.detect_hardware()

    def list_models(self) -> list[ModelInfo]:
        """Lists models currently loaded or available."""
        return self._transport.list_models()

    def load_model(
        self,
        model_path_or_config: str | ModelConfig,
        config: ModelConfig | None = None,
        session_config: SessionConfig | None = None,
    ) -> Model:
        """
        Loads a GGUF model into runtime memory.

        Example:
            model = engine.load_model("./models/model.gguf")
        """
        if isinstance(model_path_or_config, ModelConfig):
            cfg = model_path_or_config
        elif config is not None:
            cfg = config
            cfg.model_path = model_path_or_config
        else:
            cfg = ModelConfig(model_path=model_path_or_config)

        info = self._transport.load_model(cfg, session_config)
        m = Model(model_id=info.id, path=cfg.model_path, transport=self._transport, info=info)
        self._loaded_models[info.id] = m
        return m

    def unload_model(self, model_or_id: str | Model) -> bool:
        """Unloads a model from runtime memory."""
        m_id = model_or_id.id if isinstance(model_or_id, Model) else model_or_id
        res = self._transport.unload_model(m_id)
        if isinstance(model_or_id, Model):
            model_or_id._is_unloaded = True
        if m_id in self._loaded_models:
            self._loaded_models[m_id]._is_unloaded = True
            del self._loaded_models[m_id]
        return res

    def create_session(
        self,
        model_or_id: str | Model,
        config: SessionConfig | None = None,
    ) -> Session:
        """Creates a dedicated multi-turn session for the given model."""
        m_id = model_or_id.id if isinstance(model_or_id, Model) else model_or_id
        s_id = self._transport.create_session(m_id, config)
        return Session(session_id=s_id, model_id=m_id, transport=self._transport)

    def close(self) -> None:
        """Closes all active sessions and disconnects transport."""
        for m in list(self._loaded_models.values()):
            try:
                m.unload()
            except Exception:
                pass
        self._loaded_models.clear()
        self._transport.close()
        self._is_connected = False

    def __enter__(self) -> "Engine":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @classmethod
    def async_client(
        cls,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        use_native: bool = False,
        lib_path: str | None = None,
        transport: AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> "AsyncEngine":
        """
        Creates an asynchronous engine client.

        Example:
            async with Engine.async_client() as engine:
                model = await engine.load_model("model.gguf")
        """
        return AsyncEngine(
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            use_native=use_native,
            lib_path=lib_path,
            transport=transport,
            headers=headers,
        )


class AsyncEngine:
    """Asynchronous MobileLLM Engine client."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        use_native: bool = False,
        lib_path: str | None = None,
        transport: AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self._custom_headers = headers

        if transport is not None:
            self._transport = transport
        elif use_native or (endpoint is not None and endpoint.startswith("native://")):
            self._transport = AsyncNativeTransport(lib_path=lib_path)
        elif endpoint is not None:
            self._transport = AsyncHTTPTransport(
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
                headers=headers,
            )
        else:
            try:
                self._transport = AsyncNativeTransport(lib_path=lib_path)
            except Exception:
                self.endpoint = "http://127.0.0.1:8080"
                self._transport = AsyncHTTPTransport(
                    endpoint=self.endpoint,
                    api_key=api_key,
                    timeout=timeout,
                    headers=headers,
                )

        self._loaded_models: dict[str, AsyncModel] = {}
        self._is_connected = False

    async def connect(self) -> "AsyncEngine":
        """Establishes connection to the engine or initializes runtime."""
        await self._transport.connect()
        self._is_connected = True
        return self

    async def health(self) -> RuntimeHealth:
        """Returns runtime health, memory, and uptime metrics."""
        return await self._transport.health()

    async def status(self) -> EngineStatus:
        """Returns current engine status, loaded models, and active sessions."""
        return await self._transport.status()

    async def capabilities(self) -> dict[str, Any]:
        """Returns hardware features (NEON, Vulkan, NPU)."""
        return await self._transport.detect_hardware()

    async def list_models(self) -> list[ModelInfo]:
        """Lists models currently loaded or available."""
        return await self._transport.list_models()

    async def load_model(
        self,
        model_path_or_config: str | ModelConfig,
        config: ModelConfig | None = None,
        session_config: SessionConfig | None = None,
    ) -> AsyncModel:
        """Loads a GGUF model into runtime memory asynchronously."""
        if isinstance(model_path_or_config, ModelConfig):
            cfg = model_path_or_config
        elif config is not None:
            cfg = config
            cfg.model_path = model_path_or_config
        else:
            cfg = ModelConfig(model_path=model_path_or_config)

        info = await self._transport.load_model(cfg, session_config)
        m = AsyncModel(model_id=info.id, path=cfg.model_path, transport=self._transport, info=info)
        self._loaded_models[info.id] = m
        return m

    async def unload_model(self, model_or_id: str | AsyncModel) -> bool:
        """Unloads a model from runtime memory asynchronously."""
        m_id = model_or_id.id if isinstance(model_or_id, AsyncModel) else model_or_id
        res = await self._transport.unload_model(m_id)
        if isinstance(model_or_id, AsyncModel):
            model_or_id._is_unloaded = True
        if m_id in self._loaded_models:
            self._loaded_models[m_id]._is_unloaded = True
            del self._loaded_models[m_id]
        return res

    async def create_session(
        self,
        model_or_id: str | AsyncModel,
        config: SessionConfig | None = None,
    ) -> AsyncSession:
        """Creates a dedicated multi-turn session for the given model asynchronously."""
        m_id = model_or_id.id if isinstance(model_or_id, AsyncModel) else model_or_id
        s_id = await self._transport.create_session(m_id, config)
        return AsyncSession(session_id=s_id, model_id=m_id, transport=self._transport)

    async def close(self) -> None:
        """Closes all active sessions and disconnects transport asynchronously."""
        for m in list(self._loaded_models.values()):
            try:
                await m.unload()
            except Exception:
                pass
        self._loaded_models.clear()
        await self._transport.close()
        self._is_connected = False

    async def __aenter__(self) -> "AsyncEngine":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
