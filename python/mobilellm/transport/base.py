"""
Abstract transport interfaces defining communication contracts for MobileLLM.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from mobilellm.models import (
    EngineStatus,
    GenerateResponse,
    ModelConfig,
    ModelInfo,
    RuntimeHealth,
    SamplingParams,
    SessionConfig,
    TokenChunk,
)


class BaseTransport(ABC):
    """Synchronous transport interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establishes connection to the engine or verifies runtime availability."""

    @abstractmethod
    def close(self) -> None:
        """Closes connections and frees allocated resources."""

    @abstractmethod
    def health(self) -> RuntimeHealth:
        """Retrieves runtime health and memory telemetry."""

    @abstractmethod
    def status(self) -> EngineStatus:
        """Retrieves runtime status and detected hardware features."""

    @abstractmethod
    def detect_hardware(self) -> dict[str, Any]:
        """Returns detected compute hardware capabilities."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Lists models currently known or loaded in the runtime."""

    @abstractmethod
    def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        """Loads a model into runtime memory."""

    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        """Unloads a model from runtime memory."""

    @abstractmethod
    def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        """Creates a stateful multi-turn inference session and returns session ID."""

    @abstractmethod
    def destroy_session(self, session_id: str) -> bool:
        """Destroys an active inference session."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        """Executes full unary text generation."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> Iterator[TokenChunk]:
        """Executes streaming text generation, yielding token chunks."""

    @abstractmethod
    def cancel(self, gen_handle: str | None = None) -> None:
        """Requests cancellation of an ongoing generation."""


class AsyncBaseTransport(ABC):
    """Asynchronous transport interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Establishes connection to the engine or verifies runtime availability."""

    @abstractmethod
    async def close(self) -> None:
        """Closes connections and frees allocated resources."""

    @abstractmethod
    async def health(self) -> RuntimeHealth:
        """Retrieves runtime health and memory telemetry."""

    @abstractmethod
    async def status(self) -> EngineStatus:
        """Retrieves runtime status and detected hardware features."""

    @abstractmethod
    async def detect_hardware(self) -> dict[str, Any]:
        """Returns detected compute hardware capabilities."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Lists models currently known or loaded in the runtime."""

    @abstractmethod
    async def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        """Loads a model into runtime memory."""

    @abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """Unloads a model from runtime memory."""

    @abstractmethod
    async def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        """Creates a stateful multi-turn inference session and returns session ID."""

    @abstractmethod
    async def destroy_session(self, session_id: str) -> bool:
        """Destroys an active inference session."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        """Executes full unary text generation."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Executes streaming text generation, yielding token chunks."""

    @abstractmethod
    async def cancel(self, gen_handle: str | None = None) -> None:
        """Requests cancellation of an ongoing generation."""
