"""
Model abstraction managing loaded weights and inference requests.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from mobilellm.models import (
    ChatMessage,
    GenerateResponse,
    ModelInfo,
    SamplingParams,
    SessionConfig,
    TokenChunk,
    TokenStr,
)
from mobilellm.session import AsyncSession, Session
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport


def _format_messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Formats chat messages into a standard prompt if raw text is needed."""
    lines: list[str] = []
    for msg in messages:
        if msg.role == "system":
            lines.append(f"<|system|>\n{msg.content}\n")
        elif msg.role == "user":
            lines.append(f"<|user|>\n{msg.content}\n")
        elif msg.role == "assistant":
            lines.append(f"<|assistant|>\n{msg.content}\n")
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


class Model:
    """Active model handle ready to execute text generation."""

    def __init__(
        self,
        model_id: str,
        path: str,
        transport: BaseTransport,
        info: ModelInfo | None = None,
    ):
        self.id = model_id
        self.path = path
        self._transport = transport
        self.info = info or ModelInfo(id=model_id, path=path, loaded=True)
        self._is_unloaded = False

    def generate(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Executes unary text generation."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        return self._transport.generate(p_text, self.id, params)

    def generate_stream(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> Iterator[TokenStr]:
        """Streams generated tokens yielding TokenStr instances."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        for chunk in self._transport.generate_stream(p_text, self.id, params):
            yield chunk.to_token_str()

    def generate_stream_chunks(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> Iterator[TokenChunk]:
        """Streams full TokenChunk instances including logprobs and token IDs."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        yield from self._transport.generate_stream(p_text, self.id, params)

    def chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Multi-turn conversation generation."""
        return self.generate(messages, params)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> Iterator[TokenStr]:
        """Streaming multi-turn conversation generation."""
        yield from self.generate_stream(messages, params)

    def create_session(self, config: SessionConfig | None = None) -> Session:
        """Creates a dedicated stateful conversation session for this model."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        s_id = self._transport.create_session(self.id, config)
        return Session(session_id=s_id, model_id=self.id, transport=self._transport)

    def unload(self) -> bool:
        """Unloads model weights from memory."""
        if not self._is_unloaded:
            res = self._transport.unload_model(self.id)
            self._is_unloaded = True
            return res
        return True

    def close(self) -> None:
        self.unload()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()


class AsyncModel:
    """Asynchronous model handle executing text generation."""

    def __init__(
        self,
        model_id: str,
        path: str,
        transport: AsyncBaseTransport,
        info: ModelInfo | None = None,
    ):
        self.id = model_id
        self.path = path
        self._transport = transport
        self.info = info or ModelInfo(id=model_id, path=path, loaded=True)
        self._is_unloaded = False

    async def generate(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Executes unary text generation asynchronously."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        return await self._transport.generate(p_text, self.id, params)

    async def generate_stream(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenStr]:
        """Streams generated tokens asynchronously yielding TokenStr instances."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        async for chunk in self._transport.generate_stream(p_text, self.id, params):
            yield chunk.to_token_str()

    async def generate_stream_chunks(
        self,
        prompt: str | list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Streams full TokenChunk instances asynchronously."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        p_text = prompt if isinstance(prompt, str) else _format_messages_to_prompt(prompt)
        async for chunk in self._transport.generate_stream(p_text, self.id, params):
            yield chunk

    async def chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Multi-turn conversation generation asynchronously."""
        return await self.generate(messages, params)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenStr]:
        """Streaming multi-turn conversation generation asynchronously."""
        async for token in self.generate_stream(messages, params):
            yield token

    async def create_session(self, config: SessionConfig | None = None) -> AsyncSession:
        """Creates a dedicated stateful conversation session for this model."""
        if self._is_unloaded:
            raise RuntimeError(f"Model '{self.id}' has been unloaded")
        s_id = await self._transport.create_session(self.id, config)
        return AsyncSession(session_id=s_id, model_id=self.id, transport=self._transport)

    async def unload(self) -> bool:
        """Unloads model weights from memory asynchronously."""
        if not self._is_unloaded:
            res = await self._transport.unload_model(self.id)
            self._is_unloaded = True
            return res
        return True

    async def close(self) -> None:
        await self.unload()

    async def __aenter__(self) -> "AsyncModel":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.unload()
