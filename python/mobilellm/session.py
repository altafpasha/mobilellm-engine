"""
Multi-turn conversational session primitives for MobileLLM.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from mobilellm.models import GenerateResponse, SamplingParams, TokenChunk, TokenStr
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport


class Session:
    """Synchronous stateful inference session preserving KV-cache across turns."""

    def __init__(
        self,
        session_id: str,
        model_id: str,
        transport: BaseTransport,
    ):
        self.session_id = session_id
        self.model_id = model_id
        self._transport = transport
        self._is_closed = False

    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Generates text within this session context."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        return self._transport.generate(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        )

    def generate_stream(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> Iterator[TokenStr]:
        """Streams tokens within this session context."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        for chunk in self._transport.generate_stream(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        ):
            yield chunk.to_token_str()

    def generate_stream_chunks(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> Iterator[TokenChunk]:
        """Streams full TokenChunk objects within this session context."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        yield from self._transport.generate_stream(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        )

    def cancel(self) -> None:
        """Interrupts ongoing generation in this session."""
        self._transport.cancel(self.session_id)

    def close(self) -> None:
        """Destroys the session on the runtime and releases KV-cache."""
        if not self._is_closed:
            try:
                self._transport.destroy_session(self.session_id)
            finally:
                self._is_closed = True

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AsyncSession:
    """Asynchronous stateful inference session preserving KV-cache across turns."""

    def __init__(
        self,
        session_id: str,
        model_id: str,
        transport: AsyncBaseTransport,
    ):
        self.session_id = session_id
        self.model_id = model_id
        self._transport = transport
        self._is_closed = False

    async def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> GenerateResponse:
        """Generates text within this session context asynchronously."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        return await self._transport.generate(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        )

    async def generate_stream(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenStr]:
        """Streams tokens within this session context asynchronously."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        async for chunk in self._transport.generate_stream(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        ):
            yield chunk.to_token_str()

    async def generate_stream_chunks(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenChunk]:
        """Streams full TokenChunk objects within this session context asynchronously."""
        if self._is_closed:
            raise RuntimeError(f"Session '{self.session_id}' is closed")
        async for chunk in self._transport.generate_stream(
            prompt=prompt,
            model_id=self.model_id,
            params=params,
            session_id=self.session_id,
        ):
            yield chunk

    async def cancel(self) -> None:
        """Interrupts ongoing generation in this session."""
        await self._transport.cancel(self.session_id)

    async def close(self) -> None:
        """Destroys the session on the runtime and releases KV-cache."""
        if not self._is_closed:
            try:
                await self._transport.destroy_session(self.session_id)
            finally:
                self._is_closed = True

    async def __aenter__(self) -> "AsyncSession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
