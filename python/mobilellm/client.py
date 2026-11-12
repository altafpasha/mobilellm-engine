"""
Asynchronous HTTP/SSE Client for MobileLLM with OpenAI-compatible chat interface.
"""

from collections.abc import AsyncIterator
from typing import Any

from mobilellm.engine import AsyncEngine
from mobilellm.exceptions import InvalidRequestError
from mobilellm.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelConfig,
    RuntimeHealth,
    SamplingParams,
)
from mobilellm.transport.base import AsyncBaseTransport


class _ChatCompletions:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def create(
        self,
        model: str,
        messages: list[ChatMessage | dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        max_tokens: int = 512,
        stream: bool = False,
        stop: list[str] | None = None,
        session_id: str | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        """Creates a chat completion matching the OpenAI client specification."""
        clean_msgs: list[ChatMessage] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                clean_msgs.append(m)
            elif isinstance(m, dict):
                clean_msgs.append(
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                )
            else:
                raise InvalidRequestError("Message must be ChatMessage or dict")

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            stop=stop or [],
        )

        # Extract prompt from last message or entire history
        prompt_parts: list[str] = []
        for msg in clean_msgs:
            prompt_parts.append(f"{msg.role}: {msg.content}")
        prompt_text = "\n".join(prompt_parts)

        if not stream:
            gen_resp = await self._engine._transport.generate(
                prompt=prompt_text,
                model_id=model,
                params=params,
                session_id=session_id,
            )
            from mobilellm.models import ChatCompletionChoice

            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=gen_resp.text),
                finish_reason=gen_resp.finish_reason,
            )
            usage = None
            if gen_resp.metrics:
                usage = {
                    "prompt_tokens": gen_resp.metrics.prompt_tokens,
                    "completion_tokens": gen_resp.metrics.completion_tokens,
                    "total_tokens": gen_resp.metrics.prompt_tokens
                    + gen_resp.metrics.completion_tokens,
                }
            return ChatCompletionResponse(
                id=f"chatcmpl-{model}",
                choices=[choice],
                model=model,
                usage=usage,
            )

        async def _stream_generator() -> AsyncIterator[ChatCompletionChunk]:
            chunk_id = f"chatcmpl-{model}"
            async for chunk in self._engine._transport.generate_stream(
                prompt=prompt_text,
                model_id=model,
                params=params,
                session_id=session_id,
            ):
                choice = ChatCompletionChunkChoice(
                    index=0,
                    delta={"content": chunk.text, "role": "assistant"},
                    finish_reason=None,
                )
                yield ChatCompletionChunk(
                    id=chunk_id,
                    choices=[choice],
                    model=model,
                )

        return _stream_generator()


class _Chat:
    def __init__(self, engine: AsyncEngine):
        self.completions = _ChatCompletions(engine)


class _Models:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def list(self) -> list[ModelCard]:
        infos = await self._engine.list_models()
        return [ModelCard(id=info.id) for info in infos]


class AsyncMobileLLMClient:
    """
    High-level Async client implementing OpenAI SDK compatible namespaces (`chat.completions`, `models`).
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: AsyncBaseTransport | None = None,
    ):
        self._engine = AsyncEngine(
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )
        self.chat = _Chat(self._engine)
        self.models = _Models(self._engine)

    async def connect(self) -> "AsyncMobileLLMClient":
        await self._engine.connect()
        return self

    async def health(self) -> RuntimeHealth:
        return await self._engine.health()

    async def load_model(self, model_path: str, context_length: int = 2048) -> Any:
        return await self._engine.load_model(
            ModelConfig(model_path=model_path, context_length=context_length)
        )

    async def close(self) -> None:
        await self._engine.close()

    async def __aenter__(self) -> "AsyncMobileLLMClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
