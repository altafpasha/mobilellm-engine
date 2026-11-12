"""
Unit tests for asynchronous Engine, Model, and AsyncMobileLLMClient with AsyncMockTransport.
"""

import pytest

from mobilellm.client import AsyncMobileLLMClient
from mobilellm.engine import Engine
from mobilellm.models import ChatMessage
from mobilellm.transport.mock import AsyncMockTransport


@pytest.mark.asyncio
async def test_async_engine_context_manager() -> None:
    mock_transport = AsyncMockTransport(default_tokens=["Async", " token", " stream"])
    async with Engine.async_client(transport=mock_transport) as engine:
        health = await engine.health()
        assert health.status == "ok"

        model = await engine.load_model("./models/model.gguf")
        assert model.id == "model"

        # Unary generation
        response = await model.generate("Hello")
        assert response.text == "Async token stream"

        # Streaming generation
        collected = []
        async for token in model.generate_stream("Streaming test"):
            collected.append(token)
        assert len(collected) == 3
        assert collected[0] == "Async"
        assert collected[0].text == "Async"

        # Unload
        res = await model.unload()
        assert res


@pytest.mark.asyncio
async def test_async_mobilellm_client_openai_compatibility() -> None:
    mock_transport = AsyncMockTransport(default_tokens=["OpenAI", " compatible", " chunk"])
    async with AsyncMobileLLMClient(transport=mock_transport) as client:
        # Load model
        await client.load_model("./models/gpt.gguf")

        # Chat completions unary
        resp = await client.chat.completions.create(
            model="gpt",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        assert not isinstance(resp, str)
        assert resp.choices[0].message.content == "OpenAI compatible chunk"
        assert resp.usage is not None
        assert resp.usage["completion_tokens"] == 3

        # Chat completions stream
        stream_resp = await client.chat.completions.create(
            model="gpt",
            messages=[ChatMessage(role="user", content="Hi stream")],
            stream=True,
        )
        chunks = []
        async for chunk in stream_resp:
            chunks.append(chunk.choices[0].delta["content"])
        assert len(chunks) == 3
        assert "".join(chunks) == "OpenAI compatible chunk"
