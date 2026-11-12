"""
Unit tests for cancellation, cooperative preemption, and timeout handling.
"""

import pytest

from mobilellm.engine import Engine
from mobilellm.exceptions import InferenceCancelledError, MobileLLMTimeoutError
from mobilellm.transport.mock import AsyncMockTransport, MockTransport


def test_sync_streaming_cancellation() -> None:
    mock_transport = MockTransport(
        default_tokens=["Token1", "Token2", "Token3", "Token4"],
        latency_per_token=0.01,
    )
    with Engine(transport=mock_transport) as engine:
        model = engine.load_model("cancel_model.gguf")

        gen = model.generate_stream("Cancel test")
        first = next(gen)
        assert first == "Token1"

        # Signal cancellation
        mock_transport.cancel()

        with pytest.raises(InferenceCancelledError):
            next(gen)


@pytest.mark.asyncio
async def test_async_streaming_cancellation() -> None:
    mock_transport = AsyncMockTransport(
        default_tokens=["Token1", "Token2", "Token3", "Token4"],
        latency_per_token=0.01,
    )
    async with Engine.async_client(transport=mock_transport) as engine:
        model = await engine.load_model("cancel_model.gguf")

        tokens = []
        try:
            async for token in model.generate_stream("Cancel test"):
                tokens.append(token)
                if len(tokens) == 2:
                    await mock_transport.cancel()
        except InferenceCancelledError:
            pass

        assert len(tokens) == 2


def test_simulated_timeout_error() -> None:
    mock_transport = MockTransport(simulate_error=MobileLLMTimeoutError("Operation timed out", 408))
    engine = Engine(transport=mock_transport)
    with pytest.raises(MobileLLMTimeoutError, match="Operation timed out"):
        engine.connect()

