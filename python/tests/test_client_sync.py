"""
Unit tests for synchronous Engine, Model, and Session workflows with MockTransport.
"""

import pytest

from mobilellm.engine import Engine
from mobilellm.models import SamplingParams
from mobilellm.transport.mock import MockTransport


def test_engine_connect_and_health() -> None:
    mock_transport = MockTransport()
    engine = Engine(transport=mock_transport)
    engine.connect()
    assert mock_transport.is_connected

    health = engine.health()
    assert health.status == "ok"
    assert health.backend == "MOCK"

    status = engine.status()
    assert status.status == "ok"

    caps = engine.capabilities()
    assert caps["backend"] == "MOCK"

    engine.close()
    assert not mock_transport.is_connected


def test_engine_context_manager() -> None:
    mock_transport = MockTransport()
    with Engine(transport=mock_transport) as engine:
        assert mock_transport.is_connected
        assert engine.health().status == "ok"
    assert not mock_transport.is_connected


def test_model_load_generate_unload() -> None:
    mock_transport = MockTransport(default_tokens=["Python", " is", " simple."])
    with Engine(transport=mock_transport) as engine:
        model = engine.load_model("test_model.gguf")
        assert model.id == "test_model"

        # Unary generation
        resp = model.generate("Explain Python")
        assert resp.text == "Python is simple."
        assert resp.metrics is not None
        assert resp.metrics.completion_tokens == 3

        # Streaming generation
        tokens = list(model.generate_stream("Explain ML", SamplingParams(max_tokens=2)))
        assert len(tokens) == 2
        assert tokens[0] == "Python"
        assert tokens[0].text == "Python"

        # Unload model
        unloaded = engine.unload_model(model)
        assert unloaded

        # Generating on unloaded model raises RuntimeError
        with pytest.raises(RuntimeError, match="unloaded"):
            model.generate("test")


def test_session_lifecycle() -> None:
    mock_transport = MockTransport(default_tokens=["Hello", " world"])
    with Engine(transport=mock_transport) as engine:
        model = engine.load_model("dialogue.gguf")
        session = model.create_session()

        resp = session.generate("Turn 1")
        assert resp.text == "Hello world"

        tokens = list(session.generate_stream("Turn 2"))
        assert len(tokens) == 2

        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.generate("Turn 3")
