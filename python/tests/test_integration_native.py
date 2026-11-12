"""
End-to-end integration tests verifying Python SDK against real native C-ABI and GGUF model.
"""

import os

import pytest

from mobilellm.bindings.native_client import NativeClient
from mobilellm.engine import Engine
from mobilellm.models import ModelConfig, SamplingParams

MODEL_PATH = "models/smollm2-135m-instruct-q4_k_m.gguf"
LIB_PATH = "build_llama/runtime/libmobilellm.dylib"


@pytest.fixture(scope="module")
def check_environment() -> None:
    if not os.path.isfile(MODEL_PATH):
        pytest.skip(f"Model file {MODEL_PATH} not found; skipping native integration test")
    if not os.path.isfile(LIB_PATH):
        pytest.skip(f"Library {LIB_PATH} not found; skipping native integration test")


def test_native_client_direct_ffi(check_environment: None) -> None:
    with NativeClient(lib_path=LIB_PATH) as client:
        version = client.get_version()
        assert "0.1.0" in version

        hw = client.detect_hardware()
        assert len(hw) > 0

        meta = client.inspect_model(MODEL_PATH)
        assert meta.get("architecture") == "llama"
        assert meta.get("quantization") == "Q4_K_M"

        cfg = ModelConfig(model_path=MODEL_PATH, context_length=256, n_threads=2)
        with client.load_model(cfg) as session:
            tokens = []
            for chunk in session.generate("Hello", SamplingParams(max_tokens=4)):
                tokens.append(chunk.text)
            assert len(tokens) > 0


def test_engine_sync_native(check_environment: None) -> None:
    engine = Engine(use_native=True, lib_path=LIB_PATH)
    engine.connect()
    assert engine.health().status == "ok"

    model = engine.load_model(MODEL_PATH)
    assert model.info.architecture == "llama"

    # Unary generate
    resp = model.generate("The capital of France is", SamplingParams(max_tokens=4))
    assert len(resp.text) > 0

    # Stream generate
    streamed = list(model.generate_stream("The capital of Japan is", SamplingParams(max_tokens=4)))
    assert len(streamed) > 0
    assert isinstance(streamed[0].text, str)

    # Session test
    session = model.create_session()
    sess_resp = session.generate("Say hello", SamplingParams(max_tokens=4))
    assert len(sess_resp.text) > 0
    session.close()

    # Unload
    assert engine.unload_model(model)
    engine.close()


@pytest.mark.asyncio
async def test_engine_async_native(check_environment: None) -> None:
    async with Engine.async_client(use_native=True, lib_path=LIB_PATH) as engine:
        health = await engine.health()
        assert health.status == "ok"

        model = await engine.load_model(MODEL_PATH)

        # Async unary generate
        resp = await model.generate("The color of the sky is", SamplingParams(max_tokens=4))
        assert len(resp.text) > 0

        # Async stream generate
        tokens = []
        async for t in model.generate_stream("1 + 1 equals", SamplingParams(max_tokens=4)):
            tokens.append(t)
        assert len(tokens) > 0

        await model.unload()
