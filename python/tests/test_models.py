"""
Unit tests for data models, sampling parameters, and wire schemas.
"""

import pytest

from mobilellm.models import (
    BackendType,
    ChatCompletionRequest,
    ChatMessage,
    GenerateResponse,
    GenerationMetrics,
    ModelConfig,
    SamplingParams,
    SessionConfig,
    TokenChunk,
)


def test_sampling_params_defaults() -> None:
    p = SamplingParams()
    assert p.temperature == 0.7
    assert p.top_p == 0.9
    assert p.top_k == 40
    assert p.min_p == 0.05
    assert p.repetition_penalty == 1.1
    assert p.max_tokens == 512
    assert p.stop == []


def test_sampling_params_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.1)

    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=0.0)

    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=1.5)

    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=0)

    with pytest.raises(ValueError, match="repetition_penalty"):
        SamplingParams(repetition_penalty=0.9)

    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)


def test_model_config_validation() -> None:
    with pytest.raises(ValueError, match="model_path cannot be empty"):
        ModelConfig(model_path="")

    with pytest.raises(ValueError, match="context_length"):
        ModelConfig(model_path="dummy.gguf", context_length=8)

    with pytest.raises(ValueError, match="n_threads"):
        ModelConfig(model_path="dummy.gguf", n_threads=0)

    cfg = ModelConfig(model_path="model.gguf", context_length=4096)
    assert cfg.preferred_backend == BackendType.AUTO
    assert cfg.to_dict()["context_length"] == 4096


def test_session_config_validation() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        SessionConfig(max_context_tokens=10)

    sc = SessionConfig(max_context_tokens=1024, enable_prefix_caching=True)
    assert sc.to_dict()["enable_prefix_caching"] is True


def test_token_str_and_chunk() -> None:
    chunk = TokenChunk(text="Hello", token_id=42, is_special=False, logprob=-0.12)
    assert str(chunk) == "Hello"

    t_str = chunk.to_token_str()
    assert isinstance(t_str, str)
    assert t_str == "Hello"
    assert t_str.text == "Hello"
    assert t_str.token_id == 42
    assert t_str.logprob == -0.12


def test_chat_message_and_request() -> None:
    msg1 = ChatMessage(role="system", content="You are a helpful assistant.")
    msg2 = ChatMessage(role="user", content="Hello!")
    assert msg1.to_dict() == {"role": "system", "content": "You are a helpful assistant."}

    req = ChatCompletionRequest(
        model="smollm2",
        messages=[msg1, msg2],
        temperature=0.8,
        stream=True,
    )
    d = req.to_dict()
    assert d["model"] == "smollm2"
    assert len(d["messages"]) == 2
    assert d["stream"] is True

    parsed = ChatCompletionRequest.from_dict(
        {
            "model": "smollm2",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": "0.5",
        }
    )
    assert parsed.temperature == 0.5
    assert len(parsed.messages) == 1


def test_generate_response_and_metrics() -> None:
    m = GenerationMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        time_to_first_token_ms=15.5,
        tokens_per_second=45.2,
        peak_rss_mb=256.0,
    )
    resp = GenerateResponse(
        text="Sample output",
        model="test-model",
        metrics=m,
    )
    d = resp.to_dict()
    assert d["text"] == "Sample output"
    assert d["metrics"]["tokens_per_second"] == 45.2
