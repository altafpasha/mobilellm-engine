# MobileLLM Python SDK & Local Edge Server

The official Python SDK, CLI, and local loopback OpenAI-compatible server for the **MobileLLM C++20 Inference Engine**.

---

## Features

- **Direct C++20 FFI (`NativeClient`)**: Run zero-overhead local inference via `ctypes` calling `libmobilellm` directly on macOS and Linux without any running daemon or server.
- **OpenAI-Compatible Loopback Server**: Local HTTP & Server-Sent Events (SSE) server implementing `/v1/chat/completions` and `/v1/models` matching the OpenAI API wire specification.
- **Strict Loopback Security**: Server binds to `127.0.0.1` by default. Exposing to `0.0.0.0` strictly requires `--allow-remote-connections` and an `--api-key`.
- **Zero Heavy Dependencies**: The core FFI client, CLI, and HTTP server are fully functional with Python's standard library (`ctypes`, `dataclasses`, `http.server`, `asyncio`, `urllib`).
- **Interactive CLI**: `mobilellm run`, `mobilellm chat`, `mobilellm serve`, `mobilellm bench`, and `mobilellm info`.

---

## Quickstart

### 1. Direct Python FFI (Zero Server Required)

```python
from mobilellm.bindings.native_client import NativeClient
from mobilellm.models import ModelConfig, SamplingParams

with NativeClient() as client:
    with client.load_model(ModelConfig(model_path="models/SmolLM2-135M-Instruct-Q4_K_M.gguf")) as session:
        for chunk in session.generate("Why is the sky blue?", SamplingParams(max_tokens=64)):
            print(chunk.text, end="", flush=True)
```

### 2. Launching Local OpenAI-Compatible Server

```bash
# Start server on 127.0.0.1:8080 serving a GGUF model
mobilellm serve --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --port 8080
```

### 3. Using with the Official OpenAI Python SDK

```python
from openai import OpenAI

# Connect standard OpenAI client to local MobileLLM instance
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello edge AI!"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

