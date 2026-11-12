"""
HTTP and Server-Sent Events (SSE) transports connecting to MobileLLM OpenAI-compatible endpoints.
"""

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from mobilellm.exceptions import (
    MobileLLMConnectionError,
    MobileLLMTimeoutError,
    map_http_status,
)
from mobilellm.models import (
    EngineStatus,
    GenerateResponse,
    GenerationMetrics,
    ModelConfig,
    ModelInfo,
    RuntimeHealth,
    SamplingParams,
    SessionConfig,
    TokenChunk,
)
from mobilellm.transport.base import AsyncBaseTransport, BaseTransport

logger = logging.getLogger("mobilellm.transport.http")


def _normalize_endpoint(endpoint: str) -> str:
    url = endpoint.strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    return url


def _v1_url(endpoint: str, path: str) -> str:
    base = _normalize_endpoint(endpoint)
    # If endpoint ends with /v1, strip it first to ensure clean join
    base = base.removesuffix("/v1")
    clean_path = path.lstrip("/")
    if not clean_path.startswith("v1/"):
        clean_path = f"v1/{clean_path}"
    return f"{base}/{clean_path}"


class HTTPTransport(BaseTransport):
    """Synchronous HTTP client for MobileLLM REST and SSE APIs."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ):
        self._endpoint = _normalize_endpoint(endpoint)
        self._api_key = api_key
        self._timeout = timeout
        self._custom_headers = headers or {}
        self._client: httpx.Client | None = None

    def _get_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "MobileLLM-Python-SDK/0.1.0"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        h.update(self._custom_headers)
        return h

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers=self._get_headers(),
            )
        return self._client

    def connect(self) -> None:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
        except httpx.ConnectError as e:
            raise MobileLLMConnectionError(
                f"Failed to connect to MobileLLM at {self._endpoint}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Connection to {self._endpoint} timed out: {e}") from e

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None

    def health(self) -> RuntimeHealth:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            return RuntimeHealth(
                status=data.get("status", "ok"),
                version=data.get("version", "0.1.0"),
                uptime_seconds=float(data.get("uptime_seconds", 0.0)),
                backend=data.get("backend", "CPU_NEON"),
                memory_used_mb=float(data.get("memory_used_mb", 0.0)),
                memory_budget_mb=float(data.get("memory_budget_mb", 0.0)),
                models_loaded=int(data.get("models_loaded", 0)),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise MobileLLMConnectionError(f"Health check failed: {e}") from e

    def status(self) -> EngineStatus:
        h = self.health()
        client = self._get_client()
        models: list[str] = []
        try:
            resp = client.get(_v1_url(self._endpoint, "models"))
            if resp.status_code == 200:
                data = resp.json()
                raw_models = data.get("data", data.get("models", []))
                models = [
                    m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in raw_models
                ]
        except Exception:
            pass

        return EngineStatus(
            status=h.status,
            version=h.version,
            hardware={"backend": h.backend, "memory_budget_mb": h.memory_budget_mb},
            models=models,
            active_sessions=0,
        )

    def detect_hardware(self) -> dict[str, Any]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("hardware", {"backend": data.get("backend", "AUTO")})
        except Exception:
            pass
        return {"backend": "AUTO"}

    def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models")
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            raw_models = data.get("data", data.get("models", []))
            result: list[ModelInfo] = []
            for item in raw_models:
                if isinstance(item, dict):
                    m_id = item.get("id", "default")
                    result.append(
                        ModelInfo(
                            id=m_id,
                            path=str(item.get("path") or item.get("model_path") or m_id),
                            architecture=item.get("architecture", "unknown"),
                            quantization=item.get("quantization", "unknown"),
                            context_length=int(item.get("context_length", 2048)),
                            file_size_bytes=int(item.get("file_size_bytes", 0)),
                            valid=bool(item.get("valid", True)),
                            loaded=bool(item.get("loaded", True)),
                        )
                    )
                elif isinstance(item, str):
                    result.append(ModelInfo(id=item, path=item, loaded=True))
            return result
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Failed to list models: {e}") from e

    def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models/load")
        payload: dict[str, Any] = {
            "model_path": config.model_path,
            "context_length": config.context_length,
            "backend": int(config.preferred_backend),
            "n_threads": config.n_threads,
            "use_mmap": config.use_mmap,
            "memory_budget_mb": config.memory_budget_mb,
        }
        if session_config:
            payload["max_context_tokens"] = session_config.max_context_tokens
        try:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            m_id = data.get("id", data.get("model_id", config.model_path))
            return ModelInfo(
                id=m_id,
                path=config.model_path,
                architecture=data.get("architecture", "unknown"),
                quantization=data.get("quantization", "unknown"),
                context_length=config.context_length,
                file_size_bytes=int(data.get("file_size_bytes", 0)),
                valid=True,
                loaded=True,
            )
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Model load request failed: {e}") from e

    def unload_model(self, model_id: str) -> bool:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models/unload")
        try:
            resp = client.post(url, json={"model": model_id, "model_id": model_id})
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            return True
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Model unload request failed: {e}") from e

    def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        client = self._get_client()
        url = _v1_url(self._endpoint, "sessions")
        payload: dict[str, Any] = {"model": model_id}
        if config:
            payload["max_context_tokens"] = config.max_context_tokens
            payload["enable_prefix_caching"] = config.enable_prefix_caching
        try:
            resp = client.post(url, json=payload)
            if resp.status_code not in (200, 201):
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            return str(data.get("session_id", data.get("id", "")))
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Session creation failed: {e}") from e

    def destroy_session(self, session_id: str) -> bool:
        client = self._get_client()
        url = _v1_url(self._endpoint, f"sessions/{session_id}")
        try:
            resp = client.delete(url)
            return resp.status_code in (200, 204)
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Session destroy failed: {e}") from e

    def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        client = self._get_client()
        url = _v1_url(self._endpoint, "chat/completions")
        p = params or SamplingParams()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "repetition_penalty": p.repetition_penalty,
            "max_tokens": p.max_tokens,
            "stream": False,
        }
        if session_id:
            payload["session_id"] = session_id
        if p.stop:
            payload["stop"] = p.stop

        try:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            choices = data.get("choices", [])
            text = ""
            finish_reason = "stop"
            if choices:
                first = choices[0]
                text = first.get("message", {}).get("content", first.get("text", ""))
                finish_reason = first.get("finish_reason", "stop")
            usage = data.get("usage", {})
            metrics = None
            if usage:
                metrics = GenerationMetrics(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    time_to_first_token_ms=float(usage.get("ttft_ms", 0.0)),
                    tokens_per_second=float(usage.get("tps", 0.0)),
                    peak_rss_mb=float(usage.get("peak_rss_mb", 0.0)),
                )
            return GenerateResponse(
                text=text,
                model=model_id,
                finish_reason=finish_reason,
                metrics=metrics,
                session_id=session_id,
            )
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Generation timed out: {e}") from e
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Generation request failed: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> Iterator[TokenChunk]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "chat/completions")
        p = params or SamplingParams()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "repetition_penalty": p.repetition_penalty,
            "max_tokens": p.max_tokens,
            "stream": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if p.stop:
            payload["stop"] = p.stop

        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"

        try:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise map_http_status(resp.status_code, body)

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(raw_data)
                        choices = chunk_json.get("choices", [])
                        if choices:
                            c = choices[0]
                            delta = c.get("delta", {})
                            content = delta.get("content", c.get("text", ""))
                            if content:
                                yield TokenChunk(
                                    text=content,
                                    token_id=int(c.get("token_id", 0)),
                                    is_special=bool(c.get("is_special", False)),
                                    logprob=float(c.get("logprob", 0.0)),
                                )
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Streaming generation timed out: {e}") from e
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Streaming request failed: {e}") from e

    def cancel(self, gen_handle: str | None = None) -> None:
        client = self._get_client()
        url = _v1_url(self._endpoint, "cancel")
        try:
            client.post(url, json={"handle": gen_handle})
        except Exception:
            pass


class AsyncHTTPTransport(AsyncBaseTransport):
    """Asynchronous HTTP client for MobileLLM REST and SSE APIs."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ):
        self._endpoint = _normalize_endpoint(endpoint)
        self._api_key = api_key
        self._timeout = timeout
        self._custom_headers = headers or {}
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "MobileLLM-Python-SDK/0.1.0"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        h.update(self._custom_headers)
        return h

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers=self._get_headers(),
            )
        return self._client

    async def connect(self) -> None:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
        except httpx.ConnectError as e:
            raise MobileLLMConnectionError(
                f"Failed to connect to MobileLLM at {self._endpoint}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Connection to {self._endpoint} timed out: {e}") from e

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health(self) -> RuntimeHealth:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            return RuntimeHealth(
                status=data.get("status", "ok"),
                version=data.get("version", "0.1.0"),
                uptime_seconds=float(data.get("uptime_seconds", 0.0)),
                backend=data.get("backend", "CPU_NEON"),
                memory_used_mb=float(data.get("memory_used_mb", 0.0)),
                memory_budget_mb=float(data.get("memory_budget_mb", 0.0)),
                models_loaded=int(data.get("models_loaded", 0)),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise MobileLLMConnectionError(f"Health check failed: {e}") from e

    async def status(self) -> EngineStatus:
        h = await self.health()
        client = self._get_client()
        models: list[str] = []
        try:
            resp = await client.get(_v1_url(self._endpoint, "models"))
            if resp.status_code == 200:
                data = resp.json()
                raw_models = data.get("data", data.get("models", []))
                models = [
                    m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in raw_models
                ]
        except Exception:
            pass

        return EngineStatus(
            status=h.status,
            version=h.version,
            hardware={"backend": h.backend, "memory_budget_mb": h.memory_budget_mb},
            models=models,
            active_sessions=0,
        )

    async def detect_hardware(self) -> dict[str, Any]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "health")
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("hardware", {"backend": data.get("backend", "AUTO")})
        except Exception:
            pass
        return {"backend": "AUTO"}

    async def list_models(self) -> list[ModelInfo]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models")
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            raw_models = data.get("data", data.get("models", []))
            result: list[ModelInfo] = []
            for item in raw_models:
                if isinstance(item, dict):
                    m_id = item.get("id", "default")
                    result.append(
                        ModelInfo(
                            id=m_id,
                            path=str(item.get("path") or item.get("model_path") or m_id),
                            architecture=item.get("architecture", "unknown"),
                            quantization=item.get("quantization", "unknown"),
                            context_length=int(item.get("context_length", 2048)),
                            file_size_bytes=int(item.get("file_size_bytes", 0)),
                            valid=bool(item.get("valid", True)),
                            loaded=bool(item.get("loaded", True)),
                        )
                    )
                elif isinstance(item, str):
                    result.append(ModelInfo(id=item, path=item, loaded=True))
            return result
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Failed to list models: {e}") from e

    async def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> ModelInfo:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models/load")
        payload: dict[str, Any] = {
            "model_path": config.model_path,
            "context_length": config.context_length,
            "backend": int(config.preferred_backend),
            "n_threads": config.n_threads,
            "use_mmap": config.use_mmap,
            "memory_budget_mb": config.memory_budget_mb,
        }
        if session_config:
            payload["max_context_tokens"] = session_config.max_context_tokens
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            m_id = data.get("id", data.get("model_id", config.model_path))
            return ModelInfo(
                id=m_id,
                path=config.model_path,
                architecture=data.get("architecture", "unknown"),
                quantization=data.get("quantization", "unknown"),
                context_length=config.context_length,
                file_size_bytes=int(data.get("file_size_bytes", 0)),
                valid=True,
                loaded=True,
            )
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Model load request failed: {e}") from e

    async def unload_model(self, model_id: str) -> bool:
        client = self._get_client()
        url = _v1_url(self._endpoint, "models/unload")
        try:
            resp = await client.post(url, json={"model": model_id, "model_id": model_id})
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            return True
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Model unload request failed: {e}") from e

    async def create_session(
        self,
        model_id: str,
        config: SessionConfig | None = None,
    ) -> str:
        client = self._get_client()
        url = _v1_url(self._endpoint, "sessions")
        payload: dict[str, Any] = {"model": model_id}
        if config:
            payload["max_context_tokens"] = config.max_context_tokens
            payload["enable_prefix_caching"] = config.enable_prefix_caching
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code not in (200, 201):
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            return str(data.get("session_id", data.get("id", "")))
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Session creation failed: {e}") from e

    async def destroy_session(self, session_id: str) -> bool:
        client = self._get_client()
        url = _v1_url(self._endpoint, f"sessions/{session_id}")
        try:
            resp = await client.delete(url)
            return resp.status_code in (200, 204)
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Session destroy failed: {e}") from e

    async def generate(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> GenerateResponse:
        client = self._get_client()
        url = _v1_url(self._endpoint, "chat/completions")
        p = params or SamplingParams()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "repetition_penalty": p.repetition_penalty,
            "max_tokens": p.max_tokens,
            "stream": False,
        }
        if session_id:
            payload["session_id"] = session_id
        if p.stop:
            payload["stop"] = p.stop

        try:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise map_http_status(resp.status_code, resp.text)
            data = resp.json()
            choices = data.get("choices", [])
            text = ""
            finish_reason = "stop"
            if choices:
                first = choices[0]
                text = first.get("message", {}).get("content", first.get("text", ""))
                finish_reason = first.get("finish_reason", "stop")
            usage = data.get("usage", {})
            metrics = None
            if usage:
                metrics = GenerationMetrics(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    time_to_first_token_ms=float(usage.get("ttft_ms", 0.0)),
                    tokens_per_second=float(usage.get("tps", 0.0)),
                    peak_rss_mb=float(usage.get("peak_rss_mb", 0.0)),
                )
            return GenerateResponse(
                text=text,
                model=model_id,
                finish_reason=finish_reason,
                metrics=metrics,
                session_id=session_id,
            )
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Generation timed out: {e}") from e
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Generation request failed: {e}") from e

    async def generate_stream(
        self,
        prompt: str,
        model_id: str,
        params: SamplingParams | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[TokenChunk]:
        client = self._get_client()
        url = _v1_url(self._endpoint, "chat/completions")
        p = params or SamplingParams()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "repetition_penalty": p.repetition_penalty,
            "max_tokens": p.max_tokens,
            "stream": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if p.stop:
            payload["stop"] = p.stop

        headers = self._get_headers()
        headers["Accept"] = "text/event-stream"

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise map_http_status(resp.status_code, body)

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(raw_data)
                        choices = chunk_json.get("choices", [])
                        if choices:
                            c = choices[0]
                            delta = c.get("delta", {})
                            content = delta.get("content", c.get("text", ""))
                            if content:
                                yield TokenChunk(
                                    text=content,
                                    token_id=int(c.get("token_id", 0)),
                                    is_special=bool(c.get("is_special", False)),
                                    logprob=float(c.get("logprob", 0.0)),
                                )
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as e:
            raise MobileLLMTimeoutError(f"Streaming generation timed out: {e}") from e
        except httpx.HTTPError as e:
            raise MobileLLMConnectionError(f"Streaming request failed: {e}") from e

    async def cancel(self, gen_handle: str | None = None) -> None:
        client = self._get_client()
        url = _v1_url(self._endpoint, "cancel")
        try:
            await client.post(url, json={"handle": gen_handle})
        except Exception:
            pass
