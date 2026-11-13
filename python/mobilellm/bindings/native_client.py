"""
Direct C-ABI FFI wrapper linking libmobilellm dynamic library using ctypes.
"""

import ctypes
import os
import queue
import threading
from collections.abc import Callable, Generator
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
)
from pathlib import Path
from typing import Any, Dict

from mobilellm.exceptions import (
    MobileLLMError,
    map_error_code,
)
from mobilellm.models import (
    GenerationMetrics,
    ModelConfig,
    SamplingParams,
    SessionConfig,
    TokenChunk,
)

# --- CTYPES Structures matching c_api.h ---


class _CMllmStatus(Structure):
    _fields_ = [
        ("code", c_int32),
        ("message", c_char_p),
    ]


class _CMllmModelConfig(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("backend", c_int32),
        ("context_length", c_uint32),
        ("n_threads", c_uint32),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("memory_budget_mb", c_float),
    ]


class _CMllmSessionConfig(Structure):
    _fields_ = [
        ("max_context_tokens", c_uint32),
        ("enable_prefix_caching", c_bool),
    ]


class _CMllmSamplingParams(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("top_p", c_float),
        ("top_k", c_int32),
        ("min_p", c_float),
        ("repetition_penalty", c_float),
        ("max_tokens", c_uint32),
    ]


class _CMllmTokenChunk(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("token_id", c_int32),
        ("is_special", c_bool),
        ("logprob", c_float),
    ]


class _CMllmMetrics(Structure):
    _fields_ = [
        ("prompt_tokens", c_uint32),
        ("completion_tokens", c_uint32),
        ("ttft_ms", c_double),
        ("tps", c_double),
        ("peak_rss_mb", c_double),
        ("prompt_tps", c_double),
        ("prefill_ms", c_double),
        ("decode_ms", c_double),
        ("total_ms", c_double),
    ]


# Callback function prototypes
_TOKEN_CB_TYPE = ctypes.CFUNCTYPE(c_bool, POINTER(_CMllmTokenChunk), c_void_p)
_COMPLETE_CB_TYPE = ctypes.CFUNCTYPE(None, _CMllmStatus, POINTER(_CMllmMetrics), c_void_p)


def _find_library() -> str:
    """Discovers the built libmobilellm dynamic library on the system."""
    env_path = os.environ.get("MOBILELLM_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Candidate file names
    names = [
        "libmobilellm.dylib",
        "libmobilellm.so",
        "libmobilellm.0.1.0.dylib",
        "libmobilellm.0.dylib",
    ]

    # Search search paths relative to repo root
    current = Path(__file__).resolve().parent
    repo_root = current.parent.parent.parent

    search_dirs = [
        repo_root / "build_llama" / "runtime",
        repo_root / "build" / "runtime",
        repo_root / "build",
        repo_root / "lib",
        Path("/usr/local/lib"),
        Path("/opt/homebrew/lib"),
    ]

    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)

    raise FileNotFoundError(
        "Could not find libmobilellm dynamic library. "
        "Build the project using CMake or set MOBILELLM_LIB_PATH."
    )


class NativeSession:
    """Active inference session bound to a loaded model in native memory."""

    def __init__(
        self,
        lib: ctypes.CDLL,
        engine_handle: int,
        session_handle: int,
        model_handle: int,
    ):
        self._lib = lib
        self._engine_handle = engine_handle
        self._session_handle = session_handle
        self._model_handle = model_handle
        self._active_gen: int | None = None
        self._is_closed = False

    @property
    def session_handle(self) -> int:
        return self._session_handle

    @property
    def model_handle(self) -> int:
        return self._model_handle

    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        on_metrics: Callable[[GenerationMetrics], None] | None = None,
    ) -> Generator[TokenChunk, None, None]:
        """
        Executes streaming token generation yielding TokenChunk instances.
        """
        if self._is_closed:
            raise MobileLLMError("Session is closed")
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        p = params or SamplingParams()
        c_sampling = _CMllmSamplingParams(
            temperature=p.temperature,
            top_p=p.top_p,
            top_k=p.top_k,
            min_p=p.min_p,
            repetition_penalty=p.repetition_penalty,
            max_tokens=p.max_tokens,
        )

        token_q: queue.Queue[TokenChunk | None] = queue.Queue()
        error_holder: list[Exception | None] = [None]
        metrics_holder: list[GenerationMetrics | None] = [None]
        done_event = threading.Event()

        def _on_token(chunk_ptr: Any, user_data: Any) -> bool:
            if not chunk_ptr:
                return True
            c = chunk_ptr.contents
            text = c.text.decode("utf-8", errors="replace") if c.text else ""
            token = TokenChunk(
                text=text,
                token_id=c.token_id,
                is_special=bool(c.is_special),
                logprob=float(c.logprob),
            )
            token_q.put(token)
            return True

        def _on_complete(
            status: Any,
            metrics_ptr: Any,
            user_data: Any,
        ) -> None:
            try:
                if metrics_ptr:
                    m = metrics_ptr.contents
                    metrics_holder[0] = GenerationMetrics(
                        prompt_tokens=m.prompt_tokens,
                        completion_tokens=m.completion_tokens,
                        time_to_first_token_ms=m.ttft_ms,
                        tokens_per_second=m.tps,
                        peak_rss_mb=m.peak_rss_mb,
                        prompt_tokens_per_second=getattr(m, "prompt_tps", 0.0),
                        prefill_time_ms=getattr(m, "prefill_ms", 0.0),
                        decode_time_ms=getattr(m, "decode_ms", 0.0),
                        total_duration_ms=getattr(m, "total_ms", 0.0),
                    )
                    if on_metrics and metrics_holder[0]:
                        on_metrics(metrics_holder[0])

                if status.code != 0 and status.code != 9:  # 9 = cancelled
                    msg = status.message.decode("utf-8", errors="replace") if status.message else "Generation failed"
                    err = map_error_code(status.code, msg)
                    error_holder[0] = err
            finally:
                token_q.put(None)
                done_event.set()

        c_token_cb = _TOKEN_CB_TYPE(_on_token)
        c_complete_cb = _COMPLETE_CB_TYPE(_on_complete)
        gen_handle = c_uint64(0)

        status = self._lib.mllm_generate(
            c_uint64(self._engine_handle),
            c_uint64(self._session_handle),
            prompt.encode("utf-8"),
            ctypes.byref(c_sampling),
            c_token_cb,
            c_complete_cb,
            None,
            ctypes.byref(gen_handle),
        )

        if status.code != 0:
            msg = status.message.decode("utf-8", errors="replace") if status.message else "mllm_generate failed"
            raise map_error_code(status.code, msg) or MobileLLMError(msg, status.code)

        self._active_gen = gen_handle.value

        try:
            while True:
                item = token_q.get()
                if item is None:
                    break
                yield item

            if error_holder[0] is not None:
                raise error_holder[0]
        finally:
            self._active_gen = None

    def cancel(self) -> None:
        """Interrupts currently active generation."""
        if self._active_gen and not self._is_closed:
            self._lib.mllm_cancel(
                c_uint64(self._engine_handle),
                c_uint64(self._active_gen),
            )
            self._active_gen = None

    def close(self) -> None:
        """Destroys the native session context."""
        if not self._is_closed:
            self.cancel()
            self._lib.mllm_session_destroy(
                c_uint64(self._engine_handle),
                c_uint64(self._session_handle),
            )
            self._is_closed = True

    def __enter__(self) -> "NativeSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class NativeClient:
    """Zero-overhead Python FFI wrapper directly invoking libmobilellm."""

    def __init__(self, lib_path: str | None = None):
        self._path = lib_path or _find_library()
        self._lib = ctypes.CDLL(self._path)
        self._setup_function_signatures()
        self._engine_handle: int | None = None
        self._loaded_models: list[int] = []
        self._init_engine()

    def _setup_function_signatures(self) -> None:
        # Engine lifecycle
        self._lib.mllm_engine_create.argtypes = [POINTER(c_uint64)]
        self._lib.mllm_engine_create.restype = _CMllmStatus

        self._lib.mllm_engine_destroy.argtypes = [c_uint64]
        self._lib.mllm_engine_destroy.restype = _CMllmStatus

        # Model management
        self._lib.mllm_model_load.argtypes = [
            c_uint64,
            POINTER(_CMllmModelConfig),
            POINTER(c_uint64),
        ]
        self._lib.mllm_model_load.restype = _CMllmStatus

        self._lib.mllm_model_unload.argtypes = [c_uint64, c_uint64]
        self._lib.mllm_model_unload.restype = _CMllmStatus

        # Session management
        self._lib.mllm_session_create.argtypes = [
            c_uint64,
            c_uint64,
            POINTER(_CMllmSessionConfig),
            POINTER(c_uint64),
        ]
        self._lib.mllm_session_create.restype = _CMllmStatus

        self._lib.mllm_session_destroy.argtypes = [c_uint64, c_uint64]
        self._lib.mllm_session_destroy.restype = _CMllmStatus

        # Generation
        self._lib.mllm_generate.argtypes = [
            c_uint64,
            c_uint64,
            c_char_p,
            POINTER(_CMllmSamplingParams),
            _TOKEN_CB_TYPE,
            _COMPLETE_CB_TYPE,
            c_void_p,
            POINTER(c_uint64),
        ]
        self._lib.mllm_generate.restype = _CMllmStatus

        self._lib.mllm_cancel.argtypes = [c_uint64, c_uint64]
        self._lib.mllm_cancel.restype = _CMllmStatus

        # Model inspect (if supported by library build)
        if hasattr(self._lib, "mllm_model_inspect"):
            self._lib.mllm_model_inspect.argtypes = [c_char_p, c_char_p, c_size_t]
            self._lib.mllm_model_inspect.restype = _CMllmStatus

        # Hardware & Version
        self._lib.mllm_get_version.argtypes = [c_char_p, c_size_t]
        self._lib.mllm_get_version.restype = _CMllmStatus

        self._lib.mllm_detect_hardware.argtypes = [c_char_p, c_size_t]
        self._lib.mllm_detect_hardware.restype = _CMllmStatus

    def _init_engine(self) -> None:
        h = c_uint64(0)
        st = self._lib.mllm_engine_create(ctypes.byref(h))
        if st.code != 0:
            msg = st.message.decode("utf-8") if st.message else "Engine creation failed"
            raise map_error_code(st.code, msg) or MobileLLMError(msg, st.code)
        self._engine_handle = h.value

    def get_version(self) -> str:
        """Returns the native engine version string."""
        buf = ctypes.create_string_buffer(64)
        st = self._lib.mllm_get_version(buf, 64)
        if st.code != 0:
            return "0.1.0-dev"
        return buf.value.decode("utf-8")

    def detect_hardware(self) -> str:
        """Returns native hardware features as JSON string."""
        buf = ctypes.create_string_buffer(1024)
        st = self._lib.mllm_detect_hardware(buf, 1024)
        if st.code != 0:
            return "{}"
        return buf.value.decode("utf-8")

    def inspect_model(self, model_path: str) -> Dict[str, Any]:
        """Inspects GGUF file metadata using native loader."""
        if not hasattr(self._lib, "mllm_model_inspect"):
            return {"model_path": model_path, "valid": True}
        buf = ctypes.create_string_buffer(4096)
        st = self._lib.mllm_model_inspect(model_path.encode("utf-8"), buf, 4096)
        if st.code != 0:
            msg = st.message.decode("utf-8") if st.message else "inspect failed"
            raise map_error_code(st.code, msg) or MobileLLMError(msg, st.code)
        try:
            import json

            return json.loads(buf.value.decode("utf-8"))
        except Exception:
            return {"model_path": model_path, "valid": True}

    def load_model_handle(self, config: ModelConfig) -> int:
        """Loads GGUF model weights into native memory and returns model handle."""
        if self._engine_handle is None:
            raise MobileLLMError("Engine is destroyed")

        c_mcfg = _CMllmModelConfig(
            model_path=config.model_path.encode("utf-8"),
            backend=int(config.preferred_backend),
            context_length=config.context_length,
            n_threads=config.n_threads,
            use_mmap=config.use_mmap,
            use_mlock=False,
            memory_budget_mb=float(config.memory_budget_mb),
        )

        m_handle = c_uint64(0)
        st = self._lib.mllm_model_load(
            c_uint64(self._engine_handle),
            ctypes.byref(c_mcfg),
            ctypes.byref(m_handle),
        )
        if st.code != 0:
            msg = st.message.decode("utf-8") if st.message else "Model load failed"
            raise map_error_code(st.code, msg) or MobileLLMError(msg, st.code)

        self._loaded_models.append(m_handle.value)
        return m_handle.value

    def create_session(
        self,
        model_handle: int,
        session_config: SessionConfig | None = None,
    ) -> NativeSession:
        """Initializes a new inference session for a loaded model handle."""
        if self._engine_handle is None:
            raise MobileLLMError("Engine is destroyed")

        s_cfg = session_config or SessionConfig()
        c_scfg = _CMllmSessionConfig(
            max_context_tokens=s_cfg.max_context_tokens,
            enable_prefix_caching=s_cfg.enable_prefix_caching,
        )

        s_handle = c_uint64(0)
        st_sess = self._lib.mllm_session_create(
            c_uint64(self._engine_handle),
            c_uint64(model_handle),
            ctypes.byref(c_scfg),
            ctypes.byref(s_handle),
        )
        if st_sess.code != 0:
            msg = st_sess.message.decode("utf-8") if st_sess.message else "Session creation failed"
            raise map_error_code(st_sess.code, msg) or MobileLLMError(msg, st_sess.code)

        return NativeSession(
            lib=self._lib,
            engine_handle=self._engine_handle,
            session_handle=s_handle.value,
            model_handle=model_handle,
        )

    def load_model(
        self,
        config: ModelConfig,
        session_config: SessionConfig | None = None,
    ) -> NativeSession:
        """
        Loads GGUF model weights into native memory and initializes an inference session.
        """
        m_handle = self.load_model_handle(config)
        s_cfg = session_config or SessionConfig(max_context_tokens=config.context_length)
        return self.create_session(m_handle, s_cfg)

    def unload_model(self, model_handle: int) -> None:
        """Unloads model weights from native memory."""
        if self._engine_handle is not None and model_handle in self._loaded_models:
            self._lib.mllm_model_unload(
                c_uint64(self._engine_handle),
                c_uint64(model_handle),
            )
            self._loaded_models.remove(model_handle)

    def close(self) -> None:
        """Frees all loaded models and shuts down native engine."""
        if self._engine_handle is not None:
            for m in list(self._loaded_models):
                self._lib.mllm_model_unload(
                    c_uint64(self._engine_handle),
                    c_uint64(m),
                )
            self._loaded_models.clear()
            self._lib.mllm_engine_destroy(c_uint64(self._engine_handle))
            self._engine_handle = None

    def __enter__(self) -> "NativeClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
