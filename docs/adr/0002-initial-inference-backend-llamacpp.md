# ADR-0002: Initial Inference Engine Foundation via llama.cpp / GGML

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 5 (Hardware)

---

## Context

Developing an inference runtime from scratch requires thousands of lines of low-level tensor operations, quantized matrix kernels (Q4_0, Q4_K_M, Q8_0), memory layout management, and model architecture definitions (Llama, Qwen, Mistral, Gemma). 

Re-implementing these foundational tensor kernels before establishing mobile memory management, Kotlin SDKs, and request scheduling would delay the project by several months and distract from the primary objective: building a mobile-first LLM runtime and scheduling ecosystem.

## Decision

We adopt **llama.cpp / GGML** as the initial baseline inference backend for MobileLLM Engine.

### Integration Strategy:
1. **Incremental Independence:** We start by integrating GGML/llama.cpp as an isolated compute engine under `backends/cpu/` and `backends/vulkan/`.
2. **Strict Abstraction Barrier:** The public API (`include/mobilellm/engine.h`, `c_api.h`) and upper subsystems (`scheduler/`, `android/`, `python/`, `server/`) must **never** include `llama.h` or `ggml.h` headers directly. All interactions are mediated through the internal `IBackend` abstraction.
3. **Vendoring & Pinning:** We pin a specific, verified commit of llama.cpp as a Git submodule or vendored third-party dependency with a documented license audit (MIT).
4. **Selective Adaptation:** Over time, mobile-specific innovations (paged KV memory management, dynamic CPU core pinning, thermal-aware throttling) are layered on top of or substituted into the backend where they provide measurable value.

## Consequences

### Positive:
- Immediate access to highly optimized ARM NEON kernels for quantized integer dot products.
- Day-one support for widely available GGUF quantized models (Qwen 2.5, Llama 3.2, Gemma).
- Allows the team to focus immediately on mobile-specific engineering challenges: Android lifecycle, low-memory handling, JNI stability, request scheduling, and battery/thermal profiling.

### Negative / Trade-offs:
- Dependency on an external codebase that evolves rapidly; upstream updates must be selectively audited and pinned.
- Internal GGML memory structures must be adapted or bridged to work with our mobile paged KV-cache architecture.

## Alternatives Considered

- **Custom Kernel Engine from Scratch:** Re-implementing transformer kernels from scratch was rejected per `CONTEXT.md` Section 2 ("Non-goals: Reimplement every transformer architecture or inference kernel from scratch").
- **MLC-LLM (TVM):** Provides impressive Vulkan compilation, but requires complex ahead-of-time model weight compilation and large artifact packaging, which is less accessible for mobile developers loading arbitrary local GGUF models.
- **ONNX Runtime Mobile:** Has broader operator support, but GGUF model support and quantized LLM generation throughput on mobile CPU are significantly less optimized than GGML.
