# ADR-0008: Scheduler and Mobile Resource Manager Runtime Integration

- **Status:** Approved & Accepted (Architectural Decision)
- **Decider:** Agent 0 (Lead Architect & Principal C++ Inference Runtime Engineer)
- **Date:** 2026-11-12
- **Consulted:** Agent 1 (Runtime), Agent 2 (Scheduler), Agent 3 (Android), Agent 4 (Loopback API), Agent 5 (Hardware/Backends)

---

## Context

The MobileLLM codebase features a clean C++20 core runtime, a functional `LlamaCppBackend` executing ARM64 NEON inference, a working Android JNI layer verified on an Android API 37 16 KB page-size AVD emulator, and a passing Python test suite. All critical P0 remediation defects (thread-local error lifetime, per-session sampler concurrency, JNI reference management, Python error alignment, and server path containment) have been resolved.

However, an architectural audit reveals that the **scheduler subsystem (`scheduler/`) remains disconnected from the active runtime execution path**:
1. `EngineImpl::generate_async` currently bypasses the scheduler, launching an isolated `std::thread` per request that calls `LlamaCppBackend` synchronously in an uncoordinated loop.
2. `runtime/CMakeLists.txt` does not link `mobilellm_scheduler`.
3. `KVCacheManager` allocates dummy heap vectors (`KVBlock::buffer`) in isolation, while `LlamaCppBackend` relies on `llama.cpp`'s internal contiguous context buffer allocations.
4. Mobile memory budgeting and Android OS `onTrimMemory` events are not connected to the engine or JNI bridge, leaving the process vulnerable to Android Low Memory Killer Daemon (`lmkd`) `SIGKILL` termination under memory pressure.

We require a production-quality integration architecture that connects the scheduler and mobile resource manager to the runtime engine while maintaining C/C++ and Kotlin ABI stability, without replacing `llama.cpp` and without falsely claiming hardware paged-attention execution.

---

## Decision

We adopt an **Engine-Level Coordination Service Architecture** connecting the `scheduler/` subsystem to `EngineImpl` and the hardware backend.

### Core Architectural Decisions:

1. **Scheduler Ownership & Topology:**
   - The scheduler (`ContinuousBatchScheduler`) is owned directly by `EngineImpl` as a runtime coordination service.
   - `EngineImpl` initializes the scheduler during `initialize()` and gracefully drains and joins it during `shutdown()`.
   - `ContinuousBatchScheduler` runs a single dedicated background worker thread executing iteration-level scheduling.

2. **Separation of Scheduling from `IBackend`:**
   - Scheduling remains strictly above the backend abstraction layer.
   - `IBackend` remains a pure hardware execution interface (prefill prompt, single/batched token decode, sampling, detokenization).
   - Compute backends (CPU, Vulkan, NPU) do not manage request queues, priorities, or Android memory trim events.
   - A lightweight `BackendCapabilities` query enables the scheduler to adapt its batching strategy (e.g. batch size 1 on CPU and fixed-shape NPUs; batch size 4–8 on Vulkan GPUs).

3. **Realistic KV-Cache Architecture (Virtual Accounting vs. Physical Storage):**
   - **We explicitly distinguish accounting from execution.**
   - Physical KV cache allocation and attention computation remain owned and managed by `llama.cpp` (`llama_context`).
   - We eliminate phantom memory allocations: `KVBlock::buffer` will not allocate dummy heap byte vectors during llama.cpp execution.
   - The scheduler's `KVCacheManager` operates as a **Virtual Block Accounting & Sequence Compaction Adapter**: it tracks 16-token virtual blocks, computes exact memory usage based on model tensor hyper-parameters, and coordinates sequence truncation via `llama_kv_cache_seq_rm()` and `llama_kv_cache_seq_add()`.
   - True physical paged attention (indirect page table gathering in custom attention kernels) is deferred to future custom backends (Vulkan / NEON Phase 2).

4. **Dynamic Mobile Memory Budgeting:**
   - `MemoryBudgetManager` queries physical system RAM dynamically (`/proc/meminfo` on Linux/Android, `sysctl`/Mach VM on macOS).
   - Enforces an inviolable safety margin of at least 512 MB reserved for the Android OS and host application UI.
   - Caps total model weights, active KV cache, and scratch buffers at $\le 70\%$ of available RAM.
   - `EngineImpl::load_model()` and `EngineImpl::create_session()` gate allocations through `MemoryBudgetManager::can_allocate()`, failing cleanly with `StatusCode::OUT_OF_MEMORY` instead of risking `lmkd` process death.

5. **Android OS Lifecycle Integration (`onTrimMemory`):**
   - Add `mllm_engine_trim_memory()` to the C API and `nativeTrimMemory()` to the Android JNI bridge.
   - Plumb Android `ComponentCallbacks2.onTrimMemory(level)` from `MobileLLMClient.kt` down to `MemoryBudgetManager`.
   - On `TRIM_MEMORY_RUNNING_CRITICAL` and `TRIM_MEMORY_UI_HIDDEN`, trigger emergency preemption of idle sessions.

6. **LRU Session Eviction with Transparent Context Rehydration:**
   - When memory is constrained or a critical trim event occurs, the least recently used (LRU) idle sessions are evicted.
   - Eviction releases the backend's heavy `llama_context` (saving 50–250 MB per session) while preserving historical token IDs in `SessionMetadata`.
   - When a preempted session receives a new prompt, the scheduler transparently allocates a new context and re-prefills the token history before continuing autoregressive generation.

7. **Strict Active Context Invariant:**
   - **Session eviction must never destroy an active backend context.** Active sessions currently in generation are strictly protected and exempt from preemption.

8. **Configurable Chunked vs. Single-Burst Prefill:**
   - Long prompts (> 256 tokens) are prefilled in configurable 256-token chunks across scheduling iterations to eliminate starvation and stuttering for concurrent token decoding.
   - Short prompts ($\le 256$ tokens) use single-burst prefill for lowest Time-to-First-Token (TTFT) latency.

9. **Configurable Session History Retention:**
   - Preempted sessions retain conversation history up to a default limit of 2,048 tokens (user-configurable via `SessionConfig::max_history_tokens`). This prevents unbounded native memory consumption for idle metadata.

10. **Thread-Safe Cooperative Cancellation:**
    - Cancellation requests (`mllm_cancel()` or token callback returning `false`) update an atomic boolean flag.
    - Requests in the queue are cancelled instantly; requests currently decoding in the active batch retire at the boundary of the next token step ($< 40\text{ ms}$ latency).

---

## Consequences

### Positive:
- **Zero OOM SIGKILL Crashes:** Pre-checked memory budgeting and emergency trim eviction prevent uncatchable Android process kills.
- **Elimination of Core Thrashing:** Replacing ad-hoc per-request `std::thread` workers with a dedicated scheduler worker thread stabilizes CPU utilization and prevents thermal throttling on ARM big.LITTLE architectures.
- **Fair Prioritized Scheduling:** Interactive UI requests preempt or take priority over background batch workloads.
- **Zero Phantom Memory Waste:** Removing redundant heap vectors in `KVCacheManager` saves tens of megabytes of RAM.
- **Hardware Backend Portability:** Keeping `IBackend` clean preserves future compatibility with Vulkan GPU and vendor NPU backends.
- **ABI Stability:** Integration requires no breaking changes to existing C-API signatures (`c_api.h`) or Kotlin interfaces.

### Negative / Trade-offs:
- **Context Rehydration Latency:** Resuming an evicted session requires a prompt prefill pass over historical tokens, incurring a transient Time-to-First-Token (TTFT) latency overhead (typically 100–300 ms on mobile CPU).
- **Chunked Prefill Scheduling Overhead:** Chunking long prompts adds slight book-keeping overhead across iterations, traded off against avoiding decode thread starvation.

---

## Alternatives Considered & Rejected Options

1. **Option A: Embed Scheduler inside `IBackend`**  
   - *Description:* Implement scheduling, queuing, and memory management inside `LlamaCppBackend`.  
   - *Rejection Reason:* Violates the Single Responsibility Principle and destroys backend portability. Every subsequent backend (`VulkanBackend`, `NPUBackend`) would have to duplicate queue synchronization, priority handling, and Android lifecycle callbacks. Furthermore, memory budgets are global to the mobile device, not specific to an individual backend.

2. **Option B: Retain Ad-Hoc `std::thread` Spawning in `EngineImpl`**  
   - *Description:* Keep launching independent threads per request, using mutexes to serialize access to the backend.  
   - *Rejection Reason:* Causes thread contention, cache thrashing, and high latency jitter on asymmetric mobile CPU cores. Lacks priority scheduling, fair queuing, and batching capabilities.

3. **Option C: Implement True Paged Attention Immediately by Forking GGML Kernels**  
   - *Description:* Patch `llama.cpp`'s internal GGML computation graph to support non-contiguous block table lookups in FlashAttention kernels.  
   - *Rejection Reason:* Explicitly rejected per project guidelines ("Do not replace llama.cpp", "Do not claim paged attention exists"). Modifying internal GGML kernels introduces extreme maintenance overhead, breaks upstream compatibility, and is unnecessary for mobile single-stream and small-batch workloads.

4. **Option D: Monolithic Static Memory Reservation**  
   - *Description:* Pre-allocate a large static buffer (e.g. 2 GB) at startup for all models and contexts.  
   - *Rejection Reason:* Cloud server paradigm that fails catastrophically on Android. Reserving gigabytes upfront triggers immediate background process eviction by Android `lmkd` and starves the host application UI.

---

## References

- Architecture Specification: [`docs/architecture/SCHEDULER_RUNTIME_INTEGRATION.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/architecture/SCHEDULER_RUNTIME_INTEGRATION.md)
- Foundational Memory ADR: [`docs/adr/0006-mobile-memory-budgeting-and-paged-kv-cache.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0006-mobile-memory-budgeting-and-paged-kv-cache.md)
- Backend Architecture: [`docs/architecture/backend-limitations.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/architecture/backend-limitations.md)
