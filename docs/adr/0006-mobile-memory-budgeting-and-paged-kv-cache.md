# ADR-0006: Mobile Memory Budgeting, Paged KV-Cache & Zero SQLite Tensors

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect) & Agent 2 (Scheduler Engineer)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 3 (Android)

---

## Context

On Android, there is no swap space and no dedicated VRAM. Memory is shared between the Android operating system, background processes, the user interface, and inference buffers. Exceeding available memory triggers the Android Low Memory Killer Daemon (`lmkd`), sending an uncatchable `SIGKILL` to the process.

Server-grade LLM runtimes (such as standard vLLM) assume gigabytes of dedicated GPU memory and allocate vast contiguous arrays upfront. Direct application of server memory paradigms to mobile leads to immediate process death.

Additionally, architectural boundaries must be set regarding what can and cannot be stored in relational databases (SQLite).

## Decision

We design a **mobile-aware memory management subsystem** featuring dynamic budgeting, paged virtual KV blocks, and a strict separation between tensor memory and relational metadata.

### Architectural Rules:
1. **Dynamic Memory Budgeting:**
   - Before model loading, query available system memory (`/proc/meminfo` or `ActivityManager.MemoryInfo`).
   - Reserve an untouchable safety margin of at least 512MB for the OS and foreground application.
   - Limit total model + KV-cache allocation to at most 70% of available memory.
   - If a requested model exceeds this budget, reject loading immediately with `StatusCode::OUT_OF_MEMORY`.
2. **Paged KV-Cache Architecture:**
   - Instead of allocating a single large tensor for maximum context upfront, allocate KV-cache memory in discrete virtual blocks of 16 tokens.
   - Allocate blocks dynamically as tokens are generated; free blocks immediately when a generation finishes or a session is closed.
   - Share identical prompt prefix blocks across multi-turn chat sessions to save RAM.
3. **Android Lifecycle Integration:**
   - Connect to Android's `ComponentCallbacks2.onTrimMemory(level)`.
   - On `TRIM_MEMORY_RUNNING_CRITICAL`, immediately flush inactive session KV caches and truncate context history to root checkpoints.
4. **Strict SQLite Scope: Zero Tensors in Relational Storage:**
   - **SQLite is strictly for metadata:** Session IDs, model names, file paths, quantization tags, benchmark timestamps, and user settings.
   - **Never store KV-cache tensors or weight tensors in SQLite.** Tensors belong exclusively in native memory buffers managed by the inference backend. Storing tensors in SQLite would destroy memory locality, add massive serialization overhead, and cause severe disk thrashing.

## Consequences

### Positive:
- Drastic reduction in Android OOM crash rates.
- Efficient multi-turn conversation caching via virtual block sharing.
- Clean architectural boundary between relational metadata and high-performance native tensor memory.

### Negative / Trade-offs:
- Slightly increased logic complexity in tracking virtual block tables and memory budgets.

## Alternatives Considered

- **Pre-allocating Contiguous Context Buffer (Static KV-Cache):** Traditional approach in minimal inference engines. Wastes hundreds of megabytes of RAM when generating short responses, starving the mobile OS and provoking `lmkd` kills.
- **Serializing KV-Cache to SQLite Disk Storage:** Explicitly rejected per `CONTEXT.md` Section 2 ("Non-goals: Store inference tensors or KV-cache contents in SQLite"). SQLite disk I/O latency (even on UFS 3.1/4.0 flash storage) is orders of magnitude too slow for real-time autoregressive token generation.
