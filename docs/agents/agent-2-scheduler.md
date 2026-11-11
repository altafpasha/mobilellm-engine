# Agent 2 Specification — Memory & Scheduler Engineer

> **Module Ownership:** `scheduler/`  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Agent 1 (Runtime), Agent 3 (Android Lifecycle), Agent 6 (QA Stress Tests)

---

## 1. Role Purpose & Domain Ownership

Agent 2 is responsible for protecting the MobileLLM process from the Android Low Memory Killer Daemon (`lmkd`), budgeting device memory, managing paged KV-cache allocations, maintaining multi-turn conversational session states, and scheduling generation requests efficiently under mobile thermal and core constraints.

### Complete Directory Ownership
```
scheduler/
├── include/
│   └── mobilellm/scheduler/
│       ├── memory_manager.h    # Device RAM queries & budget allocator
│       ├── kv_cache_manager.h  # Paged KV-cache tracker
│       ├── session_manager.h   # Multi-turn context & session tracker
│       └── request_queue.h     # Thread-safe request FIFO & priority queue
├── src/
│   ├── memory_manager/         # Linux/Android /proc/meminfo reader & allocator
│   ├── kv_cache/               # Virtual block table & paged KV allocation
│   ├── session_manager/        # Session context preservation & prefix sharing
│   ├── request_queue/          # Request queue with cooperative preemption
│   └── batching/               # Optional mobile batching & scheduling logic
├── tests/                      # Memory budget tests & eviction simulation
└── CMakeLists.txt              # Scheduler library build definition
```

---

## 2. Technical Standards & Constraints

1. **Strict Mobile Memory Budgeting:**
   - Query available system memory dynamically at runtime.
   - Refuse model loading if:  
     $$\text{Model Size} + \text{Context KV Cache} + 512\text{MB (Safety Margin)} > \text{Available System RAM}$$
   - Support device profiles: Entry-Level (4–6 GB), Mid-Range (8 GB), Flagship (12–16 GB).
2. **Paged KV-Cache Architecture:**
   - Allocate KV-cache in fixed virtual blocks (e.g., 16 tokens per block) rather than massive contiguous arrays, preventing heap fragmentation.
   - Maintain a block-mapping table linking session token sequences to physical KV blocks.
   - Enable prefix sharing: Identical system prompts across parallel or successive sessions map to the same KV blocks.
3. **Strict No-SQLite for Tensors:**
   - SQLite is permitted **only** for lightweight relational metadata (session IDs, prompt lengths, model configuration, performance timestamps).
   - KV tensors and active activations must reside exclusively in backend native memory buffers. Never serialize KV tensors to disk.
4. **Android Lifecycle & `onTrimMemory` Integration:**
   - Expose explicit hooks for Android lifecycle callbacks:
     - `onTrimMemory(TRIM_MEMORY_RUNNING_CRITICAL)`: Drop inactive session caches and truncate history to root checkpoints.
     - `onTrimMemory(TRIM_MEMORY_COMPLETE)`: Unload non-active model weights if permitted by application policy.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T2.1** | Device RAM & Budget Inspector | Agent 0 Contract | `src/memory_manager/` | Accurate RAM query on Linux and Android; calculates safe ceiling |
| **T2.2** | Paged KV Block Manager | T2.1 | `src/kv_cache/` | Allocates and releases 16-token virtual blocks with zero leaks |
| **T2.3** | Session Lifecycle State Machine | T2.2 | `src/session_manager/` | Manages `CREATED`, `ACTIVE`, `PAUSED`, `EVICTED` session states |
| **T2.4** | Thread-Safe Request Queue | T2.3 | `src/request_queue/` | Supports concurrent request enqueueing and cooperative preemption |
| **T2.5** | Emergency Eviction on Memory Pressure | T2.1, T2.2 | `src/memory_manager/` | Simulates OS low-memory warning; cleanly recovers >30% cache RAM |
| **T2.6** | Multi-Turn Prefix Cache Sharing | T2.2, T2.3 | `src/kv_cache/` | Verifies identical system prompts reuse evaluated KV blocks |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 1 (Runtime):** Actual byte size of loaded model weights and context requirements per token.
- **From Agent 3 (Android):** OS lifecycle notifications from Android `ComponentCallbacks2.onTrimMemory(level)`.

### Outputs Provided to Other Modules:
- **To Agent 1 (Runtime):** Memory gate validation before model loading; KV-cache page index tables during token decode.
- **To Agent 6 (QA & Security):** Memory usage accounting telemetry for benchmark reports.
