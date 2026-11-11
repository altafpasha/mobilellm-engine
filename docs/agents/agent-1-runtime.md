# Agent 1 Specification — C++ Runtime Engineer

> **Module Ownership:** `runtime/`  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Agent 3 (Android JNI), Agent 2 (Scheduler), Agent 6 (QA & Security)

---

## 1. Role Purpose & Domain Ownership

Agent 1 is responsible for the core native execution engine of MobileLLM. This includes the public C-ABI and C++ APIs, GGUF model parsing and loading, tokenizer and detokenizer mechanics, sampling algorithms, execution loop orchestration, and integration with the baseline compute engine (`llama.cpp` / GGML).

### Complete Directory Ownership
```
runtime/
├── include/
│   └── mobilellm/
│       ├── c_api.h            # Canonical C-ABI interface
│       ├── engine.h           # Core C++ engine class interface
│       ├── types.h            # Data structures, error codes, Result<T>
│       ├── model_loader.h     # GGUF parser & loader interface
│       ├── tokenizer.h        # Tokenizer / Detokenizer interface
│       └── sampler.h          # Sampler pipeline interface
├── src/
│   ├── engine/                # Engine orchestration, cancellation handles
│   ├── model_loader/          # GGUF parser, mmap, tensor metadata
│   ├── tokenizer/             # BPE/SentencePiece UTF-8 streaming detokenizer
│   ├── inference/             # Context evaluation, prefill & decode loop
│   └── sampling/              # Logit processing (Temp, Top-K, Top-P, Min-P)
├── tests/                     # GoogleTest unit and integration tests
└── CMakeLists.txt             # Native build definition
```

---

## 2. Technical Standards & Constraints

1. **C++ Standard:** Modern C++20 (`-std=c++20`).
2. **Memory Safety & Resource Ownership:**
   - Strict RAII for all native allocations, file handles, and backend contexts.
   - Zero raw pointer leaks across module boundaries. Use `std::unique_ptr` and `std::shared_ptr` internally; expose opaque integer/pointer handles across the C-ABI (`c_api.h`).
   - All string and buffer inputs from external callers must be checked for `nullptr`, length bounds, and valid UTF-8 encoding.
3. **Threading & Concurrency:**
   - `IEngine` methods must be thread-safe for distinct sessions.
   - Generation execution loops must check the cooperative cancellation flag at every token decode step to ensure cancellation latency < 50ms.
4. **Error Handling Model:**
   - No unhandled C++ exceptions may cross the C-ABI boundary.
   - All internal exceptions must be caught and mapped to `mobilellm::Status` and `StatusCode`.
   - Use `mobilellm::Result<T>` for methods that return either a valid object or an error status.
5. **No Functional Duplication:**
   - Do not re-implement transformer kernels or attention compute from scratch in Phase 1. Delegate math execution to the integrated baseline backend.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T1.1** | Native CMake Build Setup | None | `runtime/CMakeLists.txt` | Builds clean with GoogleTest under `-Wall -Wextra -Werror` |
| **T1.2** | Core Header Definitions | Agent 0 Review | `include/mobilellm/*.h` | Compiles across Linux, macOS, and Android NDK toolchains |
| **T1.3** | GGUF Parser & Validator | T1.2 | `src/model_loader/` | Rejects malformed headers, extracts tensors, parses metadata |
| **T1.4** | Integrate Baseline Backend | T1.1 | `backends/cpu/` bridge | Successfully links GGML/llama.cpp static/object libraries |
| **T1.5** | Streaming Detokenizer | T1.2 | `src/tokenizer/` | Correctly handles multi-byte UTF-8 split tokens across boundaries |
| **T1.6** | Sampler Pipeline | T1.2 | `src/sampling/` | Numerically verified Greedy, Temp, Top-K, Top-P, Min-P |
| **T1.7** | Engine Orchestration | T1.3, T1.4, T1.5, T1.6 | `src/engine/` | End-to-end `load_model -> generate -> cancel -> unload` flow |
| **T1.8** | Native CLI Smoke Test | T1.7 | `tools/developer-cli/` | Host CLI loads 1B model and streams generation to terminal |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 2 (Scheduler):** Resource budget check before loading weights; KV-cache allocation handles during context generation.
- **From Agent 5 (Hardware):** `IBackend` compute context initialized for target device (CPU or Vulkan).

### Outputs Provided to Other Modules:
- **To Agent 3 (Android):** Stable `c_api.h` and compiled `libmobilellm.so` shared library.
- **To Agent 4 (Python / Server):** C++ engine bindings and deterministic token streaming callbacks.
- **To Agent 6 (QA & Security):** Standalone unit test targets and clean symbol exports for fuzzing harnesses.
