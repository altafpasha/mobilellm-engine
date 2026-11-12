# MobileLLM — Milestone 1 Implementation Report

**Milestone:** Milestone 1 — Scheduler Build Linkage and Memory Admission Control  
**Date:** November 13, 2026  
**Status:** COMPLETE & VERIFIED  
**Governing Architecture:** [ADR-0008: Scheduler Runtime Integration](../docs/adr/ADR-0008-scheduler-runtime-integration.md) & [SCHEDULER_RUNTIME_INTEGRATION.md](../docs/architecture/SCHEDULER_RUNTIME_INTEGRATION.md)

---

## 1. Executive Summary

Milestone 1 connects the `mobilellm_scheduler` subsystem to the primary `MobileLLM` runtime and Android JNI build targets, establishes explicit runtime ownership of the memory manager and scheduler, and introduces pre-flight memory admission control into model loading and session creation.

All 10 approved architectural decisions from `ADR-0008` have been respected:
- `ContinuousBatchScheduler` and `MemoryBudgetManager` are owned directly by `EngineImpl` as runtime coordination services.
- Scheduling and memory admission remain strictly above `IBackend`.
- Memory budgeting is runtime-managed, distinguishing estimated reservations from measured backend allocations.
- Physical KV-cache ownership remains with `llama.cpp` (virtual block accounting only; zero dummy buffer allocations).
- Public C++, C, Android Kotlin/JNI, and Python SDK APIs and ABIs are completely preserved.

Validation confirms 100% test pass rate across all suites:
- **C++ Tests:** 59/59 passed (100%)
- **Python Tests:** 34 passed, 1 skipped (100% pass)
- **Android SDK Build:** Successful assembly of debug AAR for both `arm64-v8a` and `x86_64` targets with `mobilellm_scheduler` linked.

---

## 2. Exact Files Changed

### Build System & Linkage
| File | Changes Made |
| :--- | :--- |
| `CMakeLists.txt` | Propagated `mobilellm_scheduler` library dependency across runtime and test targets. |
| `runtime/CMakeLists.txt` | Added scheduler include paths (`${MOBILELLM_ROOT_DIR}/scheduler/include`), linked `mobilellm_scheduler` to `mobilellm` (shared), `mobilellm_static`, and test executables. Added `test_scheduler_integration.cpp`. |
| `scheduler/CMakeLists.txt` | Added `find_package(Threads REQUIRED)`, resolved include paths to use `${MOBILELLM_ROOT_DIR}/runtime/include` instead of `CMAKE_SOURCE_DIR` for clean NDK cross-compilation. |
| `android/jni/CMakeLists.txt` | Added `find_package(Threads REQUIRED)`, added `${MOBILELLM_ROOT_DIR}/scheduler/include` to includes, and linked `mobilellm_scheduler` to the native JNI shared library. |

### Scheduler Subsystem
| File | Changes Made |
| :--- | :--- |
| `scheduler/include/mobilellm/scheduler/scheduler.h` | Added dependency injection constructor accepting external `std::shared_ptr<MemoryBudgetManager>`. Changed default `bytes_per_kv_block` from 128KB to 0 (virtual-only block accounting). |
| `scheduler/src/scheduler.cpp` | Implemented DI constructor; initialized scheduler components with zero-byte virtual blocks to prevent 32 MB dummy buffer allocations. |
| `scheduler/src/kv_cache/kv_cache_manager.cpp` | Allowed `bytes_per_block == 0` for virtual block tracking without physical heap memory allocation. |

### Runtime Engine Implementation
| File | Changes Made |
| :--- | :--- |
| `runtime/src/engine/engine_impl.h` | Added `std::shared_ptr<scheduler::MemoryBudgetManager> memory_mgr_` and `std::unique_ptr<scheduler::ContinuousBatchScheduler> scheduler_`. Added `calculate_model_memory_bytes` and `calculate_session_kv_bytes`. Added `session_memory_reserved_` map. Added introspection getters `memory_manager()` and `scheduler()` for testing. |
| `runtime/src/engine/engine_impl.cpp` | Initialized `MemoryBudgetManager` and `ContinuousBatchScheduler` in `EngineImpl::initialize`. Integrated pre-flight admission checks in `load_model` and `create_session` returning `StatusCode::OUT_OF_MEMORY` when budget exceeded. Implemented failure rollback logic. Implemented de-allocation and reservation release in `destroy_session`, `unload_model`, and `shutdown`. Updated `get_metrics()` to surface memory budget metrics. |

### Backends & Testing
| File | Changes Made |
| :--- | :--- |
| `backends/mock/mock_backend.h` & `.cpp` | Added `simulate_oom_` check in `create_session` to enable testing reservation rollbacks on backend failures. |
| `runtime/tests/test_scheduler_integration.cpp` | Created 10 comprehensive unit/integration tests verifying scheduler initialization, memory reservation, budget exhaustion, rollback on failure, model unload cleanup, and multi-threaded reservations. |

### Documentation & ADRs
| File | Changes Made |
| :--- | :--- |
| `docs/architecture/SCHEDULER_RUNTIME_INTEGRATION.md` | Updated sections 6.2, 6.3, 8.4, and added Section 15 documenting approved decisions. |
| `docs/adr/ADR-0008-scheduler-runtime-integration.md` | Updated status to Approved & Accepted with all 10 architectural decisions. |
| `docs/adr/0008-scheduler-runtime-integration.md` | Mirror of ADR-0008 updated to Approved. |
| `docs/adr/README.md` | Added ADR-0008 entry. |

---

## 3. Architecture Changes Implemented

```
+-----------------------------------------------------------------------------------+
|                                  EngineImpl                                       |
|                                                                                   |
|  +--------------------------------+       +------------------------------------+  |
|  |      MemoryBudgetManager       | <---  |      ContinuousBatchScheduler      |  |
|  |  (Ledger & Admission Authority)|       |   (Virtual Sequence & Request Mgr) |  |
|  +--------------------------------+       +------------------------------------+  |
|         ^                   ^                                                     |
|         |                   |                                                     |
|  [Model Admission]   [Session Admission]                                          |
|         |                   |                                                     |
+---------|-------------------|-----------------------------------------------------+
          v                   v
+-----------------------------------------------------------------------------------+
|                                   IBackend                                        |
|                       (MockBackend / LlamaCppBackend)                             |
|                                                                                   |
|  Physical Weights Allocation & llama_context (Physical KV-Cache via llama.cpp)    |
+-----------------------------------------------------------------------------------+
```

### 1. Explicit Component Ownership and Initialization
- `EngineImpl` explicitly constructs and owns:
  - `std::shared_ptr<scheduler::MemoryBudgetManager> memory_mgr_`
  - `std::unique_ptr<scheduler::ContinuousBatchScheduler> scheduler_`
- When initialized, `EngineImpl` queries the system total memory (or accepts a configurable `RuntimeConfig::memory_budget_bytes`) to establish the budget ceiling.
- The `ContinuousBatchScheduler` is constructed using dependency injection with `bytes_per_kv_block = 0`, ensuring that the scheduler does not allocate dummy heap memory for KV blocks.

### 2. Pre-Flight Memory Admission Control
- **Model Loading (`load_model`):**
  1. Computes estimated footprint from model file size and base execution overhead.
  2. Queries `memory_mgr_->reserve(ReservationType::kModelWeights, estimated_bytes)`.
  3. If reservation fails, immediately returns `StatusCode::OUT_OF_MEMORY` with a detailed error message before touching backend loaders.
  4. Calls `backend_->load_model(config)`. If backend loading fails, calls `memory_mgr_->release(ReservationType::kModelWeights, estimated_bytes)` to roll back cleanly.
  5. If loading succeeds, measures actual resident memory and adjusts the reservation ledger.
- **Session Creation (`create_session`):**
  1. Computes estimated KV-cache memory requirement:
     $$\text{KV bytes} = \text{context\_length} \times N_{\text{layers}} \times 2 \times N_{\text{heads}} \times D_{\text{head}} \times \text{sizeof}(\text{element})$$
  2. Queries `memory_mgr_->reserve(ReservationType::kKVCache, kv_bytes)`.
  3. If reservation fails, returns `StatusCode::OUT_OF_MEMORY`.
  4. Calls `backend_->create_session(config)`. If backend creation fails, rolls back the reservation via `memory_mgr_->release(ReservationType::kKVCache, kv_bytes)`.
  5. Tracks the reserved amount per `session_id` in `session_memory_reserved_`.

### 3. Cleanup & Teardown Invariants
- `destroy_session(session_id)` releases the exact reserved KV bytes associated with the session.
- `unload_model()` cleans up model weight reservations and any associated active session KV reservations.
- `shutdown()` performs complete reservation and resource teardown.

---

## 4. Build Commands and Actual Outputs

### 1. Host C++ Build & Test Execution
```bash
cmake --build build && ctest --test-dir build --output-on-failure
```
**Output:**
```
[ 16%] Built target mobilellm_scheduler
[ 35%] Built target mobilellm_static
[ 54%] Built target mobilellm_scheduler_tests
[ 72%] Built target mobilellm
[ 94%] Built target mobilellm_tests
[100%] Built target mobilellm-cli
Test project /Users/Apple/Developer/ai-apps/MobileLLM/build
      Start  1: MemoryManagerTest.QuerySystemMemory
 1/59 Test  #1: MemoryManagerTest.QuerySystemMemory .........................................   Passed    0.01 sec
...
      Start 50: SchedulerIntegrationTest.SchedulerInitializationAndShutdown
50/59 Test #50: SchedulerIntegrationTest.SchedulerInitializationAndShutdown .................   Passed    0.00 sec
      Start 51: SchedulerIntegrationTest.ModelMemoryReservation
51/59 Test #51: SchedulerIntegrationTest.ModelMemoryReservation .............................   Passed    0.00 sec
      Start 52: SchedulerIntegrationTest.SessionMemoryReservation
52/59 Test #52: SchedulerIntegrationTest.SessionMemoryReservation ...........................   Passed    0.00 sec
      Start 53: SchedulerIntegrationTest.BudgetExhaustionReturnsOOM
53/59 Test #53: SchedulerIntegrationTest.BudgetExhaustionReturnsOOM .........................   Passed    0.00 sec
      Start 54: SchedulerIntegrationTest.SessionBudgetExhaustionReturnsOOM
54/59 Test #54: SchedulerIntegrationTest.SessionBudgetExhaustionReturnsOOM ..................   Passed    0.00 sec
      Start 55: SchedulerIntegrationTest.ReservationRollbackOnModelLoadFailure
55/59 Test #55: SchedulerIntegrationTest.ReservationRollbackOnModelLoadFailure ..............   Passed    0.00 sec
      Start 56: SchedulerIntegrationTest.ReservationRollbackOnSessionCreateFailure
56/59 Test #56: SchedulerIntegrationTest.ReservationRollbackOnSessionCreateFailure ..........   Passed    0.00 sec
      Start 57: SchedulerIntegrationTest.ModelUnloadReleasesAssociatedSessionReservations
57/59 Test #57: SchedulerIntegrationTest.ModelUnloadReleasesAssociatedSessionReservations ...   Passed    0.00 sec
      Start 58: SchedulerIntegrationTest.MultipleIndependentSessionsAccounting
58/59 Test #58: SchedulerIntegrationTest.MultipleIndependentSessionsAccounting ..............   Passed    0.00 sec
      Start 59: SchedulerIntegrationTest.ConcurrentMemoryReservation
59/59 Test #59: SchedulerIntegrationTest.ConcurrentMemoryReservation ........................   Passed    0.01 sec

100% tests passed out of 59

Total Test time (real) =   0.25 sec
```

### 2. Android Cross-Compilation & AAR Assembly
```bash
cd android && export JAVA_HOME="/opt/homebrew/Cellar/openjdk@21/21.0.13.0.3/libexec/openjdk.jdk/Contents/Home" && ./gradlew :sdk:assembleDebug
```
**Output:**
```
> Task :sdk:preBuild UP-TO-DATE
> Task :sdk:preDebugBuild UP-TO-DATE
> Task :sdk:configureCMakeDebug[arm64-v8a]
> Task :sdk:buildCMakeDebug[arm64-v8a]
> Task :sdk:configureCMakeDebug[x86_64]
> Task :sdk:buildCMakeDebug[x86_64]
> Task :sdk:mergeDebugJniLibFolders UP-TO-DATE
> Task :sdk:checkKotlinGradlePluginConfigurationErrors
> Task :sdk:generateDebugResValues UP-TO-DATE
> Task :sdk:generateDebugResources UP-TO-DATE
> Task :sdk:packageDebugResources UP-TO-DATE
> Task :sdk:parseDebugLocalResources UP-TO-DATE
> Task :sdk:processDebugManifest UP-TO-DATE
> Task :sdk:generateDebugRFile UP-TO-DATE
> Task :sdk:mergeDebugNativeLibs
> Task :sdk:compileDebugKotlin UP-TO-DATE
> Task :sdk:extractDebugAnnotations UP-TO-DATE
> Task :sdk:extractDeepLinksForAarDebug UP-TO-DATE
> Task :sdk:javaPreCompileDebug UP-TO-DATE
> Task :sdk:compileDebugJavaWithJavac NO-SOURCE
> Task :sdk:mergeDebugGeneratedProguardFiles UP-TO-DATE
> Task :sdk:mergeDebugConsumerProguardFiles UP-TO-DATE
> Task :sdk:mergeDebugShaders UP-TO-DATE
> Task :sdk:compileDebugShaders NO-SOURCE
> Task :sdk:generateDebugAssets UP-TO-DATE
> Task :sdk:packageDebugAssets UP-TO-DATE
> Task :sdk:prepareDebugArtProfile UP-TO-DATE
> Task :sdk:prepareLintJarForPublish UP-TO-DATE
> Task :sdk:processDebugJavaRes UP-TO-DATE
> Task :sdk:mergeDebugJavaResource UP-TO-DATE
> Task :sdk:syncDebugLibJars UP-TO-DATE
> Task :sdk:writeDebugAarMetadata UP-TO-DATE
> Task :sdk:stripDebugDebugSymbols
> Task :sdk:copyDebugJniLibsProjectAndLocalJars
> Task :sdk:bundleDebugAar
> Task :sdk:assembleDebug

BUILD SUCCESSFUL in 19s
29 actionable tasks: 9 executed, 20 up-to-date
```

### 3. Python SDK Test Suite
```bash
python/.venv/bin/pytest python/tests
```
**Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.15.2, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/Apple/Developer/ai-apps/MobileLLM/python
configfile: pyproject.toml
plugins: anyio-4.16.0, asyncio-1.4.1.post1
collected 35 items

python/tests/test_cancellation_timeout.py ...                            [  8%]
python/tests/test_cli.py ....                                            [ 20%]
python/tests/test_client_async.py ..                                     [ 25%]
python/tests/test_client_sync.py ....                                    [ 37%]
python/tests/test_exceptions.py .....                                    [ 51%]
python/tests/test_integration_native.py ...                              [ 60%]
python/tests/test_models.py .......                                      [ 80%]
python/tests/test_server.py ......s                                      [100%]

=================== 34 passed, 1 skipped in 68.24s (0:01:08) ===================
```

---

## 5. Memory Accounting Assumptions

### 1. Estimated vs Measured Memory Footprint
- **Model Weights Estimation:**
  Estimated as `file_size_bytes` on disk plus a 15% execution scratchpad margin. On mobile devices with memory mapping (`mmap`), page faults load weights on demand; however, admission control treats weights as pinned resident memory to prevent mobile OS low-memory killer (LMK) kills during peak inference.
- **Model Weights Reconciliation (Measured):**
  After `backend_->load_model` successfully completes, the runtime queries `backend_->get_metrics().resident_memory_bytes` or platform RSS delta. If the actual measured resident size differs from the estimate, the reservation ledger in `MemoryBudgetManager` is adjusted by the delta, ensuring accounting accuracy.
- **KV-Cache Size Calculation:**
  Calculated using standard transformer memory geometry:
  $$\text{KV bytes} = 2 \times \text{layers} \times \text{context\_length} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_elem}$$
  For standard 16-bit half-precision (FP16) caches in llama.cpp, this yields `context_length * layers * 2 * n_embd * sizeof(uint16_t)`. For test/mock backends, default is `context_length * 512 * sizeof(float)`.

### 2. Elimination of Dummy Allocations
- In prior scheduler standalone tests, `KVCacheManager` allocated raw heap buffers (`bytes_per_kv_block = 131,072 bytes`).
- In runtime integration, `bytes_per_kv_block` is set to `0`. `KVCacheManager` allocates zero buffer memory, operating strictly as a virtual block manager and admission tracker.
- Physical memory allocation remains 100% inside `llama.cpp`'s `llama_context`.

---

## 6. Remaining Limitations & Non-Implemented Features (As Scoped)

In strict accordance with the Milestone 1 boundaries:
1. **Request Execution Delegation (Milestone 2):**  
   Inference requests (`generate_stream`) currently invoke the backend directly. Forwarding requests through the scheduler queue and cooperative stepping will be implemented in Milestone 2.
2. **Chunked Prefill (Milestone 3):**  
   Prompt evaluation is not yet split into 256-token chunks; prompts are evaluated in a single burst.
3. **Session Eviction & Android `onTrimMemory` (Milestone 4):**  
   Low memory triggers do not yet evict inactive sessions. Active context protection invariants have been designed and documented, but runtime eviction handler hooks await Milestone 4.
4. **Paged Attention & Continuous Batching:**  
   No paged attention kernels are implemented; llama.cpp owns the contiguous KV cache. Virtual block sequence tracking will be mapped in Milestone 5.

---

## 7. SDK & Interface Compatibility Confirmation

- **C++ Public Interface (`runtime/include/mobilellm/engine.h`):**  
  All public methods and classes maintain exact signatures and semantics. No changes made to headers that consumers link against.
- **C API (`runtime/include/mobilellm/c_api.h`):**  
  ABI completely intact. All function exports (`mobilellm_engine_create`, `mobilellm_model_load`, `mobilellm_session_create`, etc.) are unchanged.
- **Android Kotlin SDK:**  
  Kotlin bindings in `android/sdk/` build cleanly with no modifications required.
- **Python SDK:**  
  Native bindings load the shared library without error; all 34 active tests pass.

---

## 8. Conclusion

Milestone 1 is complete, verified, and adheres strictly to approved architectural guidelines. The codebase is ready for review before proceeding to Milestone 2 (Request Execution Delegation).

