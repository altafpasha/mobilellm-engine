# MobileLLM Engine — Agent 6: Comprehensive Security & Reliability Audit

> **Auditor:** Agent 6 (Principal QA, Reliability & Security Engineer)  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Date:** November 14, 2026  
> **Status:** Stage 1 Audit Complete — 8 Findings Documented (1 Critical, 2 High, 3 Medium, 1 Low, 1 Informational)  
> **Governing Strategy:** [`docs/quality/QA_AND_SECURITY_STRATEGY.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/quality/QA_AND_SECURITY_STRATEGY.md)

---

## 1. Audit Scope & Methodology

This security audit covers the complete MobileLLM codebase across all native, Android, and Python layers:
- C++20 Core Runtime (`runtime/src/engine/`, `runtime/src/model_loader/`, `runtime/src/sampling/`)
- Scheduler Subsystem (`scheduler/src/`)
- Compute Backends (`backends/cpu/`, `backends/common/`, `backends/mock/`)
- C API and ABI Boundary (`runtime/include/mobilellm/c_api.h`, `runtime/src/engine/c_api.cpp`)
- Android JNI Bridge (`android/jni/mobilellm_jni.cpp`)
- Android Kotlin SDK (`android/sdk/src/`)
- Python SDK and FFI Layer (`python/mobilellm/`)

Testing methods included manual adversarial code review, static boundary inspection, concurrency race analysis, ThreadSanitizer test runs, and integer bounds evaluation.

---

## 2. Findings Summary Matrix

| Finding ID | Severity | Category | Affected Component | Title | Status |
| :---: | :---: | :---: | :---: | :--- | :---: |
| **SEC-01** | **CRITICAL** | Memory Safety | `MemoryBudgetManager` | Integer Overflow in Memory Admission Control | **VERIFIED** |
| **SEC-02** | **HIGH** | Reliability / DoS | `ContinuousBatchScheduler` | Unhandled Exceptions in `noexcept` Callbacks Causing `std::terminate` | **VERIFIED** |
| **SEC-03** | **HIGH** | Concurrency / UAF | `EngineImpl` | Potential Use-After-Free on Session Destruction During Active Inference | **VERIFIED** |
| **SEC-04** | **MEDIUM** | Android Stability | JNI Bridge (`mobilellm_jni.cpp`) | Missing `DetachCurrentThread` on Native Scheduler Worker Thread Termination | **VERIFIED** |
| **SEC-05** | **MEDIUM** | FFI / Memory Safety| Python Native Client | Premature Garbage Collection of ctypes Callbacks During Generator Lifecycles | **VERIFIED** |
| **SEC-06** | **MEDIUM** | DoS / Bounds | `ModelLoader` | Unbounded Array Loop & Stack Exhaustion in GGUF Metadata Parser | **VERIFIED** |
| **SEC-07** | **LOW** | Robustness | C API (`c_api.cpp`) | Silent Output Truncation in `mllm_model_inspect` Without Error Status | **VERIFIED** |
| **SEC-08** | **INFORMATIONAL**| Correctness | Core HAL / C API | Synthetic Hardware Capabilities Reported Across Public APIs | **VERIFIED** |

---

## 3. Detailed Security Findings

---

### Finding SEC-01 (CRITICAL): Integer Overflow in Memory Admission Control

- **Category:** Memory Safety / Integer Overflow / Admission Bypass
- **Severity:** **CRITICAL**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`scheduler/src/memory_manager/memory_manager.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/scheduler/src/memory_manager/memory_manager.cpp#L193-L208)

#### Description & Code Analysis:
In `MemoryBudgetManager::can_allocate` and `allocate`:
```cpp
bool MemoryBudgetManager::can_allocate(uint64_t bytes, AllocationType /*type*/) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t current = model_allocated_ + kv_allocated_ + scratch_allocated_;
    return (current + bytes) <= budget_ceiling_;
}

Status MemoryBudgetManager::allocate(uint64_t bytes, AllocationType type) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t current = model_allocated_ + kv_allocated_ + scratch_allocated_;
    if (current + bytes > budget_ceiling_) {
        return Status::error(StatusCode::OUT_OF_MEMORY, ...);
    }
    // ...
    model_allocated_ += bytes;
    return Status::ok();
}
```

#### Vulnerability & Reproduction:
1. `current` holds existing allocations (e.g. $1\text{ GB} = 1{,}073{,}741{,}824\text{ bytes}$).
2. A malformed GGUF model file or hostile configuration supplies a tensor size or memory budget `bytes = 0xFFFFFFFFFF000000ULL` ($2^{64} - 16\text{ MB}$).
3. The addition `current + bytes` wraps around `uint64_t` due to unsigned integer overflow:
   $$1{,}073{,}741{,}824 + (2^{64} - 16\text{ MB}) = 1{,}056{,}964{,}608 < \text{budget\_ceiling\_}$$
4. `can_allocate` evaluates to `true`! `allocate` succeeds!
5. `model_allocated_ += bytes` wraps around to a small number.
6. The engine attempts to allocate hundreds of gigabytes or invalid buffers in downstream components, completely bypassing memory admission limits and causing an uncatchable Android Low Memory Killer (`lmkd`) `SIGKILL` termination.

#### Remediation:
Enforce safe saturation arithmetic prior to addition:
```cpp
bool MemoryBudgetManager::can_allocate(uint64_t bytes, AllocationType /*type*/) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes > budget_ceiling_) {
        return false;
    }
    uint64_t current = model_allocated_ + kv_allocated_ + scratch_allocated_;
    if (current > budget_ceiling_ - bytes) {
        return false;
    }
    return true;
}
```

---

### Finding SEC-02 (HIGH): Unhandled Exceptions in `noexcept` Callbacks Causing `std::terminate`

- **Category:** Availability / Denial of Service / Exception Safety
- **Severity:** **HIGH**
- **Verification Status:** **VERIFIED**
- **Affected Files:** [`scheduler/src/scheduler.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/scheduler/src/scheduler.cpp#L264-L270), [`scheduler/src/scheduler.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/scheduler/src/scheduler.cpp#L327-L329), and [`scheduler/src/request_queue/request_queue.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/scheduler/src/request_queue/request_queue.cpp#L98-L103)

#### Description & Code Analysis:
In `ContinuousBatchScheduler::step_batch() noexcept`:
```cpp
uint32_t ContinuousBatchScheduler::step_batch() noexcept {
    // ...
    if (req->token_cb) {
        TokenChunk chunk{piece, tok_id, false, 0.0f};
        keep_going = req->token_cb(chunk); // UNPROTECTED USER CALLBACK
    }
    // ...
    for (auto& item : to_complete) {
        if (item.cb) item.cb(item.status, item.metrics); // UNPROTECTED USER CALLBACK
    }
}
```
And in `RequestQueue::dequeue(...) noexcept`:
```cpp
for (auto& item : cancelled_items) {
    if (item->complete_cb) {
        item->complete_cb(Status::error(StatusCode::CANCELLED, "Request cancelled in queue"), item->metrics);
    }
}
```

#### Vulnerability & Reproduction:
1. `ContinuousBatchScheduler::step_batch()` and `RequestQueue::dequeue()` are explicitly decorated with `noexcept`.
2. Under C++20 standard rules ([except.terminate]), if an exception escapes from a function marked `noexcept`, `std::terminate()` is invoked immediately.
3. If a C++ application provides a `token_cb` or `complete_cb` that throws `std::bad_alloc`, `std::runtime_error`, or any other exception, the exception escapes through `step_batch()`.
4. The entire application process is instantly aborted with `SIGABRT`, with no opportunity for cleanup, state flushing, or diagnostic logging.

#### Remediation:
Wrap all external callback invocations in `try/catch` handlers inside `step_batch()` and `dequeue()`. If `token_cb` throws, treat it as an implicit cancellation request (`keep_going = false`); if `complete_cb` throws, absorb the exception safely to protect the runtime process.

---

### Finding SEC-03 (HIGH): Potential Use-After-Free on Session Destruction During Active Inference

- **Category:** Concurrency / Use-After-Free / Memory Corruption
- **Severity:** **HIGH**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`runtime/src/engine/engine_impl.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/src/engine/engine_impl.cpp#L404-L430)

#### Description & Code Analysis:
In `EngineImpl::destroy_session(SessionHandle session)`:
```cpp
in_flight_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this, session]() {
    for (const auto& [_, info] : active_requests_) {
        if (info && info->session_handle == session) return false;
    }
    return true;
});

std::lock_guard<std::mutex> lock(engine_mutex_);
// ...
Status st = backend_->destroy_session(it->second.backend_handle);
sessions_.erase(it);
```

#### Vulnerability & Reproduction:
1. An active request is in prefill or decode on the scheduler worker thread.
2. The user calls `destroy_session(session)`.
3. Cancellation is requested via `scheduler_->cancel_request()`.
4. However, if the backend is currently blocked in a heavy prompt evaluation (`evaluate_prompt` on a 2,048-token context) that takes $> 1{,}000\text{ ms}$, `in_flight_cv_.wait_for()` times out after 1.0 second.
5. The function proceeds to call `backend_->destroy_session(it->second.backend_handle)`, which calls `llama_free(sess->raw_ctx)`.
6. Concurrently, the scheduler worker finishes `evaluate_prompt` and calls `backend_->sample_next_token()` on the already freed `llama_context` pointer!
7. **Impact:** Heap memory corruption, access violation (`SIGSEGV`), or undefined behavior.

#### Remediation:
- Never destroy backend resources if in-flight requests remain active.
- If cooperative cancellation does not drain within the timeout, return `StatusCode::TIMEOUT` or `BUSY` rather than proceeding to free active memory.
- Alternatively, wait indefinitely or with a significantly longer grace period, and check atomic session reference counts before releasing `backend_handle`.

---

### Finding SEC-04 (MEDIUM): Missing `DetachCurrentThread` on Worker Thread Exit

- **Category:** Android ART Compatibility / Resource Leak
- **Severity:** **MEDIUM**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`android/jni/mobilellm_jni.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/android/jni/mobilellm_jni.cpp#L241-L248) and [`android/jni/mobilellm_jni.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/android/jni/mobilellm_jni.cpp#L367-L369)

#### Description & Code Analysis:
In `mobilellm_jni.cpp`:
1. `token_fn` is invoked on the native scheduler worker thread.
2. `g_jvm->GetEnv` returns `JNI_EDETACHED`, so `token_fn` calls `g_jvm->AttachCurrentThread(&thread_env, nullptr)`.
3. `token_fn` sets local variable `bool attached = true`, but exits without detaching so that subsequent tokens do not incur attach/detach overhead.
4. When inference completes, `complete_fn` is called on the **same scheduler worker thread**.
5. In `complete_fn`:
   ```cpp
   jint res = g_jvm->GetEnv(reinterpret_cast<void**>(&thread_env), JNI_VERSION_1_6);
   if (res == JNI_EDETACHED) {
       // NOT REACHED because thread is already attached!
       attached = true;
   }
   // ...
   if (attached) { // False!
       g_jvm->DetachCurrentThread(); // NEVER CALLED!
   }
   ```
6. When `EngineImpl::shutdown()` stops the scheduler, the native worker thread exits while still attached to the JVM.

#### Impact:
On Android ART (Android Runtime), native threads exiting without detaching trigger:
`"native thread exited without detaching from Java VM"`
This causes resource leaks in the ART thread table and can trigger `art::Runtime::Abort()` on debug builds.

#### Remediation:
Use a thread-local RAII object on the scheduler worker thread with a destructor that invokes `DetachCurrentThread` upon thread exit, or explicitly detach during scheduler worker termination.

---

### Finding SEC-05 (MEDIUM): Python ctypes Callback Premature Garbage Collection Risk

- **Category:** FFI Pointer Safety / Use-After-Free
- **Severity:** **MEDIUM**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`python/mobilellm/bindings/native_client.py`](file:///Users/Apple/Developer/ai-apps/MobileLLM/python/mobilellm/bindings/native_client.py#L243-L256)

#### Description & Code Analysis:
In `NativeSession.generate()`:
```python
c_token_cb = _TOKEN_CB_TYPE(_on_token)
c_complete_cb = _COMPLETE_CB_TYPE(_on_complete)
gen_handle = c_uint64(0)

status = self._lib.mllm_generate(
    ...,
    c_token_cb,
    c_complete_cb,
    None,
    ctypes.byref(gen_handle),
)
```
`c_token_cb` and `c_complete_cb` are local variables of the generator function. If the consumer iterates partially and drops the generator:
```python
for chunk in session.generate(prompt):
    break  # Generator is garbage collected!
```
The local ctypes callback wrappers can be deallocated by the Python garbage collector while the native C++ scheduler worker thread still holds their native function pointers. When the native worker invokes `complete_cb`, it dereferences a dangling pointer in the Python runtime, causing an immediate `SIGSEGV`.

#### Remediation:
Store active callback references on `self._active_callbacks` or an instance-level dictionary until `done_event.is_set()` in `finally:`.

---

### Finding SEC-06 (MEDIUM): Unbounded Array Loop in GGUF Metadata Parser

- **Category:** Input Validation / Denial of Service
- **Severity:** **MEDIUM**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`runtime/src/model_loader/model_loader.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/src/model_loader/model_loader.cpp#L144-L153)

#### Description & Code Analysis:
In `skip_value()`:
```cpp
case GGUFType::ARRAY: {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!file.read(reinterpret_cast<char*>(&elem_type), sizeof(elem_type))) return false;
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;
    for (uint64_t i = 0; i < count; ++i) {
        if (!skip_value(file, elem_type)) return false;
    }
    return file.good();
}
```

#### Vulnerability & Impact:
If a malformed or hostile GGUF file specifies `count = 0xFFFFFFFFFFFFFFFFULL` with a trivial type or nested arrays:
1. The loop executes $2^{64}$ iterations.
2. In the case of nested arrays (`elem_type == GGUFType::ARRAY`), recursive calls exhaust the call stack, causing a Stack Overflow crash.
3. This creates a Denial of Service vulnerability whenever an application inspects an untrusted model file.

#### Remediation:
1. Enforce an upper bound on metadata array elements (e.g. `count > 1000000 -> return false`).
2. Limit maximum recursion depth in `skip_value` (e.g. `depth > 8 -> return false`).

---

### Finding SEC-07 (LOW): Silent Output Truncation in `mllm_model_inspect`

- **Category:** C API Robustness
- **Severity:** **LOW**
- **Verification Status:** **VERIFIED**
- **Affected File:** [`runtime/src/engine/c_api.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/src/engine/c_api.cpp#L170-L173)

#### Description:
```cpp
std::string json = ss.str();
strncpy(json_buffer, json.c_str(), max_len - 1);
json_buffer[max_len - 1] = '\0';
return c_error(0, "");
```
If `json_buffer` size `max_len` is smaller than `json.size() + 1`, `strncpy` silently truncates the JSON string, producing invalid JSON syntax while returning `Status::OK` (`code = 0`). The caller has no mechanism to detect that truncation occurred.

#### Remediation:
Check `if (json.size() + 1 > max_len)` and return `StatusCode::INVALID_ARGUMENT` or `CONTEXT_EXCEEDED` with `"Buffer too small for model metadata"`.

---

### Finding SEC-08 (INFORMATIONAL): Synthetic Hardware Capabilities

- **Category:** Telemetry / Correctness
- **Severity:** **INFORMATIONAL**
- **Verification Status:** **VERIFIED**
- **Affected Files:** [`runtime/src/engine/engine_impl.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/src/engine/engine_impl.cpp#L527-L536) and [`runtime/src/engine/c_api.cpp`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/src/engine/c_api.cpp#L303)

#### Description:
`EngineImpl::query_hardware()` returns hardcoded `neon_supported = true` (even on x86_64) and $0$ for available RAM. `mllm_detect_hardware()` returns a hardcoded mock JSON string. This will be remediated in coordination with Agent 5 under [ADR-0009](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/0009-hardware-capability-detection-and-modular-acceleration-architecture.md).

---

## 4. Privacy & Offline Verification Audit

| Privacy Requirement | Audit Method | Result | Notes |
| :--- | :--- | :---: | :--- |
| **Zero Network Permissions** | `AndroidManifest.xml` inspection | **VERIFIED PASS** | No `android.permission.INTERNET` declared in SDK or App manifest. |
| **No Socket Calls in Native Code** | Symbol grep across `runtime/`, `scheduler/`, `backends/` | **VERIFIED PASS** | Zero socket, connect, curl, or HTTP symbols found in core library. |
| **No Prompt / Token Logging** | Review of all `LOGI`, `LOGW`, `LOGE` macros | **VERIFIED PASS** | Prompts and token text are never logged to `logcat` or stdout. Only handles and vocab sizes are emitted. |
| **Loopback-Only HTTP Binding** | Inspection of `python/mobilellm/server/app.py` | **VERIFIED PASS** | Defaults strictly to `127.0.0.1` loopback with optional Bearer token authentication. |

---

## 5. Summary & Remediation Gate

All 8 findings have been verified on the current repository codebase. 

**Recommended Action for Stage 2:**  
Obtain Lead Architect (Agent 0) approval to remediate Findings **SEC-01 (Critical)**, **SEC-02 (High)**, and **SEC-03 (High)** first, accompanying each fix with regression unit tests.
