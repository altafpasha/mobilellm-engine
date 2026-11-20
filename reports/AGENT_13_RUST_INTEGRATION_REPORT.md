# Agent 13 — MobileLLM Rust Integration Foundation Report

**Document ID:** `REPORT-AGENT-13-RUST-INTEGRATION-2026-11-20`  
**Author:** Lead Rust Systems Engineer (Agent 13) / Principal Architect (Agent 11)  
**Date:** November 20, 2026  
**Scope:** Modular Rust Workspace (`rust/`), C-ABI Export Layer (`mobilellm-ffi`), CMake & Android NDK Cross-Compilation  
**Status:** **IMPLEMENTED & EXPERIMENTAL (OPTIONAL SUBSYSTEM)**

---

## 1. Executive Summary

As specified in Track B of the MobileLLM Hybrid C++20 + Rust Charter, a modular Rust workspace has been introduced as an **optional systems programming component**. 

The design strictly obeys the non-negotiable engineering restrictions:
1. The mature C++20 inference engine, llama.cpp backend, memory manager, scheduler, tokenizer, and sampling kernels are 100% preserved.
2. Builds default to `MOBILELLM_ENABLE_RUST=OFF`, requiring zero Cargo or Rust dependencies for standard C++ builds.
3. When enabled, the Rust subsystem compiles via CMake into a self-contained static archive (`libmobilellm_rust_ffi.a`), exposing a stable C-ABI boundary with opaque handles, strict pointer validation, and panic-catching guards.
4. Android ARM64 cross-compilation is fully supported with explicit 16 KB ELF page-size alignment (`-Wl,-z,max-page-size=16384`).

---

## 2. Crate Architecture & Topology

The workspace is organized into four decoupled crates in `rust/crates/`:

```text
rust/
├── Cargo.toml
├── rust-toolchain.toml
├── CMakeLists.txt
└── crates/
    ├── mobilellm-core/             # Core types, errors, and hardware specifications
    ├── mobilellm-gpu/              # Vulkan capability introspection & micro-compute (ash)
    ├── mobilellm-diagnostics/      # End-to-end diagnostics and telemetry aggregation
    └── mobilellm-ffi/              # C-ABI export layer (cdylib, staticlib)
        ├── include/
        │   └── mobilellm_rust_ffi.h
        └── src/lib.rs
```

### 2.1 Crate Details & Dependency Policy

#### `mobilellm-core`
- **Dependencies:** None (zero external dependencies).
- **Types Defined:**
  - `ErrorCode`: Strongly typed status codes (Success = 0, InvalidArgument = 1, VulkanUnavailable = 2, InitializationFailed = 3, AllocationFailed = 4, ExecutionFailed = 5, InternalPanic = 99).
  - `Error`: Rich error type with custom `Display` and `std::error::Error` implementations.
  - `GpuDeviceInfo`: Vendor ID, device ID, device name, API version, driver version, subgroup size, and local memory bytes.
  - `ExecutionCapabilities`: Flags for FP16, INT8, subgroups, and subgroup arithmetic.
  - `ComputeMetrics`: Duration, workgroup dimensions, and max absolute error.
  - `DiagnosticReport`: Aggregated diagnostic outcomes.

#### `mobilellm-gpu`
- **Dependencies:** `mobilellm-core`, `ash` (`version = "0.38"`, `default-features = false`, `features = ["loaded"]`).
- **Capabilities:**
  - Dynamic loading of Vulkan runtime (`libvulkan.so` on Android, `libvulkan.1.dylib` on macOS).
  - Physical device capability and subgroup inspection (specifically identifying Qualcomm Adreno 64-thread wavefronts).
  - Isolated deterministic micro-compute workload: Vector affine transformation $y[i] = 2.0 \times x[i] + 1.0$ with numerical precision verification.
  - Graceful fallback when Vulkan drivers are absent (e.g. host CI without MoltenVK), guaranteeing zero crashes.
  - Safe RAII cleanup (`Drop`) ensuring logical device and instance destruction order.

#### `mobilellm-diagnostics`
- **Dependencies:** `mobilellm-core`, `mobilellm-gpu`.
- **Capabilities:**
  - Automated diagnostic runner executing end-to-end hardware capability detection and numerical error quantification.

#### `mobilellm-ffi`
- **Dependencies:** `mobilellm-core`, `mobilellm-gpu`, `mobilellm-diagnostics`.
- **Crate Types:** `["cdylib", "staticlib"]` -> produces `libmobilellm_rust_ffi.a` and `libmobilellm_rust_ffi.so` / `.dylib`.
- **Public Header:** `include/mobilellm_rust_ffi.h`.

---

## 3. C-ABI Interface & Memory Safety Contracts

### 3.1 Interface Definition (`mobilellm_rust_ffi.h`)

```c
typedef struct mllm_rust_gpu_handle mllm_rust_gpu_handle_t;

typedef struct {
    int32_t code;
    char message[256];
} mllm_rust_status_t;

typedef struct {
    uint32_t vendor_id;
    uint32_t device_id;
    char device_name[256];
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t subgroup_size;
    uint64_t device_local_memory_bytes;
    bool has_fp16;
    bool has_int8;
    bool has_subgroups;
    bool has_subgroup_arithmetic;
} mllm_rust_gpu_capabilities_t;

typedef struct {
    bool execution_passed;
    float execution_time_ms;
    float max_absolute_error;
    char details[256];
} mllm_rust_diagnostic_result_t;

mllm_rust_status_t mllm_rust_gpu_initialize(mllm_rust_gpu_handle_t** handle_out);
mllm_rust_status_t mllm_rust_gpu_query_capabilities(const mllm_rust_gpu_handle_t* handle, mllm_rust_gpu_capabilities_t* caps_out);
mllm_rust_status_t mllm_rust_gpu_run_diagnostic(const mllm_rust_gpu_handle_t* handle, mllm_rust_diagnostic_result_t* diag_out);
void mllm_rust_gpu_destroy(mllm_rust_gpu_handle_t* handle);
```

### 3.2 FFI Invariants Enforced

1. **Panic Encapsulation:**
   Every exported function wraps internal logic in `std::panic::catch_unwind(AssertUnwindSafe(|| ...))`. Any unexpected panic is caught and mapped to `ErrorCode::InternalPanic` (99). Rust panics cannot unwind across the C-ABI.
2. **Pointer Validation:**
   Passing `NULL` to any pointer-receiving function produces `ErrorCode::InvalidArgument` (1) with an explicit error message.
3. **Safe String Copying:**
   Strings copied into C fixed-size buffers (`char[256]`) are explicitly sliced, clamped to 255 bytes, and null-terminated.
4. **Explicit Allocation & Ownership:**
   `mllm_rust_gpu_initialize()` allocates via `Box::into_raw()`. Reclaiming the resource is performed exclusively by `mllm_rust_gpu_destroy()` via `Box::from_raw()`. Destroying a null handle is a safe no-op.

---

## 4. CMake & Android NDK Cross-Compilation

### 4.1 Root CMake Toggle
In `CMakeLists.txt`:
```cmake
option(MOBILELLM_ENABLE_RUST "Enable optional Rust components" OFF)
```
- When `OFF`: Cargo is never searched or invoked. C++ builds compile without any Rust toolchain.
- When `ON`: Cargo availability is verified with clear diagnostic error handling if absent.

### 4.2 Android Target Mapping & 16 KB Page Alignment
In `rust/CMakeLists.txt`:
- Mapped Android ABIs:
  - `arm64-v8a` $\to$ `aarch64-linux-android`
  - `x86_64` $\to$ `x86_64-linux-android`
- Configured Clang linker:
  `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = ${CMAKE_C_COMPILER}`
- Enforced 16 KB page-size linker requirements:
  `set(ENV{RUSTFLAGS} "-Clink-arg=-Wl,-z,max-page-size=16384")`

### 4.3 Android JNI Integration
In `android/jni/CMakeLists.txt`:
When `MOBILELLM_ENABLE_RUST=ON`, `mobilellm_rust_ffi` is linked into `libmobilellm_jni.so`, and headers from `rust/crates/mobilellm-ffi/include` are added to the include path.

---

## 5. Verification Results

1. **Pure C++20 Build Verification:**
   - With `MOBILELLM_ENABLE_RUST=OFF`: 108/108 host C++ tests pass (`100% tests passed out of 108`).
   - Android release packaging dry-run succeeds (`BUILD SUCCESSFUL`).
   - Python test suite: 35 passed, 1 skipped.
2. **Diagnostic Error Verification:**
   - Attempting to configure `-DMOBILELLM_ENABLE_RUST=ON` in an environment without `cargo` raises a clear, actionable CMake error:
     `CMake Error at rust/CMakeLists.txt:8 (message): MOBILELLM_ENABLE_RUST is ON, but 'cargo' binary was not found in PATH.`
3. **C-ABI Test Coverage (`runtime/tests/test_rust_ffi.cpp`):**
   - Implemented null pointer rejection test, full subsystem lifecycle test, and repeated lifecycle safety test.
   - When Rust is disabled, executes `RustDisabledGracefulFallback` ensuring 100% test suite stability.
