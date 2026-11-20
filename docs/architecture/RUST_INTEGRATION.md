# MobileLLM Hybrid C++20 + Rust Architecture Specification

**Status:** Approved / Experimental Integration  
**Lead Systems Architect:** Agent 11  
**Last Updated:** November 20, 2026  

---

## 1. System Overview

MobileLLM Engine integrates an optional Rust subsystem designed to operate alongside the core C++20 inference engine. This document specifies the crate topology, C ABI contracts, Android ARM64 compilation pipeline, and safety boundaries.

---

## 2. Workspace Layout

The Rust workspace resides in `rust/` and consists of four focused crates:

```text
rust/
├── Cargo.toml                      # Workspace manifest
├── rust-toolchain.toml             # Pinned toolchain (stable, android targets)
├── CMakeLists.txt                  # CMake integration logic
└── crates/
    ├── mobilellm-core/             # Hardware models, errors, and metrics
    │   ├── Cargo.toml
    │   └── src/lib.rs
    ├── mobilellm-gpu/              # Vulkan capability introspection & micro-compute (ash)
    │   ├── Cargo.toml
    │   └── src/lib.rs
    ├── mobilellm-diagnostics/      # End-to-end diagnostic runner
    │   ├── Cargo.toml
    │   └── src/lib.rs
    └── mobilellm-ffi/              # C-ABI export layer (cdylib, staticlib)
        ├── Cargo.toml
        ├── include/
        │   └── mobilellm_rust_ffi.h
        └── src/lib.rs
```

### 2.1 Crate Responsibilities

| Crate | Responsibility | Dependencies |
| :--- | :--- | :--- |
| `mobilellm-core` | Defines `ErrorCode`, `Error`, `GpuDeviceInfo`, `ExecutionCapabilities`, and `ComputeMetrics`. | None (zero external dependencies). |
| `mobilellm-gpu` | Interacts with Vulkan using `ash`. Enumerates physical devices, parses Qualcomm Adreno properties, and runs isolated deterministic compute checks. | `mobilellm-core`, `ash` |
| `mobilellm-diagnostics` | Orchestrates hardware evaluation runs and calculates latency and precision deltas. | `mobilellm-core`, `mobilellm-gpu` |
| `mobilellm-ffi` | Exposes stable C-ABI symbols, manages handle lifecycles, and enforces panic boundaries. | `mobilellm-core`, `mobilellm-gpu`, `mobilellm-diagnostics` |

---

## 3. C-ABI Interface Contract

All symbols exported by `mobilellm-ffi` use the `mllm_rust_` prefix and conform to standard C calling conventions (`extern "C"`).

### 3.1 Data Types

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
```

### 3.2 Invariants & Safety Guarantees

1. **Panic Encapsulation:**
   No Rust panic may cross the FFI boundary. All public functions invoke `std::panic::catch_unwind(AssertUnwindSafe(...))`. Caught panics return status code `99` (`ErrorCode::InternalPanic`).
2. **Pointer Validation:**
   Passing null pointers to any pointer-receiving argument returns status code `1` (`ErrorCode::InvalidArgument`) with an explanatory diagnostic message.
3. **No Unbounded Memory Transfers:**
   Large tensors are never transferred across the C++ and Rust boundary. Rust receives only lightweight pointers and scalar parameters.
4. **Thread Safety:**
   `GpuContext` implements `Send` and `Sync`. Context handles can be shared or invoked across threads.

---

## 4. CMake & Android Cross-Compilation

### 4.1 CMake Toggle
```cmake
option(MOBILELLM_ENABLE_RUST "Enable optional Rust components" OFF)
```

### 4.2 Cross-Compilation Flags
When compiling for Android (`ANDROID=1`), `rust/CMakeLists.txt` automatically maps:
- `arm64-v8a` $\to$ `aarch64-linux-android`
- `x86_64` $\to$ `x86_64-linux-android`
- Sets `CARGO_TARGET_<TARGET>_LINKER` to the NDK Clang compiler.
- Sets `RUSTFLAGS="-Clink-arg=-Wl,-z,max-page-size=16384"` to guarantee Android 15 16 KB page-size compliance.

---

## 5. Verification & Testing

Tests are stratified into two suites:
1. **Cargo Test Suite:** Unit and property tests inside each crate (`cargo test --workspace`).
2. **Native C++ FFI Suite:** `runtime/tests/test_rust_ffi.cpp` executing through CMake `ctest`.
