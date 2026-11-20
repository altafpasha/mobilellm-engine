# ADR-0010: Hybrid C++20 + Optional Rust Integration Architecture

- **Status:** Proposed / Experimental Subsystem
- **Deciders:** Lead Systems Architect (Agent 11), C++ Core Engineer, Rust Systems Engineer, Android Platform Engineer
- **Date:** November 20, 2026
- **Technical Context:** MobileLLM Engine Inference Runtime

---

## 1. Context and Problem Statement

MobileLLM Engine is built on a high-performance C++20 foundation with llama.cpp/GGML powering low-level tensor computations and ARM64 NEON acceleration. While C++20 delivers maximal raw execution speed and tight hardware coupling, certain systems tasks—such as hardware capability parsing, isolated GPU compute diagnostics, cross-platform hardware introspection, and memory-safe metadata processing—can benefit from Rust's compile-time memory safety, strict concurrency invariants, and modern ergonomics.

However, a total rewrite in Rust is strictly anti-goal:
1. Rewriting the inference scheduler, memory manager, tokenizer, and GGUF tensor kernels would discard mature, production-validated optimizations (such as PERF-01 decomposed TTFT, PERF-03 persistent logits, persistent threadpools, and FUSE path mitigations).
2. Forcing a mandatory Cargo build dependency would break lightweight C++ and Android NDK build pipelines.
3. Passing full tensor buffers across language boundaries introduces translation overhead and cache invalidation.

We require a controlled architectural evolution: introducing Rust as an **optional, modular subsystem** without breaking the existing C++20 runtime or public C/Kotlin/Python APIs.

---

## 2. Decision Drivers

- **Zero Disruption to C++ Core:** The C++20 inference engine remains the authoritative execution runtime.
- **Opt-in Cargo Build Dependency:** Builds default to `MOBILELLM_ENABLE_RUST=OFF`, requiring no Rust toolchain for standard C++ development.
- **Strict FFI Invariants:** Clear C ABI boundary with opaque handles, explicit allocation and destruction, `#[repr(C)]` data structures, and zero unwinding across language boundaries.
- **Android ARM64 & 16 KB Page Alignment:** Full compatibility with Android NDK r27b, `aarch64-linux-android`, and Android 15 `-Wl,-z,max-page-size=16384`.
- **Resource Ownership Isolation:** No shared GPU queues or dual-ownership races between Rust and C++.

---

## 3. Considered Options

1. **Full Rewrite in Rust (burn-rs / candle / wgpu):**
   *Rejected.* Massive regression risk, loss of GGUF quantization kernel parity, and broken Android Kotlin/JNI bindings.
2. **Cxx / autocxx / cbindgen Bindings:**
   *Rejected for initial phase.* Generates complex intermediate C++ wrappers and template glue, increasing build fragility across Android NDK versions.
3. **Internal C ABI Boundary with Opaque Handles (Chosen):**
   *Approved.* Uses stable, predictable `extern "C"` functions with opaque pointers (`mllm_rust_gpu_handle_t`), explicit status codes (`mllm_rust_status_t`), and `catch_unwind` guards. Fully compatible with CMake, NDK Clang, and Python ctypes.

---

## 4. Architectural Decision

### 4.1 Topology & Role Distribution

```text
               +-------------------------------------------+
               |        Public MobileLLM C API             |
               |             (c_api.h)                     |
               +-------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    +----------------------+                   +-----------------------+
    | C++20 Runtime Engine |                   | Optional Rust Bridge  |
    | (EngineImpl)         |                   | (mobilellm_rust_ffi)  |
    +----------------------+                   +-----------------------+
               |                                           |
    +----------+-----------+                   +-----------+-----------+
    |                      |                   |                       |
    v                      v                   v                       v
+--------+            +----------+      +-------------+        +---------------+
| CPU    |            | Vulkan   |      | Hardware    |        | Isolated      |
| NEON   |            | GGML     |      | Diagnostics |        | Micro-Compute |
+--------+            +----------+      +-------------+        +---------------+
```

- **C++20 Domain:** Model loading, memory budgeting, KV-cache paged allocation, continuous batch scheduling, streaming token generation, and primary Vulkan GPU inference.
- **Rust Domain (Experimental):** Hardware capability parsing, structured error models, driver introspection, and isolated micro-compute diagnostics using `ash`.

### 4.2 FFI Ownership & Safety Contract

1. **No Rust Panics across FFI:**
   Every exported function wraps its body in `std::panic::catch_unwind(AssertUnwindSafe(...))`. Uncaught panics are converted to `ErrorCode::InternalPanic` (99).
2. **Explicit Memory Allocation and Destruction:**
   Rust heap allocations remain owned by Rust. Handles are returned as raw pointers from `Box::into_raw()` and reclaimed via `mllm_rust_gpu_destroy()` using `Box::from_raw()`.
3. **No Unbounded Buffers:**
   Strings crossing the boundary use fixed-size null-terminated `char[256]` buffers.
4. **Validation:**
   Null pointers, length mismatches, and invalid handles are intercepted immediately, returning explicit error codes without memory faults.

### 4.3 Build System Integration

In root `CMakeLists.txt`:
```cmake
option(MOBILELLM_ENABLE_RUST "Enable optional Rust components" OFF)
```
- When `OFF`: No Cargo binary or Rust toolchain is checked. Build remains 100% pure C++20.
- When `ON`: CMake verifies `cargo` in PATH, resolves the target triple (`aarch64-linux-android` for Android, host for desktop), enforces `-Clink-arg=-Wl,-z,max-page-size=16384`, and compiles `libmobilellm_rust_ffi.a`.

---

## 5. Consequences

### Positive
- Introduces safe, type-rich hardware diagnostics without risking C++ inference stability.
- Zero impact on build times or developer requirements for C++ engineers.
- Clean path for future incremental migrations of auxiliary tooling.

### Negative / Trade-offs
- Two build toolchains (CMake and Cargo) must be coordinated when Rust is enabled.
- Requires maintaining header synchronization between `mobilellm_rust_ffi.h` and `crates/mobilellm-ffi/src/lib.rs`.

---

## 6. Rollback Strategy

Because the Rust subsystem is behind the `MOBILELLM_ENABLE_RUST` compile option and shares zero runtime state with the inference scheduler:
- Setting `-DMOBILELLM_ENABLE_RUST=OFF` immediately reverts the engine to pure C++20.
- Removing `rust/` restores the exact pre-integration workspace without editing core inference routines.
