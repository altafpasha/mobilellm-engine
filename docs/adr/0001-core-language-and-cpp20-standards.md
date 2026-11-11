# ADR-0001: Adoption of C++20 for Native Engine & Core Runtime

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 3 (Android), Agent 6 (QA)

---

## Context

MobileLLM Engine requires a native systems programming language for its core runtime (`runtime/`), memory scheduling subsystem (`scheduler/`), and hardware backends (`backends/`). The runtime must:
1. Deliver zero-overhead memory and execution performance on Android ARM64 hardware.
2. Compile reproducibly via Android NDK (Clang toolchain) and host compilers (GCC, Apple Clang).
3. Provide modern language features (`std::span`, concepts, coroutines, `std::variant`, `std::string_view`) that improve code safety and prevent memory leaks.
4. Seamlessly integrate with native tensor backends (GGML, Vulkan, NDK APIs) without foreign-function interface overhead.

## Decision

We adopt **C++20** (`-std=c++20`) as the core native programming language across all native modules (`runtime/`, `scheduler/`, `backends/`, `android/jni/`).

### Technical Standards:
- **Compiler Support:** Target Clang 17+ (shipped in Android NDK r26c+) and GCC 12+.
- **Resource Management:** Strict RAII (Resource Acquisition Is Initialization) for all resources. Raw `new`/`delete` is forbidden in favor of `std::make_unique` and standard smart pointers.
- **Safety Flags:** All native builds must enable `-Wall -Wextra -Wpedantic -Werror` in CI.
- **Symbol Visibility:** Hidden by default (`-fvisibility=hidden -fvisibility-inlines-hidden`). Only explicitly annotated C-ABI functions (`MOBILELLM_EXPORT`) are exported.
- **Error Propagation:** Zero unhandled exceptions. Internal errors are mapped to `mobilellm::Status` and `mobilellm::Result<T>`.

## Consequences

### Positive:
- High performance and fine-grained control over cache lines, thread affinity, and memory layouts.
- Native access to Android NDK APIs (Vulkan, `/proc/meminfo`, ATrace).
- Availability of modern memory abstractions (`std::span`, `std::unique_ptr`) reducing buffer overrun hazards.
- Fully supported by Android NDK r26c+ and modern desktop toolchains.

### Negative / Trade-offs:
- Manual memory management vigilance required compared to memory-safe managed runtimes; mitigated through ASan/UBSan testing and strict RAII.
- Build times for C++20 templates and libraries; mitigated through modular CMake targets and ccache.

## Alternatives Considered

- **Rust:** Evaluated for memory safety. Rejected for initial bootstrap because major tensor inference libraries (`llama.cpp`, Vulkan compute kernels, Android NDK sample ecosystems) have native C/C++ representations. Adding a Rust-to-C-to-JNI three-tier bridge increases build toolchain complexity for Android NDK distribution. May be revisited for isolated modules in future phases.
- **C++17:** Considered for wider legacy compiler compatibility. Rejected because C++20 provides critical features (`std::span`, concepts, improved `constexpr`, coroutine primitives) that drastically clean up tensor and buffer management. NDK r26c supports C++20 out of the box.
