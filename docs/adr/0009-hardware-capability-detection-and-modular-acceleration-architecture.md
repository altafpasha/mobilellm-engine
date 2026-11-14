# ADR-0009: Hardware Capability Detection & Modular Acceleration Architecture

- **Status:** Proposed / Architectural Proposal (Stage 1 Deliverable)
- **Decider:** Agent 5 (Principal Hardware Acceleration Engineer) & Agent 0 (Lead Architect)
- **Date:** 2026-11-14
- **Consulted:** Agent 1 (Runtime), Agent 2 (Scheduler), Agent 3 (Android), Agent 6 (QA & Security)

---

## Context

MobileLLM Engine has achieved stable build linkage and single/multi-session request execution delegation across C++, Android JNI, and Python environments ([ADR-0008](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/ADR-0008-scheduler-runtime-integration.md)). However, an engineering audit of the current hardware subsystem reveals several architectural gaps:

1. **Hardcoded Capability Reporting:** `EngineImpl::query_hardware()` and `mllm_detect_hardware()` in `runtime/src/engine/c_api.cpp` return static mock strings (`"Host Machine"`, `"ARM64"`, `neon: true`, `vulkan: false`). On x86_64 host machines or Android emulators, `neon: true` is reported falsely. System memory (`total_ram_bytes`, `available_ram_bytes`) is not populated in `HardwareProfile`.
2. **Backend Selection Incompleteness:** `BackendRegistry::create(BackendType::AUTO)` naively checks if `factories_.count(BackendType::VULKAN)` is non-zero. It does not verify whether the physical GPU supports mandatory compute extensions (`VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8`), nor does it handle device creation failures with automatic fallback to CPU.
3. **Android Cross-Compilation Blocker for Vulkan:** Stock `llama.cpp` Vulkan configuration (`third_party/llama.cpp/ggml/src/ggml-vulkan/CMakeLists.txt`) executes `find_package(Vulkan COMPONENTS glslc REQUIRED)` to generate SPIR-V headers via a host binary `vulkan-shaders-gen`. During Android NDK cross-compilation from macOS or Linux, host `glslc` is typically unavailable, causing the CMake build to fail.
4. **ARM64 Compilation Flags:** Android builds currently compile with generic `-std=c++20 -O3 -DNDEBUG` without target architecture tuning. While baseline ARMv8.0 NEON is enabled, modern acceleration features (`+dotprod` in ARMv8.2-A, `+i8mm` in ARMv8.6-A) are not systematically targeted or dynamically dispatched.
5. **Thermal Throttling Blindness:** Long autoregressive decode runs on physical mobile SoCs trigger thermal throttling within 60–120 seconds, causing throughput collapse. The runtime lacks an integrated thermal advisory policy to dynamically modulate active CPU threads or offload layers.

We require a robust, evidence-based hardware acceleration architecture that resolves these gaps while preserving strict C, C++, and Kotlin ABI stability.

---

## Decision

We adopt a **Modular Hardware Abstraction Layer (HAL) with Evidence-Based Dynamic Capability Detection and Safe CPU Fallback**.

### Core Architectural Decisions:

1. **Evidence-Based Dynamic Hardware Capability Detection:**
   - Implement `hardware::CapabilityDetector` to query real platform APIs dynamically at runtime.
   - **Linux/Android:** Query `getauxval(AT_HWCAP)` and `getauxval(AT_HWCAP2)` for ARM NEON, FP16, DotProd (`HWCAP_ASIMDDP`), I8MM (`HWCAP2_I8MM`), and SVE. Parse `/proc/cpuinfo`, `/proc/meminfo`, and `/sys/devices/system/cpu/` to detect core topologies (Prime, Performance, Efficiency).
   - **Darwin/macOS:** Query `sysctlbyname` for ARM SIMD features (`hw.optional.AdvSIMD`, `FEAT_DotProd`, `FEAT_I8MM`).
   - **Vulkan Probing:** Dynamically load `libvulkan.so` via `dlopen`. Query `vkEnumeratePhysicalDevices`, `VkPhysicalDeviceProperties`, and verify mandatory compute extensions (`VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8`). Never link statically to the Vulkan loader.
   - Every detected capability is tagged with an explicit confidence level (`CONFIRMED`, `UNSUPPORTED_EXTENSION`, `DRIVER_NOT_FOUND`, `MOCKED_TEST`).

2. **Five-Stage Backend Lifecycle State Machine:**
   - Transition all compute backends through 5 distinct states:
     $$\text{COMPILED\_IN} \longrightarrow \text{HARDWARE\_SUPPORTED} \longrightarrow \text{INITIALIZABLE} \longrightarrow \text{VALIDATED} \longrightarrow \text{SELECTED}$$
   - No backend is selected solely because its code is compiled or its device exists.
   - A dry-run validation step (e.g., evaluating a single dummy token or trivial compute shader) must succeed before allocating full model weights.

3. **Strict Golden CPU Fallback Invariant:**
   - The ARM64 CPU backend utilizing ARM NEON SIMD intrinsics is the foundational reference standard.
   - If Vulkan initialization, memory allocation, or pipeline compilation fails, the runtime must:
     1. Safely release all partially allocated Vulkan handles with zero memory leaks.
     2. Log a diagnostic warning detailing the exact Vulkan error code.
     3. Automatically and transparently fall back to `CPU_NEON` using the exact same model weights and session configuration.

4. **ARM64 Compilation Tiers & Function-Level Multi-Arch Dispatch:**
   - Retain baseline ARMv8.0-A compatibility for the main shared library to avoid `SIGILL` on older mobile devices.
   - Leverage `llama.cpp`'s optimized kernels in `ggml-cpu-aarch64.c` (`SDOT` for `FEAT_DotProd`, `SMMLA` for `FEAT_I8MM`).
   - Use function-level target attributes (`__attribute__((target("+dotprod")))`, `__attribute__((target("+i8mm")))`) with runtime `getauxval` function-pointer dispatch to ensure optimal vector throughput across all ARM generations.

5. **Precompiled Offline SPIR-V Headers for Android Vulkan Buildability:**
   - Resolve the Android NDK `glslc` cross-compilation blocker by precompiling GLSL compute shaders offline into static C++ byte-array headers.
   - When building for Android, Vulkan compilation consumes these precompiled SPIR-V headers directly, eliminating any host `glslc` dependency during Android Gradle / CMake invocations.

6. **Isolated Dynamic Loading for Vendor NPUs:**
   - Qualcomm QNN (`backends/qnn/`), MediaTek NeuroPilot (`backends/neuropilot/`), and experimental NPU modules must never introduce link-time dependencies into `libmobilellm.so`.
   - All vendor SDK interactions must occur via dynamic library resolution (`dlopen`/`dlsym`).
   - Due to Google's formal deprecation of Android NNAPI in Android 15 (API 35) and its total lack of native LLM operators (RoPE, RMSNorm, dynamic attention), NNAPI is disqualified as a primary acceleration backend.

7. **Thermal Advisory Integration:**
   - Connect Android NDK thermal callbacks (`<android/thermal.h>`, API 30+) to provide real-time thermal status to `MemoryBudgetManager` and `ContinuousBatchScheduler`.
   - Under `THERMAL_STATUS_MODERATE` or `SEVERE`, the engine dynamically drops thread counts (e.g. 4 $\to$ 3 $\to$ 2) and throttles prompt prefill chunk sizes to prevent thermal runaway.

---

## Consequences

### Positive:
- **Zero Crashes on Unsupported GPUs:** Devices with buggy or incomplete Vulkan drivers gracefully fallback to CPU without application termination.
- **Accurate Diagnostic Telemetry:** Applications and automated benchmarks receive real hardware capabilities, core topologies, and memory budgets rather than hardcoded mock strings.
- **Maximized ARM SIMD Throughput:** Modern Snapdragon 8 Gen 1–3 and Dimensity 9000+ chipsets unlock up to $2\times$ integer dot product throughput via `FEAT_DotProd` and `FEAT_I8MM` kernels.
- **Clean Android CI Builds:** Precompiled SPIR-V shaders enable smooth Android NDK compilation without complex host toolchain prerequisites.
- **Licensing & Supply-Chain Cleanliness:** Core runtime remains pure Apache 2.0 / MIT with zero proprietary vendor NPU blobs.

### Negative / Trade-offs:
- **Precompiled Shader Maintenance:** Updating Vulkan compute shaders requires regenerating the static SPIR-V header files before committing.
- **CPU Fallback Latency:** Falling back from Vulkan to CPU results in lower tokens/second; the application UI must indicate that fallback occurred.

---

## Alternatives Considered & Rejected Options

1. **Option A: Static Linkage to Vulkan Loader (`libvulkan.so`)**  
   - *Rejection Reason:* Causes immediate `dlopen` failures on Android emulators or customized automotive/embedded Android builds where `libvulkan.so` is absent from `/system/lib64/`. Dynamic `dlopen` probing is strictly required.

2. **Option B: Write Custom Custom NEON SIMD Kernels from Scratch**  
   - *Rejection Reason:* `llama.cpp`'s GGML submodule already contains mature, hand-optimized assembly kernels for ARMv8.0 NEON, DotProd, and I8MM across all GGUF quantization formats. Rewriting duplicate kernels would introduce maintenance overhead and risk numerical divergence.

3. **Option C: Build Full Acceleration Pipeline on Android NNAPI**  
   - *Rejection Reason:* Google deprecated NNAPI in Android 15. Furthermore, NNAPI lacks support for causal attention masks, RMSNorm, and Rotary Position Embeddings (RoPE), requiring massive CPU-GPU tensor round-trips that destroy performance.

