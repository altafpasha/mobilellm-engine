# Agent 5 Specification — Hardware & Acceleration Engineer

> **Module Ownership:** `backends/`, `tools/profiling/`, `tools/benchmark/`  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Agent 1 (Runtime Backend Integration), Agent 6 (Benchmarking)

---

## 1. Role Purpose & Domain Ownership

Agent 5 is responsible for hardware-level performance, low-level compute backends, SIMD vectorization, GPU acceleration via Vulkan, vendor NPU evaluation (Qualcomm QNN, Android NNAPI), and device profiling (thermals, battery, and clock scaling).

### Complete Directory Ownership
```
backends/
├── common/                    # IBackend interface & capability detection
│   ├── backend_interface.h    # Common C++ backend abstraction
│   ├── capability_detector.h  # Device feature & extension query
│   └── backend_registry.h     # Dynamic backend registration
├── cpu/                       # ARM64 CPU & NEON SIMD compute path
│   ├── cpu_backend.cpp        # CPU context & thread-pool orchestration
│   └── neon_math.h            # ARM NEON intrinsics & quantized dot products
├── vulkan/                    # Vulkan Compute backend
│   ├── vulkan_backend.cpp     # VkInstance, VkDevice, queue lifecycle
│   ├── shader_compiler.cpp    # SPIR-V pipeline cache & compilation
│   └── shaders/               # GLSL/HLSL compute shaders for matmul/attention
├── qnn/                       # Experimental Qualcomm QNN integration (Isolated)
├── nnapi/                     # Experimental Android NNAPI integration (Isolated)
└── CMakeLists.txt             # Hardware backends CMake build configuration
tools/
├── benchmark/                 # Automated benchmarking harness (TTFT, TPS, RSS)
└── profiling/                 # Thermal & battery logging scripts (adb / sysfs)
```

---

## 2. Technical Standards & Constraints

1. **Evidence-Based Optimization Principle:**
   - **Never claim a speedup without reproducible data.** Every optimization pull request must include before/after benchmark results from physical hardware, detailing device model, SoC, thermal state, and exact prompt parameters.
2. **Golden CPU Baseline:**
   - The ARM64 CPU backend utilizing ARM NEON SIMD intrinsics is the foundational reference implementation.
   - All accelerated backends (Vulkan, NPU) must demonstrate numerical equivalence against the CPU baseline (e.g., maximum logit divergence $\Delta < 10^{-4}$ for fp32, or verified perplexity equivalence for fp16).
3. **Vulkan Robustness & Graceful Fallback:**
   - Android GPU drivers (Adreno, Mali, Immortalis) exhibit diverse Vulkan bugs and extension gaps.
   - Never assume Vulkan capability based on device marketing. Proactively probe required extensions at runtime (`VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8`).
   - If Vulkan device creation, memory mapping, or shader pipeline creation fails, immediately fallback to `CPU_NEON` without aborting execution.
4. **Thermal & Battery Awareness:**
   - Monitor SoC thermal throttling using `/sys/class/thermal/` or Android `PowerManager.thermalStatus`.
   - Prevent thermal runaway during long decode sessions by tuning thread pool affinity (e.g., balancing workloads between Cortex-X performance cores and Cortex-A efficiency cores).
5. **Vendor NPU Code Isolation:**
   - Qualcomm QNN and Android NNAPI code must reside strictly in isolated compilation units (`backends/qnn/`, `backends/nnapi/`).
   - Dynamic loading (`dlopen`) must be used for proprietary vendor libraries to avoid hard runtime link dependencies on devices lacking those vendor chips.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T5.1** | `IBackend` Interface & Registry | Agent 0 Contract | `backends/common/` | Defines uniform lifecycle and tensor compute dispatch |
| **T5.2** | ARM64 NEON Optimization | T5.1 | `backends/cpu/` | Measured 20%+ throughput improvement over unvectorized C++ |
| **T5.3** | Vulkan Capability Probing | T5.1 | `backends/vulkan/` | Probes Vulkan 1.1+, float16 extensions; fails safely on bad drivers |
| **T5.4** | Vulkan Matmul Compute Shaders | T5.3 | `backends/vulkan/` | Executes matrix multiplication on mobile GPU cleanly |
| **T5.5** | Graceful CPU Fallback Mechanism | T5.3, T5.4 | `backends/vulkan/` | Forces fallback to CPU if Vulkan fails; zero crashes |
| **T5.6** | Device Benchmark Script | T5.2 | `tools/benchmark/` | Measures TTFT, TPS, peak RSS across 5 runs and outputs JSON |
| **T5.7** | Thermal & Battery Profiler | T5.6 | `tools/profiling/` | Logs CPU clock, temperature, and battery mA over 10 min run |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 1 (Runtime):** Model weight tensor pointers, quantization types, and batch execution requests.
- **From Agent 2 (Scheduler):** Memory limits allocated for GPU/Vulkan buffers.

### Outputs Provided to Other Modules:
- **To Agent 1 (Runtime):** Validated `IBackend` instance for compute dispatch.
- **To Agent 6 (QA & Security):** Profiling tools and numerical equivalence verification tests.
