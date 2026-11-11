# ADR-0005: Hardware Backend Abstraction & Automatic CPU Fallback Policy

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect) & Agent 5 (Hardware Engineer)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 6 (QA)

---

## Context

Android hardware is notoriously fragmented. A given device may feature a Qualcomm Snapdragon, MediaTek Dimensity, Google Tensor, or Samsung Exynos chipset. The available GPU may be an Adreno, Mali, or Immortalis architecture, each running different proprietary GPU driver revisions with varying Vulkan feature support and compiler bugs.

Attempting to run GPU or NPU acceleration without strict capability probing and automatic fallback leads to widespread app crashes, visual corruption, or silent numerical failures in production.

## Decision

We establish an **evidence-based Hardware Abstraction Layer (HAL)** with mandatory runtime capability detection and automatic, graceful fallback to the CPU.

### Rules of Hardware Execution:
1. **CPU / ARM NEON is the Unconditional Golden Standard:**
   - The CPU backend must always be fully functional and tested first.
   - All models, quantizations, and operators must be verified on CPU before attempting GPU/NPU execution.
2. **Dynamic Capability Gating (No Device-Name Hardcoding):**
   - Vulkan is not enabled based on marketing names or OS version strings.
   - At runtime, MobileLLM initializes a `VkInstance` and verifies critical physical device extensions (`VK_KHR_16bit_storage`, `VK_KHR_shader_float16_int8`, `VK_KHR_storage_buffer_storage_class`).
   - A dry-run compilation of a trivial test compute shader is executed before loading real model weights into GPU buffers.
3. **Graceful Fallback:**
   - If Vulkan device creation fails, if extension probing fails, or if shader compilation encounters an internal driver bug, the runtime logs the exact diagnostic error and immediately switches to `CPU_NEON`.
   - The inference request completes successfully on CPU rather than terminating with an application crash.
4. **Isolated Vendor NPU Architecture:**
   - Experimental integrations (Qualcomm QNN, Android NNAPI) must reside in strictly decoupled modules (`backends/qnn/`, `backends/nnapi/`).
   - Vendor shared libraries are loaded dynamically via `dlopen`. If absent, the NPU backend reports `UNAVAILABLE` and falls back cleanly.

## Consequences

### Positive:
- Maximum device reach: MobileLLM runs reliably across any modern ARM64 Android device.
- Eliminates driver-crash surprises on low-tier or buggy GPU implementations.
- Isolates high-risk vendor code from core engine stability.

### Negative / Trade-offs:
- CPU fallback results in lower tokens/second if GPU is unavailable; user experience must gracefully indicate that fallback occurred.

## Alternatives Considered

- **Mandatory Vulkan Acceleration:** Forcing GPU compute would immediately disqualify devices with broken Vulkan drivers or unsupported extensions, reducing Android device compatibility by ~30–40%.
