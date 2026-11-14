# MobileLLM Engine — Compatibility Matrix & Support Lifecycle

> **Governing Release Policy:** [`docs/release/RELEASE_POLICY.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/release/RELEASE_POLICY.md)  
> **Maintainer:** Agent 7 (Principal Integration & Release Engineer)  
> **Current Version:** `0.1.0-dev`  
> **Last Updated:** November 14, 2026  

---

## 1. Hardware Architecture & ABI Support

| Architecture / ABI | Support Tier | Primary Optimization | 16KB Page Aligned | Minimum Requirements |
| :--- | :---: | :--- | :---: | :--- |
| **`arm64-v8a` (AArch64)** | **Tier 1 (Primary)** | ARM NEON SIMD, fp16 vectorization | **Yes (Enforced)** | ARMv8.0-A CPU, 4GB+ System RAM |
| **`x86_64` (AMD64)** | **Tier 2 (Supported)** | AVX, AVX2, FMA3 | N/A | SSE4.2 / AVX2 CPU (Dev / Emulator) |
| **`armeabi-v7a` (32-bit ARM)** | **Unsupported** | None | N/A | **Deprecated/Unsupported** (32-bit address space insufficient for LLM tensors) |
| **`x86` (32-bit x86)** | **Unsupported** | None | N/A | **Deprecated/Unsupported** |
| **`arm64-v8.2a+` (DotProd/I8MM)** | **Tier 1 (Planned Dispatch)** | DotProd (`FEAT_DotProd`), I8MM (`FEAT_I8MM`) | **Yes** | Snapdragon 8 Gen 1+, Dimensity 9000+, Tensor G2+ |

> [!IMPORTANT]
> **Android 15 (API 35+) 16KB Memory Page Alignment:**  
> All native shared libraries (`libmobilellm_runtime.so`, `libmobilellm_jni.so`, `libllama.so`) built for `arm64-v8a` pass `-Wl,-z,max-page-size=16384` to guarantee zero page-fault crashes on 16KB-kernel Android devices.

---

## 2. Operating System & Platform Compatibility

| Operating System | Minimum Version | Tested & Validated | Support Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Android** | Android 9.0 (API 28) | Android 13 (API 33), 14 (API 34), 15 (API 35) | **Tier 1** | Target SDK 34; tested on physical Galaxy, Pixel, and arm64 emulators. |
| **macOS** | macOS 12.0 (Monterey) | macOS 14 (Sonoma), 15 (Sequoia) | **Tier 1** | Full native Darwin support on Apple Silicon (`arm64`) and Intel (`x86_64`). |
| **Linux** | Ubuntu 20.04 LTS (glibc 2.31+) | Ubuntu 22.04 LTS, Debian 12 | **Tier 1** | Primary CI and server development environment. |
| **Windows (WSL2)** | Windows 10 Build 19041+ | Windows 11 with Ubuntu 22.04 WSL2 | **Tier 2** | Full runtime support via Linux subsystem. |
| **Windows (Native MSVC)** | Windows 10 64-bit | Planned (CMake MSVC build pending) | **Tier 3** | Planned for desktop deployment in Milestone 4. |
| **iOS** | iOS 16.0+ | Planned (C-ABI clean, Swift package planned) | **Tier 3** | Architectural decoupling ready; awaiting Swift bindings. |

---

## 3. Toolchain & Build System Requirements

| Tool / Dependency | Supported Range | Verified Version | Configuration Flag / Target |
| :--- | :--- | :--- | :--- |
| **C++ Standard** | ISO C++20 | ISO C++20 | `-std=c++20` (Strict, `-Werror` clean) |
| **Clang / LLVM** | Clang 14.0 — 19.0 | AppleClang 16.0.0 / Clang 18.1 | Primary runtime compiler |
| **GCC** | GCC 11.2 — 14.2 | GCC 12.3 / 13.2 | Supported on Linux CI |
| **CMake** | 3.22.1 — 3.30+ | 3.22.1 (Android NDK) / 3.31.5 (Host) | `-S . -B build` |
| **Android NDK** | r26b — r27c | NDK 27.1.12297006 | `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-28` |
| **Java / JDK** | OpenJDK 17 or 21 | OpenJDK 21.0.13 | Required for Gradle 8.5+ Android builds |
| **Android Gradle Plugin** | 8.2.0 — 8.5.2 | AGP 8.2.0 | `android/build.gradle.kts` |
| **Python** | 3.10, 3.11, 3.12, 3.13 | 3.11.11 / 3.12.7 | Required for Python SDK & test suites |

---

## 4. GGUF Quantization & Model Format Matrix

MobileLLM Engine interfaces directly with GGUF v2 and v3 specifications via `llama.cpp` (b4260 runtime core):

| Quantization Type | Status | Speed on Mobile | Quality (PPL) | Recommended Mobile RAM | Best Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Q4_K_M`** | **Optimal** | ★★★★★ | ★★★★☆ | 4GB — 8GB | **Default Recommendation** for 1B–3B parameter models. |
| **`Q4_0`** | **Fully Supported** | ★★★★★ | ★★★☆☆ | 4GB — 6GB | Ultra-low compute overhead, legacy compatibility. |
| **`Q4_K_S`** | **Fully Supported** | ★★★★★ | ★★★☆☆ | 4GB — 6GB | Compact footprint for strictly budget devices. |
| **`Q5_K_M`** | **Supported** | ★★★★☆ | ★★★★★ | 6GB — 8GB | Higher reasoning precision on 6GB+ devices. |
| **`Q8_0`** | **Supported** | ★★★☆☆ | ★★★★★ | 8GB — 12GB | Reference precision, validation, and benchmarking. |
| **`F16 / BF16`** | **Supported** | ★★☆☆☆ | ★★★★★ | 12GB+ | Heavy memory bandwidth; only recommended for small sub-500M models. |
| **`IQ2_XS / IQ3_XXS`** | **Experimental** | ★★★☆☆ | ★★☆☆☆ | 3GB — 4GB | Severe quantization loss; supported via underlying ggml. |

---

## 5. Model Architecture Verification

The following open-weight LLM architectures have been verified against MobileLLM Engine:

| Model Architecture | Representative Models | Tested Quantizations | Context Length Verified | Verification Status |
| :--- | :--- | :--- | :---: | :---: |
| **LLaMA / LLaMA-2 / LLaMA-3** | LLaMA-3.2-1B, LLaMA-3.2-3B | Q4_K_M, Q8_0 | 2048, 4096 | **Verified Working** |
| **SmolLM / SmolLM2** | SmolLM2-135M, SmolLM2-360M, 1.7B | Q4_K_M, Q8_0 | 2048 | **Verified Working (Live Native & Pytest)** |
| **Qwen2 / Qwen2.5** | Qwen2.5-0.5B, Qwen2.5-1.5B | Q4_K_M, Q5_K_M | 2048, 4096 | **Verified Working** |
| **Gemma 2** | Gemma-2-2B-IT | Q4_K_M | 2048 | **Verified Working** |
| **Phi-3 / Phi-3.5** | Phi-3.5-mini-instruct (3.8B) | Q4_K_M | 2048 | **Verified Working** |
| **DeepSeek-R1 Distill** | DeepSeek-R1-Distill-Qwen-1.5B | Q4_K_M | 2048 | **Verified Working** |

---

## 6. Language SDK & Binding Compatibility

| SDK Layer | Language / Runtime | Distribution Mechanism | Min Consumer Version | Invariant Enforced |
| :--- | :--- | :--- | :--- | :--- |
| **C++ Core Runtime** | C++20 | Shared Library (`.so`, `.dylib`) | C++20 compiler | Strict C-ABI boundary in [`c_api.h`](file:///Users/Apple/Developer/ai-apps/MobileLLM/runtime/include/mobilellm/c_api.h) |
| **Android SDK** | Kotlin 1.9+, Java 17+ | Maven AAR (`mobilellm-sdk.aar`) | Android API 28+ | Thread-safe Coroutines, JNI boundary serialization |
| **Python SDK** | Python 3.10 — 3.13 | Wheel (`mobilellm-0.1.0-py3-none-any.whl`) | CPython 3.10+ | Strict `ctypes` FFI bindings, async streaming generator |
| **React Native** | TypeScript, Hermes | NPM Package (`@codesec/mobilellm-rn`) | React Native 0.73+ | *Planned for Milestone 4 (JSI Turbomodule architecture)* |
| **Expo** | TypeScript | Expo Config Plugin | Expo SDK 50+ | *Planned for Milestone 4* |

---

## 7. Known Architectural Limitations & Deprecations

1. **32-bit Architecture Drop:** Support for `armeabi-v7a` and `x86` is permanently dropped. LLM weights and KV cache tensors cannot operate reliably within a 32-bit (2GB virtual) address space.
2. **Android NNAPI Deprecation:** Android NNAPI is formally deprecated by Google starting in Android 15 (API 35). MobileLLM Engine does not invest in legacy NNAPI backends, prioritizing direct Vulkan and native NPU SDKs (ADR-0009).
3. **Physical Paged Attention:** Paged physical KV caches remain unmerged pending Milestone 3. Current memory admission control uses virtual KV accounting while `llama.cpp` maintains physical memory ownership.
4. **Vulkan GPU Mobile Cross-Compilation:** Vulkan compute backend requires SPIR-V shader compilation via host `glslc` or pre-compiled C++ headers during cross-compilation builds.
