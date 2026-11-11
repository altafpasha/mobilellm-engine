# NEXA Android Application & MobileLLM Kotlin SDK

[![Android MinSdk](https://img.shields.io/badge/MinSdk-28-brightgreen.svg)](https://developer.android.com)
[![NDK Version](https://img.shields.io/badge/NDK-r27-blue.svg)](https://developer.android.com/ndk)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.9.22-purple.svg)](https://kotlinlang.org)
[![Compose](https://img.shields.io/badge/Jetpack_Compose-Material3-green.svg)](https://developer.android.com/jetpack/compose)
[![Security](https://img.shields.io/badge/Network_Access-Zero%20(100%25%20Offline)-success.svg)](#offline-guarantee)

**NEXA** is a production-grade, 100% offline edge AI chat application for Android, powered by the native **MobileLLM C++20 engine**.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |               NEXA App                |
                      |   (Jetpack Compose UI + ViewModel)    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |          MobileLLM Kotlin SDK         |
                      | (MobileLLMClient / Flow<TokenChunk>)  |
                      +-------------------+-------------------+
                                          | JNI (Direct C-ABI)
                                          v
                      +---------------------------------------+
                      |         libmobilellm_jni.so           |
                      |      (C++20 JNI Bridge Layer)         |
                      +-------------------+-------------------+
                                          | C-ABI Linkage
                                          v
                      +---------------------------------------+
                      |         MobileLLM Core Engine         |
                      | (Paged KV-Cache, Sampler, llama.cpp)  |
                      +-------------------+-------------------+
                                          |
                                          v
                              Qualcomm Hexagon / ARM NEON / Adreno GPU
```

---

## Modules

- `:sdk` — **MobileLLM Kotlin SDK**:
  - `MobileLLMClient`: Engine lifecycle management and model loader.
  - `ModelSession`: Active inference session wrapping the native handle.
  - `generate(prompt, params)`: Returns a cold Kotlin `Flow<TokenChunk>` with real-time streaming tokens.
  - `NativeBridge`: Direct JNI bindings into `libmobilellm_jni.so`.
  - Android lifecycle integration responding to `onTrimMemory` / `onLowMemory`.

- `:app` — **NEXA Application**:
  - 100% Jetpack Compose Material 3 interface.
  - Streaming conversational UI with automatic scrolling and markdown formatting.
  - Generation telemetry badges: Tokens/s, Time-to-First-Token (TTFT), Peak RSS Memory, Output Token Count.
  - Live cooperative cancellation: Stop token generation instantly.
  - SAF (Storage Access Framework) model importer for `.gguf` weights.
  - Hardware & Diagnostics dialog displaying ARM NEON, CPU Cores, and OS RAM pressure.
  - Runtime inference settings (Temperature, Top-P, Top-K, Repetition Penalty, Threads, Backend).

---

## Offline Guarantee & Security

- **Strict Zero-Network Policy**: `AndroidManifest.xml` does **NOT** request `android.permission.INTERNET`.
- **Private App Sandboxing**: Models imported via SAF are copied exclusively into `context.filesDir/models/`.
- **Zero-SQLite for Tensors**: KV caches and model weights are mapped directly in native virtual memory (`mmap`).

---

## Building the Project

### Prerequisites
- Android SDK (API 34, Build-Tools 34.0.0+)
- Android NDK (r27 / `27.1.12297006`)
- Java 17 or 21 (OpenJDK)
- CMake 3.22.1+

### Build Commands

```bash
# 1. Run SDK Unit Tests
./gradlew :sdk:testDebugUnitTest

# 2. Build SDK AAR
./gradlew :sdk:assembleRelease

# 3. Build NEXA Debug APK
./gradlew :app:assembleDebug

# 4. Install onto connected physical ARM64 device
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

