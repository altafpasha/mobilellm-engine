# Agent 3 Specification — Android Engineer

> **Module Ownership:** `android/`, `examples/android-chat/`  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Agent 6 (QA Device Testing), End-User Developers

---

## 1. Role Purpose & Domain Ownership

Agent 3 is responsible for delivering an idiomatic, production-grade Android integration. This encompasses the JNI bridge (`libmobilellm_jni.so`), native NDK build scripts, the modern Kotlin SDK (`mobilellm-sdk`), Android lifecycle integration, and the reference Jetpack Compose chat application.

### Complete Directory Ownership
```
android/
├── build.gradle.kts           # Root Android Gradle build configuration
├── sdk/                       # Public Kotlin SDK library module (AAR)
│   ├── src/main/
│   │   ├── java/ai/codesec/mobilellm/
│   │   │   ├── MobileLLMClient.kt     # Main entry point & lifecycle
│   │   │   ├── ModelSession.kt        # Multi-turn generation session
│   │   │   ├── models/                # Kotlin data classes & configs
│   │   │   └── NativeBridge.kt        # Internal external JNI declarations
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── jni/                       # JNI C++ implementation
│   ├── mobilellm_jni.cpp      # JNI method implementations & exception guards
│   ├── jni_helpers.h          # String conversion, callback mapping
│   └── CMakeLists.txt         # NDK CMake build configuration
├── app/                       # Reference Jetpack Compose Chat Application
│   ├── src/main/
│   │   ├── java/ai/codesec/mobilellm/app/
│   │   │   ├── MainActivity.kt
│   │   │   ├── ui/                    # Compose UI screens & chat bubbles
│   │   │   └── viewmodel/             # ViewModel with StateFlow streaming
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
└── instrumentation-tests/     # AndroidX physical device instrumentation tests
```

---

## 2. Technical Standards & Constraints

1. **JNI Safety & Boundary Discipline:**
   - Keep the JNI boundary narrow: route all operations through the stable C-ABI (`c_api.h`).
   - Catch all C++ exceptions within the JNI layer. Never allow a C++ exception to propagate across the JNI boundary into the ART runtime, which causes an instant SIGABRT crash.
   - String Handling: Use `GetStringUTFChars` and guarantee `ReleaseStringUTFChars` is called in `finally` or RAII guards. Avoid large intermediate string allocations on the JVM heap.
2. **Modern Kotlin Concurrency:**
   - Expose token streaming exclusively via Kotlin Coroutines `Flow<TokenChunk>`.
   - Provide non-blocking `suspend` functions for model loading, session creation, and generation.
   - Handle cancellation reactively via `Job.cancel()`, which triggers the native cancellation callback.
3. **Android Lifecycle Integration:**
   - Integrate with `LifecycleObserver` and `ComponentCallbacks2`.
   - Forward `onTrimMemory` signals directly to the native memory manager.
   - Support foreground service execution for sustained background generation tasks with appropriate Android notification channels.
4. **Target Platform Specifications:**
   - `minSdkVersion`: 26 (Android 8.0 Oreo).
   - `targetSdkVersion`: 34 (Android 14).
   - Primary ABI: `arm64-v8a` (strictly required for inference).
   - Secondary ABI: `x86_64` (development emulator sanity only).
5. **Physical Device Validation:**
   - Emulators do not satisfy acceptance gates for performance or stability. Real physical ARM64 hardware testing is mandatory for Milestone 1.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T3.1** | Android Project & NDK Setup | Agent 0 Contract | `android/build.gradle.kts` | Clean Gradle configuration; downloads NDK r26c+ |
| **T3.2** | JNI Native Bridge | Agent 1 (`c_api.h`) | `android/jni/` | Compiles `libmobilellm_jni.so`; zero memory leaks |
| **T3.3** | Kotlin SDK Core Classes | T3.2 | `android/sdk/` | `MobileLLMClient` loads model and initiates session |
| **T3.4** | Flow Streaming Engine | T3.3 | `android/sdk/` | Streams native tokens chunk-by-chunk to `Flow<String>` |
| **T3.5** | Cooperative Cancellation in Kotlin | T3.4 | `android/sdk/` | Cancelling coroutine stops native inference within 50ms |
| **T3.6** | Jetpack Compose Sample Chat App | T3.4 | `android/app/` | Offline chat UI displaying TTFT, tok/s, and message history |
| **T3.7** | Physical Device Test Suite | T3.6 | `instrumentation-tests/` | Runs on physical ARM64 device in Airplane Mode successfully |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 1 (Runtime):** Stable `c_api.h` and compiled native engine archive/library.
- **From Agent 2 (Scheduler):** Memory budget constraints and trim callbacks.

### Outputs Provided to Other Modules:
- **To End Users / App Developers:** `mobilellm-sdk.aar` package and sample Compose application.
- **To Agent 6 (QA & Security):** Android instrumentation test APKs for automated device testing.
