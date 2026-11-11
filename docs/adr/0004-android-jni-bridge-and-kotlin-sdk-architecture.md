# ADR-0004: Native NDK / JNI Bridge Architecture & Kotlin SDK Design

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect) & Agent 3 (Android Engineer)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 6 (QA)

---

## Context

Integrating high-performance C++20 native code into an Android application requires bridging the Java Native Interface (JNI). JNI bridges are traditionally a major source of production instability, memory leaks, global reference exhaustion, and uncatchable native process crashes (`SIGSEGV`, `SIGABRT`).

Furthermore, modern Android applications rely on Kotlin Coroutines and Jetpack Compose; exposing raw asynchronous callbacks or blocking threads across JNI leads to poor developer ergonomics and UI freezes.

## Decision

We establish a **strict C-ABI boundary** for the JNI bridge and wrap it with an idiomatic, reactive Kotlin SDK.

### Structural Requirements:
1. **Narrow C-ABI Boundary (`mobilellm/c_api.h`):** The JNI C++ layer (`mobilellm_jni.cpp`) communicates with the C++ engine exclusively through extern "C" functions. No C++ classes or complex standard library objects cross the JNI boundary.
2. **Crash Prevention (Zero Uncaught Exceptions):** Every JNI function wraps native invocations in `try ... catch (...)` blocks. Native errors are translated into descriptive JVM exceptions (`MobileLLMException`) rather than allowing native aborts.
3. **Reactive Streaming via Kotlin Flow:** Token generation is exposed to Android developers as a cold Kotlin `Flow<TokenChunk>`, powered internally by coroutine channels.
4. **Cooperative Cancellation:** Cancelling the Kotlin coroutine `Job` automatically triggers a callback to `mllm_cancel()`, halting native token decode within the current decode step.
5. **No Local HTTP Server for In-Process Android:** The Android Kotlin SDK directly calls native functions via JNI. It does **not** spin up a local HTTP daemon or route tokens through loopback sockets for internal app use.

## Consequences

### Positive:
- Bulletproof Android stability: no uncaught native crashes.
- Clean Kotlin developer experience with Jetpack Compose lifecycle awareness.
- Zero network socket overhead or port-binding permission issues inside the Android sandbox.

### Negative / Trade-offs:
- Requires maintaining both JNI C++ marshalling code and Kotlin wrapper classes.
- JNI call overhead per token; mitigated by batching token chunk strings if generation throughput exceeds 60 tokens/sec.

## Alternatives Considered

- **Running an Internal Local HTTP Server inside Android:** Some cross-platform tools run a local web server inside an Android background service and make `localhost` HTTP requests. Rejected because it wastes battery, adds socket latency, requires network permissions in `AndroidManifest.xml`, and exposes local ports to other malicious apps on the device.
