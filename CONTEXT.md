# MobileLLM Engine --- Project Context

> **Purpose:** Canonical working context for AI coding agents,
> contributors, reviewers, and maintainers.
>
> **Project:** MobileLLM Engine\
> **Proposed organization:** CodeSec\
> **Proposed repository:** `codesec-ai/mobilellm-engine`\
> **License target:** Apache-2.0, subject to dependency and
> model-license review\
> **Status:** Architecture and bootstrap planning\
> **Initial platform:** Android ARM64\
> **Primary objective:** Build an open-source, mobile-first LLM
> inference runtime inspired by selected vLLM concepts.

------------------------------------------------------------------------

## 1. Project vision

MobileLLM Engine aims to make local language-model inference practical
for Android applications and developers. It should provide a stable
native runtime, mobile-aware resource management, hardware-backend
abstraction, and accessible developer interfaces.

The project is **inspired by vLLM**, not a claim to be a direct port or
a mobile implementation of all vLLM features. Server-oriented concepts
such as request scheduling, continuous batching, and efficient KV-cache
management should be adapted to mobile constraints and implemented only
where the underlying backend supports them.

The runtime must support offline inference after the model and required
runtime assets are present on the device.

### Product principles

1.  **Offline-first:** Core inference must not require a cloud inference
    service.
2.  **Mobile-first:** RAM, battery, thermals, lifecycle, storage, and
    hardware diversity are first-class constraints.
3.  **Native performance:** Put latency- and memory-sensitive inference
    work in native code.
4.  **Developer-friendly:** Provide Kotlin and Python interfaces and an
    OpenAI-compatible local API where appropriate.
5.  **Backend-neutral architecture:** Keep public runtime interfaces
    independent of a specific inference backend.
6.  **Evidence-based optimization:** Never claim speedups or hardware
    acceleration without reproducible measurements.
7.  **Safe by default:** Protect local model files, APIs, device
    resources, and user prompts.
8.  **Incremental independence:** Start with an established inference
    backend, then add original mobile runtime capabilities where they
    create measurable value.

## 2. Goals and non-goals

### Goals

-   Run supported quantized language models locally on Android ARM64
    devices.
-   Initially support GGUF models through an established backend.
-   Provide model loading, unloading, generation, streaming,
    cancellation, and session lifecycle APIs.
-   Integrate native inference with Android through the NDK and JNI.
-   Provide a Kotlin SDK and sample Android application.
-   Provide a Python SDK and developer tools.
-   Expose an optional loopback-only HTTP API with OpenAI-compatible
    endpoints.
-   Add mobile-aware memory accounting, request limits, and scheduling.
-   Evaluate CPU/ARM NEON, Vulkan GPU, and optional vendor NPU backends.
-   Provide reproducible device benchmarks, tests, documentation, and
    release automation.

### Non-goals for the initial release

-   Reimplement every transformer architecture or inference kernel from
    scratch.
-   Guarantee compatibility with every GGUF file or model family.
-   Promise server-class throughput or vLLM feature parity on phones.
-   Require cloud APIs, accounts, or network connectivity for inference.
-   Assume all Android GPUs expose identical Vulkan capabilities.
-   Claim NPU execution unless supported operators demonstrably execute
    on the NPU.
-   Store inference tensors or KV-cache contents in SQLite.
-   Run a Python interpreter inside the Android application as a
    requirement.

## 3. Target users and use cases

-   Android developers embedding local AI features in applications.
-   Python developers controlling a local Android inference runtime.
-   Open-source contributors researching mobile inference and runtime
    design.
-   Users who need offline text generation and local model execution.
-   Device and runtime engineers measuring model performance under
    mobile resource constraints.

Representative use cases include offline chat, local text completion,
on-device assistants, private document workflows, and
application-specific generation.

## 4. Architecture overview

The intended high-level flow is:

``` text
Android application                 Python developer tools
(Kotlin / Jetpack Compose)           (Python SDK / CLI)
             |                                |
             v                                v
       Kotlin SDK                       Python client
             |                                |
             v                                v
       JNI boundary                  HTTP / local API
             |                                |
             +---------------+----------------+
                             |
                             v
                  MobileLLM Runtime API
             Model manager / sessions / scheduler
                 streaming / metrics / config
                             |
                             v
                    Native C++ runtime
          backend abstraction / sampling / lifecycle
                             |
               +-------------+-------------+
               |             |             |
               v             v             v
            CPU/NEON       Vulkan       Optional NPU
                                         vendor APIs
```

The HTTP server is optional and should not be required for direct
in-process Android use. The Kotlin SDK should be able to call the native
runtime through JNI without routing requests through a local HTTP
server.

The Python SDK may connect to a running Android runtime over an
explicitly enabled local or LAN API. Optional native Python bindings may
be provided for supported desktop platforms, but they are not a
dependency of Android inference.

## 5. Technology decisions

  -----------------------------------------------------------------------
  Area                    Initial choice          Notes
  ----------------------- ----------------------- -----------------------
  Core language           C++20                   Native runtime and
                                                  performance-sensitive
                                                  components

  Initial inference       llama.cpp               Baseline backend;
  backend                                         integrate rather than
                                                  rewrite initially

  Initial model format    GGUF                    Validate architecture
                                                  and operator
                                                  compatibility

  Android integration     Kotlin                  Public Android SDK and
                                                  sample application

  Native bridge           Android NDK / JNI       Keep JNI surface small
                                                  and lifecycle-safe

  Build                   CMake + Gradle          Reproducible native and
                                                  Android builds

  Python                  Python 3.11+            SDK, CLI, tooling,
                                                  optional desktop
                                                  bindings

  GPU                     Vulkan                  Capability-gated
                                                  backend

  CPU                     ARM64 / NEON            Baseline mobile CPU
                                                  target

  NPU                     Optional vendor         Qualcomm QNN or other
                          backends                supported integrations
                                                  after feasibility
                                                  review

  Metadata                SQLite                  Persistent metadata
                                                  only, not inference
                                                  tensors

  API                     REST + SSE              Optional local service
                                                  and streaming

  C++ tests               GoogleTest              Unit and integration
                                                  tests

  Python tests            pytest                  SDK and tooling tests

  CI                      GitHub Actions          Build, test, security,
                                                  and release workflows
  -----------------------------------------------------------------------

### Backend policy

-   Establish a correct CPU baseline before GPU optimization.
-   Detect backend capabilities at runtime.
-   Fall back safely to CPU when acceleration is unavailable or
    unsupported.
-   Test numerical correctness for every backend.
-   Do not assume that a backend supports every operator, model,
    quantization, or device.
-   Keep backend-specific code isolated under `backends/`.

## 6. Repository structure

The following is the intended monorepo layout. The team may refine it
through reviewed architecture decisions, but module ownership and
dependency boundaries must remain clear.

``` text
mobilellm-engine/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── android-build.yml
│   │   ├── python-tests.yml
│   │   ├── native-tests.yml
│   │   ├── security.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── architecture/
│   ├── adr/
│   ├── api/
│   ├── model-support/
│   ├── benchmarks/
│   ├── security/
│   └── development/
├── runtime/
│   ├── include/mobilellm/
│   ├── src/
│   │   ├── engine/
│   │   ├── tokenizer/
│   │   ├── inference/
│   │   ├── sampling/
│   │   └── model_loader/
│   ├── tests/
│   └── CMakeLists.txt
├── backends/
│   ├── common/
│   ├── cpu/
│   ├── vulkan/
│   ├── qnn/
│   └── nnapi/
├── scheduler/
│   ├── request_queue/
│   ├── session_manager/
│   ├── kv_cache/
│   ├── batching/
│   ├── memory_manager/
│   └── tests/
├── android/
│   ├── app/
│   ├── sdk/
│   ├── runtime/
│   ├── jni/
│   └── instrumentation-tests/
├── python/
│   ├── mobilellm/
│   ├── cli/
│   ├── bindings/
│   └── tests/
├── server/
│   ├── http/
│   ├── openai_compat/
│   ├── streaming/
│   └── security/
├── tools/
│   ├── model-converter/
│   ├── model-validator/
│   ├── benchmark/
│   ├── profiling/
│   └── developer-cli/
├── examples/
│   ├── android-chat/
│   ├── python-inference/
│   ├── local-api/
│   └── offline-inference/
├── tests/
│   ├── integration/
│   ├── performance/
│   ├── compatibility/
│   └── stress/
├── scripts/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── LICENSE
└── VERSION
```

## 7. Module boundaries and ownership

  -----------------------------------------------------------------------
  Role                    Primary ownership       Responsibility
  ----------------------- ----------------------- -----------------------
  Lead architect /        `docs/`, cross-module   Architecture,
  orchestrator            contracts               interfaces, task
                                                  assignment, integration

  C++ runtime engineer    `runtime/`              Native runtime, backend
                                                  integration, model
                                                  lifecycle, generation

  Memory and scheduler    `scheduler/`            Memory budgets,
  engineer                                        sessions, queues,
                                                  batching where
                                                  supported

  Android engineer        `android/`              Kotlin SDK, JNI, NDK
                                                  integration, sample
                                                  application

  Python SDK engineer     `python/`, developer    Python client, CLI,
                          tooling                 packaging, examples

  Hardware engineer       `backends/`             CPU, Vulkan, optional
                                                  NPU integrations and
                                                  profiling

  QA and security         `tests/`, security      Independent
  engineer                review                  verification, fuzzing,
                                                  security, benchmarks
  -----------------------------------------------------------------------

Agents must not edit the same files concurrently without explicit
coordination. Shared public headers, API schemas, build configuration,
and cross-module interfaces require architect review.

## 8. AI agent operating rules

The project uses multiple cloud-based coding agents with specialized
responsibilities.

### Required workflow

1.  Architect writes an issue with scope, dependencies, interface
    expectations, and acceptance criteria.
2.  One agent owns the implementation task.
3.  The agent works in an isolated branch or worktree.
4.  The agent adds or updates tests and documentation.
5.  The agent opens a pull request with evidence.
6.  QA independently runs relevant checks.
7.  Architect reviews API and architecture impact.
8.  A human maintainer approves significant changes and releases.
9.  CI must pass before merge.

### Non-negotiable rules

-   Never commit directly to `main`.
-   Never claim code works without executable test evidence.
-   Never invent benchmark numbers or test results.
-   Never silently change a public interface.
-   Never merge failing CI.
-   Never add a dependency without explaining its purpose and license.
-   Do not duplicate inference functionality without a measured reason.
-   Keep changes focused and reviewable.
-   Document limitations and known failures.
-   Require human review for architecture changes, security-sensitive
    code, and releases.

### Agent task report format

Every agent must report:

``` text
Task:
Files changed:
Design decisions:
Implementation summary:
Commands executed:
Tests passed:
Tests failed:
Build evidence:
Benchmark evidence (if applicable):
Known limitations:
Risks:
Commit / pull request:
Next dependency:
```

## 9. Model support strategy

Begin with small quantized models suitable for mobile devices. Candidate
model families include:

-   Llama 3.2
-   Qwen 2.5
-   Qwen3
-   Gemma 3
-   Phi
-   Mistral

Initial smoke-test candidates include small variants around 0.6B, 1B,
and 1.7B parameters where available in a compatible format.

Model support must be explicitly tested and documented. A `.gguf`
extension alone does not guarantee compatibility. Record model
architecture, quantization, tokenizer requirements, context settings,
backend support, and model license.

Do not redistribute model weights unless their licenses permit it.
Provide model acquisition instructions that respect the relevant terms.

## 10. Runtime API direction

The native API should be independent of HTTP and Android UI. A
conceptual interface may include:

``` cpp
class Engine {
public:
    LoadResult load_model(const ModelConfig& config);
    UnloadResult unload_model(const ModelId& model);
    Session create_session(const SessionConfig& config);
    GenerationHandle generate(
        const SessionId& session,
        const GenerationRequest& request,
        TokenCallback callback
    );
    void cancel(const GenerationHandle& handle);
    RuntimeMetrics metrics() const;
};
```

This is illustrative, not a finalized ABI. The architect must define
concrete types, ownership, threading, error handling, and ABI stability
before implementation.

### Initial optional HTTP endpoints

  Method   Endpoint                 Purpose
  -------- ------------------------ ---------------------------------
  GET      `/v1/health`             Runtime health
  GET      `/v1/models`             List available or loaded models
  POST     `/v1/models/load`        Load a model
  POST     `/v1/models/unload`      Unload a model
  POST     `/v1/chat/completions`   Chat generation
  POST     `/v1/completions`        Text completion
  GET      `/v1/metrics`            Runtime metrics
  DELETE   `/v1/sessions/{id}`      Delete a session

Use SSE for streaming responses. Define cancellation, error payloads,
timeouts, request limits, and API versioning. The server must be
optional.

## 11. Data and storage design

The initial runtime should work offline. SQLite may store metadata such
as:

-   Models: ID, name, path, format, size, checksum, status.
-   Sessions: ID, model ID, context configuration, status.
-   Requests: ID, session ID, status, timestamps.
-   Settings: key and value.
-   Benchmarks: model, device, configuration, measured metrics.
-   Devices: hardware and capability metadata.
-   Cache metadata: configuration and accounted allocation.

Do not store KV-cache tensor contents in SQLite. Tensor memory belongs
to the native inference backend. Avoid persisting prompts or generated
content unless a user-facing feature explicitly requires it and the
privacy implications are documented.

## 12. Implementation roadmap

The planning baseline is approximately 24 weeks; this is an estimate,
not a delivery guarantee.

  ------------------------------------------------------------------------
  Phase                            Indicative timing Main outcome
  --------------------- ---------------------------- ---------------------
  0\. Architecture and                    Weeks 1--2 Repository,
  foundation                                         interfaces, CI, ADRs

  1\. Native inference                    Weeks 3--5 GGUF model loads and
  MVP                                                generates

  2\. Android runtime                     Weeks 6--8 Physical-device
                                                     offline inference

  3\. Memory management                  Weeks 9--11 Memory budgets,
                                                     cleanup, sessions

  4\. Scheduler and                     Weeks 12--14 Queues, optional
  serving                                            batching, API

  5\. Hardware                          Weeks 15--17 Measured CPU/GPU
  acceleration                                       improvements

  6\. Python ecosystem                  Weeks 18--19 SDK, CLI, examples

  7\. Security                          Weeks 20--22 Fuzzing, validation,
  hardening                                          reliability

  8\. Release                           Weeks 23--24 Documentation and
  preparation                                        v1.0 release
  ------------------------------------------------------------------------

### First 14 days

    Day Task                                    Owner
  ----- --------------------------------------- ---------------------
      1 Repository and project specification    Architect
      2 Native runtime interfaces               Architect + runtime
      3 CMake and Android build skeleton        Runtime + Android
      4 Integrate initial inference backend     Runtime
      5 Model-loading test                      Runtime + QA
      6 JNI bridge                              Android
      7 First physical Android inference test   Android + QA
      8 Python client skeleton                  Python
      9 Memory budget interface                 Scheduler
     10 Request/session lifecycle               Scheduler
     11 Streaming API                           Runtime + Python
     12 Device profiling                        Hardware + QA
     13 Integration and regression tests        QA
     14 Demo and milestone review               Architect

## 13. First milestone acceptance criteria

The first milestone is complete only when all of the following are
demonstrated:

1.  The native project builds reproducibly.
2.  A supported GGUF model loads successfully.
3.  Inference runs on a physical Android ARM64 device.
4.  Text generation streams to the Android application.
5.  Generation cancellation works.
6.  Inference works with network access disabled after model setup.
7.  Memory use is measured and recorded.
8.  Errors are surfaced without crashing the application.
9.  Automated tests pass.
10. Documentation describes the device, model, configuration,
    limitations, and reproduction steps.

An emulator-only result does not satisfy the physical-device acceptance
criterion.

## 14. Testing and quality strategy

### Required checks

  Area                      Tools / methods
  ------------------------- --------------------------------
  C++ unit tests            GoogleTest
  Python tests              pytest
  Python typing             mypy or pyright
  Formatting                clang-format
  Static analysis           clang-tidy
  Native memory safety      ASan / UBSan where supported
  Race detection            TSan where supported
  Fuzzing                   libFuzzer
  Android instrumentation   AndroidX Test
  API tests                 pytest and HTTP contract tests
  Dependency security       Dependency scanning
  CI                        GitHub Actions

### Device classes

Test at least:

-   Entry-level: approximately 4--6 GB RAM.
-   Mid-range: approximately 8 GB RAM.
-   Flagship: approximately 12--16 GB RAM.

These are test categories, not guarantees that every model will fit or
perform acceptably on every device in a category.

### Benchmark metrics

Record:

-   Model load duration.
-   Time to first token.
-   Prompt processing throughput.
-   Generation tokens per second.
-   Peak resident memory.
-   Model storage footprint.
-   Battery consumption.
-   Temperature over time.
-   Crash and error rate.
-   Backend correctness.

Every result must include device model, Android version, chipset,
runtime version, model identifier, quantization, context length,
backend, test prompt, warm-up procedure, and measurement method. Compare
like-for-like configurations. Do not publish unsupported performance
claims.

## 15. Security and privacy requirements

### Model and file security

-   Validate model metadata and file size.
-   Verify checksums where available.
-   Reject malformed or unsupported model files safely.
-   Prevent path traversal.
-   Use controlled storage locations.
-   Do not execute downloaded model content as code.

### API security

-   Bind to loopback by default.
-   Require explicit configuration for LAN exposure.
-   Apply request-size and concurrency limits.
-   Validate all input fields.
-   Protect session ownership.
-   Provide predictable cancellation and timeout behavior.
-   Avoid logging prompts and generated text by default.

### Native security

-   Use safe ownership and RAII.
-   Minimize unsafe pointer exposure across JNI.
-   Validate all lengths and buffer boundaries.
-   Run sanitizers and fuzzers where supported.
-   Handle allocation failures and backend errors.
-   Test malformed inputs and repeated load/unload cycles.

### Supply chain

-   Pin dependency versions.
-   Review licenses and notices.
-   Generate an SBOM.
-   Scan dependencies.
-   Protect release workflows.
-   Sign release artifacts where practical.

## 16. Performance and engineering risks

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Model exceeds available RAM         Memory budgets, smaller
                                      quantizations, explicit context
                                      limits

  Thermal throttling                  Sustained-load profiling and
                                      adaptive resource policies

  GPU incompatibility                 Runtime capability detection and
                                      CPU fallback

  NPU fragmentation                   Treat vendor backends as optional,
                                      isolated integrations

  JNI crashes                         Narrow JNI boundary and
                                      instrumentation tests

  Backend API limitations             Document capabilities and avoid
                                      unsupported abstractions

  Model incompatibility               Explicit architecture and
                                      quantization test matrix

  Agent integration conflicts         Module ownership, contracts,
                                      isolated branches

  Unreliable benchmark claims         Reproducible scripts and device
                                      metadata

  License incompatibility             Dependency and model-license review
                                      before distribution
  -----------------------------------------------------------------------

## 17. Release strategy

Suggested release progression:

-   `v0.1.0` --- Native inference proof of concept.
-   `v0.2.0` --- Android JNI runtime.
-   `v0.3.0` --- Kotlin SDK and demo.
-   `v0.4.0` --- Memory and session management.
-   `v0.5.0` --- Optional local API and Python client.
-   `v0.6.0` --- Experimental Vulkan backend.
-   `v0.9.0` --- Release candidate and compatibility testing.
-   `v1.0.0` --- Stable documented release.

Only call a feature stable after its compatibility, test, and
documentation requirements are satisfied.

## 18. Immediate next actions

The next engineering session should:

1.  Confirm the repository name and license.
2.  Create the repository skeleton.
3.  Write `ARCHITECTURE.md` and initial ADRs.
4.  Pin and document the initial inference backend dependency.
5.  Define the C++ public interface and error model.
6.  Establish an Android ARM64 build in CI.
7.  Build the smallest working physical-device inference demonstration.
8.  Record the first reproducible benchmark.

**Do not start all agents on broad implementation simultaneously.**
First agree on module interfaces and get the smallest end-to-end
inference path working. Then parallelize memory management, Android SDK,
Python tooling, hardware experiments, and independent QA.

------------------------------------------------------------------------

## 19. Source references

-   llama.cpp: https://github.com/ggml-org/llama.cpp
-   MLC LLM: https://github.com/mlc-ai/mlc-llm
-   Android NDK: https://developer.android.com/ndk
-   Android graphics and Vulkan:
    https://developer.android.com/ndk/guides/graphics
-   Vulkan: https://www.vulkan.org/
-   GoogleTest: https://github.com/google/googletest
-   OSS-Fuzz: https://github.com/google/oss-fuzz

------------------------------------------------------------------------

**Document maintenance:** Treat this file as the shared project context
for coding agents. Update it when an approved architecture decision
changes project scope, interfaces, technology choices, ownership, or
release criteria. Record significant decisions separately in
`docs/adr/`.
