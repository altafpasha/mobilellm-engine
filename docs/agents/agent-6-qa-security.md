# Agent 6 Specification — QA, Benchmark & Security Engineer

> **Module Ownership:** `tests/`, `.github/workflows/`, Security Policies & Fuzzers  
> **Direct Reports to:** Agent 0 (Lead Architect)  
> **Key Downstream Dependents:** Agent 0 (Milestone Gates), Human Maintainers (Release Security)

---

## 1. Role Purpose & Domain Ownership

Agent 6 serves as the independent verification, testing, and security guardian of MobileLLM Engine. Operating with adversarial skepticism, Agent 6 ensures that all code meets strict quality, stability, memory safety, and security standards before merging or reaching production releases.

### Complete Directory Ownership
```
tests/
├── integration/               # End-to-end multi-module lifecycle tests
├── performance/               # Latency, TTFT, and throughput verification
├── compatibility/             # Model matrix & Android OS compatibility tests
├── stress/                    # Extended duration, high memory, rapid cycle tests
└── fuzzing/                   # libFuzzer targets (GGUF, Tokenizer, HTTP)
.github/
└── workflows/
    ├── ci.yml                 # General PR validation
    ├── native-tests.yml       # Host C++20 tests + ASan/UBSan
    ├── android-build.yml      # Android NDK & SDK Gradle build
    ├── python-tests.yml       # Pytest & Mypy validation
    ├── security.yml           # CodeQL, Bandit, SBOM, Dependency scan
    └── release.yml            # Release artifact packaging & signing
docs/
├── benchmarks/                # Official reproducible device benchmark logs
└── security/                  # Threat model, vulnerability disclosures, SBOM
```

---

## 2. Technical Standards & Constraints

1. **Independent Verification Principle:**
   - Agent 6 never accepts another agent's self-reported success without running independent, reproducible test executions.
   - Every claim of feature completion must be accompanied by executable test evidence and build logs.
2. **Sanitizer & Memory Safety Standards:**
   - Native test suites must execute under **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)** on every pull request.
   - **ThreadSanitizer (TSan)** must be run on multithreaded engine and scheduler test targets.
   - Zero memory leaks, out-of-bounds reads/writes, or undefined behavior are tolerated in passing runs.
3. **Fuzz Testing:**
   - Maintain continuous fuzzing targets using LLVM `libFuzzer` or `AFL++` for:
     - GGUF binary header and tensor metadata parser.
     - Tokenizer byte-stream decoder (handling broken UTF-8 byte sequences).
     - Local HTTP server JSON request parsing.
4. **Security & Threat Mitigation:**
   - **Path Traversal Defense:** Validate that model loading paths cannot escape permitted application directories (e.g., rejecting `../../etc/passwd` or Android private data paths).
   - **Localhost Assertion:** Verify the local API refuses to bind to non-loopback addresses unless explicit token authentication is active.
   - **Privacy Verification:** Verify that user prompts and generated outputs are never written to disk, crash logs, or Android `logcat` by default.
5. **Supply Chain & Licensing Compliance:**
   - Continuous dependency license auditing: verify all third-party dependencies are licensed under Apache-2.0, MIT, or BSD. Strictly reject GPL/AGPL dependencies in client-distributed binaries.
   - Generate Software Bill of Materials (SBOM) in SPDX and CycloneDX formats on every release.

---

## 3. Initial Implementation Task Backlog

| Task ID | Task Description | Dependencies | Deliverable | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T6.1** | Continuous Integration Workflows | Agent 0 Contract | `.github/workflows/` | GitHub Actions for Host C++, Android NDK, Python |
| **T6.2** | Sanitizer (ASan/UBSan) Harness | T6.1 | `tests/CMakeLists.txt` | Builds and passes all native unit tests under ASan |
| **T6.3** | GGUF Parser Fuzzer Target | Agent 1 (`model_loader`) | `tests/fuzzing/` | `libFuzzer` target runs 1M iterations without crash |
| **T6.4** | Rapid Load/Unload Stress Suite | Agent 1 (`engine`) | `tests/stress/` | 100 consecutive load -> generate -> unload cycles leak-free |
| **T6.5** | Security Path Traversal Suite | Agent 1, Agent 3 | `tests/security/` | Confirms path traversal payloads fail safely with errors |
| **T6.6** | Dependency License & SBOM Audit | All Modules | `docs/security/sbom.md` | Complete inventory of dependencies; 100% license compliant |
| **T6.7** | Milestone 1 Verification Run | All Modules | `docs/benchmarks/m1.md`| Executes and records all 10 Milestone 1 criteria |

---

## 4. Key Cross-Module Interfaces

### Inputs Required from Other Modules:
- **From Agent 1 (Runtime):** Testable binaries, unit tests, and GGUF loader interfaces.
- **From Agent 3 (Android):** Signed test APKs and instrumentation test scripts.
- **From Agent 4 (Python):** Local API test suites and OpenAI compatibility assertions.
- **From Agent 5 (Hardware):** Physical device benchmark numbers and profiling logs.

### Outputs Provided to Other Modules:
- **To Agent 0 (Lead Architect):** Independent pass/fail verification reports for Milestone gates.
- **To All Agents:** CI failure triage, sanitizer backtraces, and fuzzer crash reproducers.
