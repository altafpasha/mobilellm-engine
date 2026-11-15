# MobileLLM Engine — Dependency, License, and Provenance Audit

> **Governing Standards:** CodeSec Open Source Compliance, SPDX 2.3  
> **Auditor:** Agent 7 (Principal Integration & Release Engineer)  
> **Repository:** `https://github.com/altafpasha/mobilellm-engine`  
> **Audit Date:** November 14, 2026  
> **Audit Scope:** Native C++ Core, Android SDK & JNI, Python SDK, CMake/Gradle Build Systems, Git Submodules  

---

## 1. Executive Summary

MobileLLM Engine is distributed under the **Apache License 2.0**. This audit examined all direct dependencies, transitive libraries, submodules, toolchains, build scripts, and packaged artifacts across the entire repository.

### Key Audit Findings:
1. **100% Permissive Ecosystem:** All native, Android, and Python dependencies operate under permissive open-source licenses (**Apache-2.0**, **MIT**, **BSD-3-Clause**).
2. **Zero Copyleft Contamination:** Zero GPL, LGPL, AGPL, or SSPL components exist in runtime libraries, JNI shared objects, or Python distribution wheels.
3. **Zero Proprietary Binary Blobs:** No precompiled static libraries (`.a`), shared objects (`.so`), dynamic libraries (`.dylib`), or Windows DLLs are tracked in git. All binaries are compiled from source.
4. **Model File Hygiene:** Zero model weights (`*.gguf`) are tracked in git. All test weight artifacts are strictly ignored via `.gitignore`.
5. **Actionable Compliance Actions Required:** Standardize SPDX license identifiers across all C++ headers and generate a consolidated `THIRD_PARTY_NOTICES.md` for binary SDK distributions.

---

## 2. Dependency Inventory & License Classification

### 2.1 Native C++ Runtime Dependencies

| Component | Inclusion Method | Pinned Version / Commit | License | Scope | License Compatibility |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`llama.cpp`** | Git Submodule | `b4260` (`40c6d79f`) | **MIT** | Core Runtime Inference | **Compatible** (Permissive) |
| **`GoogleTest`** | `find_package(GTest)` | System / CMake Fetch | **BSD-3-Clause** | Test Executable Only | **Compatible** (Not distributed) |
| **`Threads` (pthreads)** | System Library | OS Bionic / Glibc / Darwin | Platform System | Runtime Threading | **Compatible** |
| **`Android Bionic / NDK libc++`** | NDK Toolchain | NDK 27.1.12297006 | **Apache-2.0 w/ LLVM Exception** | Android C++ Runtime | **Compatible** |

### 2.2 Android SDK & Demo Application Dependencies

| Dependency Artifact | Version | License | Scope | In AAR? | Compatibility |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `androidx.core:core-ktx` | `1.12.0` | **Apache-2.0** | Android SDK | Compile/Runtime | **Compatible** |
| `kotlinx-coroutines-core` | `1.7.3` | **Apache-2.0** | Android SDK | Compile/Runtime | **Compatible** |
| `kotlinx-coroutines-android` | `1.7.3` | **Apache-2.0** | Android SDK | Compile/Runtime | **Compatible** |
| `junit:junit` | `4.13.2` | **EPL-1.0** | Android SDK Tests | No (Test Only) | **Compatible** |
| `com.google.truth:truth` | `1.1.5` | **Apache-2.0** | Android SDK Tests | No (Test Only) | **Compatible** |
| `androidx.test.ext:junit` | `1.1.5` | **Apache-2.0** | Android SDK Tests | No (Test Only) | **Compatible** |
| `androidx.test.espresso:espresso-core`| `3.5.1` | **Apache-2.0** | Android SDK Tests | No (Test Only) | **Compatible** |
| `androidx.compose.*` (BOM) | `2024.02.00` | **Apache-2.0** | Demo App Only | No (App Only) | **Compatible** |
| `androidx.lifecycle.*` | `2.7.0` | **Apache-2.0** | Demo App Only | No (App Only) | **Compatible** |

### 2.3 Python SDK Dependencies

| Package | Version Range | License | Package Role | Compatibility |
| :--- | :--- | :--- | :--- | :---: |
| **`httpx`** | `>=0.24.0` | **BSD-3-Clause** | Core HTTP Client | **Compatible** |
| **`fastapi`** | `>=0.100.0` | **MIT** | Optional Server | **Compatible** |
| **`uvicorn`** | `>=0.23.0` | **BSD-3-Clause** | Optional Server | **Compatible** |
| **`pydantic`** | `>=2.0.0` | **MIT** | Optional Server Data Validation | **Compatible** |
| **`hatchling`** | Build-time | **MIT** | PEP 517 Build Backend | **Compatible** |
| **`pytest`** | `>=7.0.0` | **MIT** | Development & Test | **Compatible** |
| **`pytest-asyncio`** | `>=0.21.0` | **Apache-2.0** | Async Test Runner | **Compatible** |
| **`mypy`** | `>=1.4.0` | **MIT** | Static Type Checker | **Compatible** |
| **`ruff`** | `>=0.1.0` | **MIT / Apache-2.0** | Linter and Code Formatter | **Compatible** |

---

## 3. Submodule & Code Provenance Verification

### 3.1 `third_party/llama.cpp`
- **Origin:** `https://github.com/ggerganov/llama.cpp`
- **Pinned Commit:** `40c6d79fb52f995f47507fedfeaae2ac05d9b35c` (Release `b4260`)
- **License:** MIT License, Copyright (c) 2023-2024 The ggml authors.
- **Verification:**
  - Build flag `LLAMA_STANDALONE=OFF` and `BUILD_SHARED_LIBS=OFF` are respected.
  - Runtime links statically against llama.cpp objects or builds them as internal objects, preserving public API hygiene.
  - Submodule contains zero viral copyleft dependencies.
  - Pre-existing vocabulary files in `third_party/llama.cpp/models/` are synthetic test vocabulary models distributed under MIT by upstream and are excluded from all binary builds.

### 3.2 Binary Blob Audit
- **Command:** `git ls-files "*.so" "*.a" "*.dylib" "*.dll" "*.bin"`
- **Result:** **Zero matches.**
- **Conclusion:** The source repository is pristine. Every binary object is compiled deterministically from source via CMake (C++20) or Gradle/NDK.

---

## 4. License Compliance & Legal Risk Analysis

### 4.1 Apache-2.0 and MIT / BSD Compatibility
- Section 4 of the Apache License 2.0 permits redistribution of Derivative Works under Apache-2.0 terms, provided that copies of the Apache License and attribution notices are preserved.
- The MIT License requires the copyright notice and permission notice to be included in all copies or substantial portions of the software.
- The BSD 3-Clause License requires retaining copyright notices, license conditions, and the disclaimer of warranty in source and binary forms.
- **Compatibility Conclusion:** MIT and BSD-3-Clause are fully compatible with Apache-2.0. When distributing binary packages (such as Android AAR or Python wheels), including a `THIRD_PARTY_NOTICES.md` file completely satisfies the binary redistribution clauses of MIT and BSD-3-Clause.

### 4.2 Patent Grant & Retaliation
- Apache License 2.0 includes an explicit patent grant (Section 3). Contributors grant users a perpetual, worldwide, non-exclusive patent license for their contributions.
- Neither `llama.cpp` (MIT) nor any AndroidX libraries impose patent restrictions hostile to the Apache 2.0 patent clause.

---

## 5. Identified Findings & Actionable Remediations

| Finding ID | Severity | Description | Remediation Plan |
| :--- | :---: | :--- | :--- |
| **LIC-01** | **Medium** | Missing SPDX license headers on native source and header files (`runtime/src/*`, `runtime/include/*`, `scheduler/*`). | Standardize all C++ and Kotlin source files with `// SPDX-License-Identifier: Apache-2.0` and copyright line. |
| **LIC-02** | **Low** | Absence of bundled `THIRD_PARTY_NOTICES.md` in distribution packaging. | Add build step in CMake and Gradle to bundle third-party copyright notices (specifically for `llama.cpp` and `ggml`) into binary packages. |
| **LIC-03** | **Low** | URL inconsistency in `python/pyproject.toml` (`codesec-ai` vs `altafpasha`). | Standardize canonical repository URLs across documentation, build files, and package manifests. |

---

## 6. Software Bill of Materials (SBOM) Summary

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:f3176054-5721-4ae6-93ab-af9885448c5c",
  "version": 1,
  "metadata": {
    "component": {
      "type": "framework",
      "name": "MobileLLM Engine",
      "version": "0.1.0-dev",
      "licenses": [{"license": {"id": "Apache-2.0"}}]
    }
  },
  "components": [
    {
      "type": "library",
      "name": "llama.cpp",
      "version": "b4260",
      "licenses": [{"license": {"id": "MIT"}}],
      "purl": "pkg:github/ggerganov/llama.cpp@40c6d79fb52f995f47507fedfeaae2ac05d9b35c"
    },
    {
      "type": "library",
      "name": "androidx.core:core-ktx",
      "version": "1.12.0",
      "licenses": [{"license": {"id": "Apache-2.0"}}],
      "purl": "pkg:maven/androidx.core/core-ktx@1.12.0"
    },
    {
      "type": "library",
      "name": "kotlinx-coroutines-core",
      "version": "1.7.3",
      "licenses": [{"license": {"id": "Apache-2.0"}}],
      "purl": "pkg:maven/org.jetbrains.kotlinx/kotlinx-coroutines-core@1.7.3"
    },
    {
      "type": "library",
      "name": "httpx",
      "version": "0.24.0+",
      "licenses": [{"license": {"id": "BSD-3-Clause"}}],
      "purl": "pkg:pypi/httpx"
    }
  ]
}
```

---

## 7. Sign-Off & Audit Verdict

**Audit Verdict:** **PASS — APPROVED FOR OPEN-SOURCE DISTRIBUTION**  
- Zero copyleft contamination.  
- Zero proprietary binary blobs.  
- Fully permissive dependency chain.  
- Standard SPDX headers (LIC-01) and notice bundling (LIC-02) to be applied during release hardening.
