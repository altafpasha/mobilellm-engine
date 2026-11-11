# MobileLLM Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](file:///Users/Apple/Developer/ai-apps/MobileLLM/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android_ARM64-green.svg)](file:///Users/Apple/Developer/ai-apps/MobileLLM/ARCHITECTURE.md)
[![Status](https://img.shields.io/badge/Status-Phase_0_Foundation-orange.svg)](file:///Users/Apple/Developer/ai-apps/MobileLLM/ROADMAP.md)

**MobileLLM Engine** is an open-source, mobile-first Large Language Model (LLM) inference runtime engineered specifically for resource-constrained client environments, with Android ARM64 as its foundational target.

Inspired by core architectural innovations of high-performance server runtimes (notably **vLLM**—such as paged memory management, request queue scheduling, and continuous execution abstractions), MobileLLM Engine adapts these concepts to the strict physical realities of mobile phones: hard RAM limits, battery life, thermal throttling, and heterogeneous chipsets.

---

## Key Architecture & Principles

- **100% Offline-First:** Core inference runs entirely on-device without network permissions or cloud API requirements.
- **Mobile-Aware Memory Budgeting:** Dynamic device RAM query with an emergency `onTrimMemory` cache eviction protocol to protect against Android `lmkd` process termination.
- **Paged KV-Cache:** Divides KV memory into 16-token virtual blocks, eliminating heap fragmentation and enabling multi-turn prompt prefix caching.
- **Strict No-SQLite for Tensors:** Relational SQLite storage is reserved strictly for lightweight metadata. Tensor memory resides exclusively in native backend buffers.
- **Evidence-Based Hardware Acceleration:** ARM64 NEON SIMD serves as the golden baseline. Vulkan GPU compute is capability-probed at runtime with automatic, graceful CPU fallback.
- **Developer Ergonomics:** Provides an idiomatic Kotlin Coroutines/Flow SDK for Android and an async Python client with an optional OpenAI-compatible loopback HTTP server.

---

## Documentation Navigation

| Document | Description |
| :--- | :--- |
| [`CONTEXT.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/CONTEXT.md) | **Canonical Project Specification:** Vision, principles, scope, and non-goals |
| [`ARCHITECTURE.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/ARCHITECTURE.md) | **Comprehensive System Architecture:** Subsystems, data flows, C-ABI contracts, threat model |
| [`ROADMAP.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/ROADMAP.md) | **Master 24-Week Plan & 14-Day Schedule:** Milestone 1 acceptance gate and WBS backlog |
| [`docs/adr/`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/adr/) | **Architecture Decision Records:** Rationale, consequences, and alternatives |
| [`docs/agents/`](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/) | **Multi-Agent Operating Model:** Role charters, module ownership, and task protocols |

---

## Specialized Agent Roster

MobileLLM Engine is developed by a specialized multi-agent engineering team:

- [**Agent 0 — Lead Architect & Orchestrator**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/README.md)
- [**Agent 1 — C++ Runtime Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-1-runtime.md) (`runtime/`)
- [**Agent 2 — Memory & Scheduler Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-2-scheduler.md) (`scheduler/`)
- [**Agent 3 — Android & JNI Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-3-android.md) (`android/`)
- [**Agent 4 — Python SDK & Tooling Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-4-python.md) (`python/`, `server/`)
- [**Agent 5 — Hardware & Acceleration Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-5-hardware.md) (`backends/`)
- [**Agent 6 — QA, Benchmark & Security Engineer**](file:///Users/Apple/Developer/ai-apps/MobileLLM/docs/agents/agent-6-qa-security.md) (`tests/`, CI)

---

## Repository Monorepo Layout

```
mobilellm-engine/
├── ARCHITECTURE.md            # Master system architecture specification
├── ROADMAP.md                 # 24-week roadmap & Milestone 1 criteria
├── CONTEXT.md                 # Canonical project specification
├── LICENSE                    # Apache-2.0 license
├── VERSION                    # Version identifier (0.1.0-dev)
├── docs/
│   ├── adr/                   # Architecture Decision Records (ADR-0001 to 0007)
│   ├── agents/                # Multi-agent role specs and operational rules
│   ├── benchmarks/            # Device benchmark results & metrics logs
│   └── security/              # Threat model, fuzzing reports, SBOM
├── runtime/                   # Native C++20 engine & loader
│   └── include/mobilellm/     # Public headers (c_api.h, engine.h, types.h)
├── backends/                  # HAL: CPU/NEON, Vulkan compute, vendor NPU
├── scheduler/                 # Memory manager, paged KV-cache, request queue
├── android/                   # Kotlin SDK, NDK/JNI bridge, Compose sample app
├── python/                    # Python SDK, CLI, desktop bindings
├── server/                    # Optional loopback HTTP/SSE OpenAI server
├── tools/                     # Benchmarking, profiling, and model validation CLI
└── tests/                     # Integration, stress, and security fuzzing suites
```

---

## License

This project is licensed under the [Apache License, Version 2.0](file:///Users/Apple/Developer/ai-apps/MobileLLM/LICENSE).
