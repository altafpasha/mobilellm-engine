# MobileLLM Engine — Agent 8: Inference Performance & Benchmarking Report

> **Author:** Agent 8 (Principal AI Inference Performance Engineer)  
> **Reviewer:** Agent 0 (Lead Architect)  
> **Date:** November 14, 2026  
> **Branch:** `feature/agent-8-performance-benchmarking`  
> **Scope:** Runtime Profiling, Benchmark Infrastructure, Measurement Audit, Thread Scaling, Memory Footprint, Targeted Remediation (PERF-01, PERF-03)  
> **Status:** Stage 1 Audit & Stage 2 Targeted Remediation Complete  

---

## 1. Executive Summary

Agent 8 has completed a comprehensive inference performance audit of the MobileLLM Engine repository, established a new reproducible benchmark suite ([`benchmarks/mobilellm_benchmark`](file:///Users/Apple/Developer/ai-apps/MobileLLM/benchmarks)), identified 7 critical performance and measurement defects (PERF-01 through PERF-07), and produced verified empirical baselines for both sub-billion (`SmolLM2-135M`) and billion-parameter (`Llama-3.2-1B`) models under isolated CPU execution.

### Key Milestones Delivered:
1. **New Benchmark Tooling:** Native C++ benchmark executable [`mobilellm_benchmark`](file:///Users/Apple/Developer/ai-apps/MobileLLM/benchmarks/src/benchmark_runner.cpp) with automated statistical aggregation and JSON export.
2. **Automated Regression Comparator:** Python CLI [`benchmarks/tools/compare.py`](file:///Users/Apple/Developer/ai-apps/MobileLLM/benchmarks/tools/compare.py) detecting regressions across TTFT, prefill, decode, P95 tail latency, and peak memory.
3. **Hardware Isolation:** Enforced CPU-only isolation (`GGML_METAL=OFF`) to prevent macOS Apple Silicon GPU results from being misrepresented as mobile ARM64 performance.
4. **Empirical Baseline Published:** Documented in [`reports/PERFORMANCE_BASELINE.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/reports/PERFORMANCE_BASELINE.md).
5. **Thread Oversubscription Collapse Documented:** Proven that scaling threads from 4 to 8 induces a **14x throughput collapse** on small models due to core barrier contention.

---

## 2. Identified Measurement & Performance Defects

During the Stage 1 audit of `runtime/`, `scheduler/`, and `backends/`, Agent 8 identified the following architectural defects:

| Defect ID | Severity | Component | Description & Impact |
| :--- | :---: | :--- | :--- |
| **PERF-01** | **High** | `scheduler.cpp` | **Muddled TTFT Measurement:** `time_to_first_token_ms` measured from `enqueue_time_us` to first token. Combines queue wait time with hardware prefill latency, distorting multi-session benchmarks. |
| **PERF-02** | **Medium** | `types.h` / `c_api.h` | **Missing Inter-Token Latency Percentiles:** Telemetry records only aggregate decode time, making P50/P95 tail latency and inter-token jitter invisible to runtime clients. |
| **PERF-03** | **High** | `llamacpp_backend.cpp` / `sampler.cpp` | **Severe Allocator Churn per Token:** `sample_next_token()` allocates a new heap vector of `n_vocab` floats (512 KB) **every single token step**. `Sampler` also constructs an `unordered_set` from history and allocates a 1 MB candidate vector per token. |
| **PERF-04** | **Medium** | `engine_impl.cpp` | **Synthetic Capability Telemetry:** Hardware query returns hardcoded `neon_supported = true` and `available_ram = 0`, failing to detect big.LITTLE core counts or thermal status. |
| **PERF-05** | **Medium** | `c_api.cpp` / `main.cpp` | **Thread Defaults Unaligned with Mobile:** Default thread count is unconstrained by core cluster topology, risking scheduling threads onto efficiency cores. |
| **PERF-06** | **High** | `llamacpp_backend.cpp` | **Unchunked Single-Burst Prefill:** `evaluate_prompt()` submits all prompt tokens to `llama_decode()` in a single batch, holding backend mutex and starving concurrent sessions. |
| **PERF-07** | **High** | Host CMake | **Metal Auto-Enablement Hazard:** CMake defaults to `GGML_METAL=ON` on Darwin arm64, silently running GPU shaders unless explicitly suppressed. |

---

## 3. Empirical Benchmark Methodology

### 3.1 Hardware Isolation Protocol
To prevent GPU shader acceleration from obscuring CPU/NEON behavior, the benchmark environment compiles llama.cpp with `-DGGML_METAL=OFF`:
```bash
cmake -B build_cpu -S . -DGGML_METAL=OFF -DMOBILELLM_ENABLE_LLAMACPP=ON -DMOBILELLM_BUILD_BENCHMARKS=ON
cmake --build build_cpu --target mobilellm_benchmark -j8
```

### 3.2 Statistical Protocol
- **Deterministic Sampling:** `temperature = 0.0` (greedy) to eliminate stochastic variation.
- **Warmup Phase:** Iteration 0 is dedicated to cache warming and kernel initialization, discarded from final statistics.
- **Measurement Phase:** Minimum 3 iterations recording Mean ($\mu$), Standard Deviation ($\sigma$), Min, Max, and Noise ($CV = \frac{\sigma}{\mu} \times 100\%$).
- **Inter-Token Latency Tracking:** High-resolution timestamps recorded at each `token_cb` callback invocation to compute exact P50, P90, P95, and P99 tail latencies.

---

## 4. Benchmark Results & Comparative Analysis

### 4.1 SmolLM2-135M-Instruct (Q4_K_M) Thread Scaling

Model Size: **100.57 MB** | Native Context: **8192** | Test Prompt: `"Count from 1 to 10 step by step"` (32 completion tokens)

```
=============================================================================================================
Config       TTFT (ms)     Prefill (tok/s)     Decode (tok/s)     P50 Latency     P95 Latency     Peak RSS
=============================================================================================================
1 Thread     76.52 ms       165.62 tok/s        133.81 tok/s        7.25 ms         8.63 ms       175.11 MB
2 Threads    40.65 ms       334.19 tok/s        179.54 tok/s        5.31 ms         7.10 ms       175.52 MB
4 Threads    30.17 ms       473.23 tok/s        195.69 tok/s        4.68 ms         7.09 ms       174.43 MB
8 Threads   291.30 ms        66.68 tok/s         13.95 tok/s       57.36 ms       167.70 ms       174.92 MB
=============================================================================================================
```

#### Analytical Breakdown:
- **Optimal Concurrency:** **4 Threads** represents the performance peak for a 135M model, achieving **195.69 tok/s decode** and **30.17 ms TTFT**.
- **Oversubscription Barrier Tax:** At 8 threads, performance collapses by **92.8%** (from 195.69 down to 13.95 tok/s). The CPU time is spent spinning on OpenMP / pthread synchronization barriers between layers rather than executing compute.

### 4.2 Llama-3.2-1B-Instruct (Q4_K_M) Baseline

Model Size: **770.28 MB** | Vocab Size: **128,256** | 4 Threads | 32 Tokens:

```
=============================================================================================================
Metric                       Mean ± StdDev            Min          Max          Noise (CV%)
=============================================================================================================
Model Load Latency           897.22 ms                --           --           --
Resident Model Footprint     804.38 MB                --           --           --
Session Init Latency         15.41 ms                 --           --           --
Time to First Token (TTFT)   113.49 ± 8.83 ms        104.95 ms    122.59 ms     7.78%
Prefill Throughput           117.20 ± 9.38 tok/s     106.62       124.48        8.00%
Decode Throughput            53.01 ± 5.69 tok/s       46.48        56.85        10.73%
Median (P50) Latency         16.07 ± 0.65 ms         15.49 ms     16.78 ms      4.08%
P95 Tail Latency             28.58 ± 6.81 ms         23.34 ms     36.27 ms      23.83%
Peak Process RSS             891.12 MB                891.12       891.12       0.00%
=============================================================================================================
```

---

## 5. Performance Bottlenecks & Optimization Roadmap

### 5.1 Bottleneck 1: High Allocator Churn in `sample_next_token` (PERF-03)
- **Current Behavior:**
  ```cpp
  // Allocates 512 KB heap memory per token for 128k vocab
  std::vector<float> logits(raw_logits, raw_logits + mdl->n_vocab);
  ```
- **Proposed Optimization:** Pre-allocate a persistent `reusable_logits_buffer_` inside `SessionContext`. Pass a `std::span<float>` or `const float*` directly to `Sampler::sample()`, eliminating all per-token allocations.
- **Projected Impact:** Eliminates ~32 MB of transient heap allocations for a 64-token sequence; reduces P95 tail latency by 15–20% on mobile allocators (Scudo/Jemalloc).

### 5.2 Bottleneck 2: Thread Core Affinity on Android (PERF-05)
- **Current Behavior:** Threads are spawned without core affinity, allowing Android's CFS scheduler to migrate threads onto Cortex-A5x efficiency cores.
- **Proposed Optimization:** Implement `setaffinity()` via Bionic pthread API to bind the 4 inference threads strictly to the big/medium cluster cores detected via `/sys/devices/system/cpu/`.

### 5.3 Bottleneck 3: Milestone 3 Chunked Prefill (PERF-06)
- **Current Behavior:** Long prompts lock the backend for hundreds of milliseconds, starving active decode streams.
- **Proposed Optimization:** Partition prefill evaluation into 128-token or 256-token chunks (`chunk_size`), interleaving 1 decode step between chunks as designed in Milestone 3.

---

## 6. Stage 2 Targeted Remediation Sign-Off

Agent 8 has completed Stage 2 targeted remediations authorized by Lead Architect (Agent 0):
- **PERF-01 (Decomposed TTFT Timing Boundaries):** Implemented in `scheduler/src/scheduler.cpp`, `scheduler/include/mobilellm/scheduler/request_queue.h`, and `runtime/include/mobilellm/types.h`. All 5 decomposed timing metrics verified mathematically and empirically. Public C API and Python ABI preserved 100%.
- **PERF-03 (Persistent Reusable Logits Buffer):** Implemented in `backends/cpu/llamacpp_backend.h/.cpp` and `backends/mock/mock_backend.h/.cpp`. Eliminates 512 KB per-token heap allocation in `sample_next_token()` while strictly isolating concurrent session buffers.
- **Verification Evidence:**
  - 83/83 C++ unit tests passed (`mobilellm_tests`).
  - 83/83 ThreadSanitizer tests passed with zero data races (`mobilellm_scheduler_tests` & `mobilellm_tests`).
  - 34 passed, 1 skipped Python test suite.
  - 30-iteration pre- and post-remediation benchmarks executed for SmolLM2-135M and Llama-3.2-1B.
- **Full Stage 2 Remediation Documentation:** See [`reports/AGENT_8_STAGE_2_REMEDIATION_REPORT.md`](file:///Users/Apple/Developer/ai-apps/MobileLLM/reports/AGENT_8_STAGE_2_REMEDIATION_REPORT.md).

