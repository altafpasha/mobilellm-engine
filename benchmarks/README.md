# MobileLLM Inference Benchmarking Suite

> **Maintainer:** Agent 8 (Principal AI Inference Performance Engineer)  
> **Status:** Active  
> **Version:** `0.1.0-dev`  

---

## 1. Overview

The MobileLLM Benchmarking Suite provides reproducible, machine-readable performance measurements for local LLM inference across host development environments (macOS / Linux) and target Android mobile hardware.

It measures and isolates:
- **Cold Model Loading & RAM Footprint:** Time to parse GGUF headers and map weight tensors into virtual memory.
- **Session Initialization:** Context allocation and KV-cache preparation latency.
- **Prefill Throughput & Duration:** Prompt evaluation speed (tokens/sec).
- **Time to First Token (TTFT):** Decomposed into pure inference TTFT vs end-to-end TTFT (including queue time).
- **Token Decoding Throughput:** Steady-state generation speed (tokens/sec) excluding prefill.
- **Inter-Token Latency Percentiles:** Min, Max, Mean, P50 (Median), P90, P95, P99, and jitter standard deviation.
- **Process Memory Footprint:** Resident Set Size (RSS), Peak RSS, and virtual memory.

---

## 2. Building the Benchmark Runner

Ensure `MOBILELLM_BUILD_BENCHMARKS=ON` is configured in CMake:

```bash
# Configure pure CPU build (Metal disabled to isolate CPU/NEON performance)
cmake -B build_cpu -S . -DGGML_METAL=OFF -DMOBILELLM_ENABLE_LLAMACPP=ON -DMOBILELLM_BUILD_BENCHMARKS=ON

# Compile the benchmark executable
cmake --build build_cpu --target mobilellm_benchmark -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

---

## 3. Running Benchmarks

### 3.1 Basic Usage
```bash
./build_cpu/benchmarks/mobilellm_benchmark \
    --model models/smollm2-135m-instruct-q4_k_m.gguf \
    --threads 4 \
    --max-tokens 64 \
    --warmup 1 \
    --iterations 5 \
    --output-json benchmarks/results_smollm2_t4.json
```

### 3.2 Testing Variable Prompt Lengths
Synthesize prompts with target token counts (e.g., 256 tokens) to test prefill scaling:
```bash
./build_cpu/benchmarks/mobilellm_benchmark \
    --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --prompt-length 256 \
    --max-tokens 32 \
    --threads 4 \
    --output-json benchmarks/results_llama1b_prefill256.json
```

---

## 4. Performance Regression Comparator

The `compare.py` utility compares a candidate benchmark against a baseline and flags any regression exceeding a configured threshold:

```bash
python3 benchmarks/tools/compare.py \
    benchmarks/results_baseline.json \
    benchmarks/results_candidate.json \
    --threshold 5.0 \
    --fail-on-regression
```

### Exit Codes:
- `0`: All metrics within tolerance threshold.
- `1`: Performance regression detected exceeding the threshold (when `--fail-on-regression` is set).

---

## 5. Android Real-Device Execution via ADB

To run on a connected Android ARM64 device via ADB:

1. Push native benchmark executable and model:
   ```bash
   adb push build_android_arm64/benchmarks/mobilellm_benchmark /data/local/tmp/
   adb push models/smollm2-135m-instruct-q4_k_m.gguf /data/local/tmp/
   adb shell chmod +x /data/local/tmp/mobilellm_benchmark
   ```

2. Execute benchmark in shell:
   ```bash
   adb shell "/data/local/tmp/mobilellm_benchmark \
       --model /data/local/tmp/smollm2-135m-instruct-q4_k_m.gguf \
       --threads 4 \
       --max-tokens 64 \
       --iterations 3 \
       --output-json /data/local/tmp/android_t4.json"
   ```

3. Pull results:
   ```bash
   adb pull /data/local/tmp/android_t4.json ./benchmarks/
   ```
