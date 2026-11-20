# MobileLLM GPU Correctness & Reliability Architecture

**Document ID:** `ARCH-GPU-CORRECTNESS-2026-11-20`  
**Status:** Approved Reference Standard  
**Lead Systems Architect:** Agent 11 & Agent 12  
**Target Hardware:** Qualcomm Adreno 740 / 750 (Snapdragon 8 Gen 2 / SM8650 Platform)  

---

## 1. Architectural Principles for Mobile GPU Inference

GPU acceleration in mobile LLM inference environments requires strict numerical and synchronization contracts. Unlike desktop GPUs with wide memory buses and dedicated VRAM, mobile systems employ unified memory architectures where GPU compute and CPU cache coherency compete for bandwidth and thermal budgets.

This specification details the architectural rules governing Vulkan GPU offloading in MobileLLM.

---

## 2. Kernel Wavefront & Subgroup Compatibility

### 2.1 The Subgroup Size Invariant
Mobile GPUs exhibit divergent SIMD execution widths across vendors:
- **ARM Mali:** Variable (16 or 32 threads)
- **Imagination PowerVR:** 32 threads
- **Qualcomm Adreno (A6xx, A7xx):** **64 threads**

Any compute shader that assumes a 32-thread subgroup width (`layout(local_size_x = 32)`) will trigger partial wavefront execution on Qualcomm Adreno 740/750 silicon.

### 2.2 Shared-Memory Reduction Rules
When performing reduction across workgroup threads in shared memory (`shared float tmp[...]`):
1. **Thread Indexing:** The array index in the reduction tree MUST map 1:1 to `gl_LocalInvocationID.x`.
   $$\text{tmp}[gl\_LocalInvocationID.x] \mathrel{+}= \text{tmp}[gl\_LocalInvocationID.x + s]$$
2. **Never Index by Quantization Bin:** Indexing shared memory by `tid = gl_LocalInvocationID.x / K` maps multiple physical threads to identical memory cells, causing unsynchronized write races and discarding entire quantization superblocks.
3. **Single Output Write:** Only `if (gl_LocalInvocationID.x == 0)` may commit accumulated results to destination storage buffers.

### 2.3 Vendor-Guarded Dispatch Routing
In `third_party/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
On Qualcomm Adreno hardware (`vendor_id == 0x5143`) or any GPU where `subgroup_size != 32`, vector multiplications ($N=1$ decode) must route through the 2D blocked matrix multiplication pipeline (`ggml_vk_mul_mat_q_f16` / `mul_mm.comp`). This kernel dynamically enforces matrix boundary checks and utilizes 2D subgroup tiling, completely bypassing 1D reduction race hazards.

---

## 3. Backend Transition & Synchronization Lifecycle

When transitioning between execution backends (e.g., `CPU_NEON` $\longleftrightarrow$ `VULKAN`):

```text
1. Set transitioning atomic guard (reject new generate requests).
2. Drain active in-flight requests via continuous batch scheduler.
3. Stop worker thread.
4. Call llama_synchronize(ctx) on all active contexts to flush Vulkan command queues.
5. Destroy model and context instances cleanly.
6. Re-create target backend.
7. Clamp n_gpu_layers:
   - If target == CPU_NEON: clamp n_gpu_layers = 0.
   - If target == VULKAN: restore n_gpu_layers = target layers.
8. Re-instantiate sessions and restart worker thread.
9. Clear transitioning atomic guard.
```

### 3.1 Flash Attention Restriction on Vulkan
Vulkan compute backends do not support Flash Attention on quantized weights. Therefore, when initializing a Vulkan context:
```cpp
cparams.flash_attn = false;
```
Enabling `flash_attn = true` forces the GGML scheduler to route attention calculations back to CPU, causing massive VRAM-to-RAM buffer transfers and reducing decode throughput by >97%.

---

## 4. Verification & Parity Tolerances

Parity between CPU reference and Vulkan GPU execution must adhere to documented tolerances:

| Metric | Allowable Threshold | Rationale |
| :--- | :--- | :--- |
| **Prefill Output Logits $L_2$** | $< 0.05\%$ relative delta | Accumulation differences in FP16 / FP32 dot products. |
| **Decode Step 1 Activations** | $< 0.10\%$ relative delta | Fused multiply-add rounding on Adreno ALUs. |
| **Greedy Token Sequence (32 tok)** | **100% Bit-Exact Match** | Sufficient numerical margin ensures identical argmax token selection. |
| **Memory Growth (20 transitions)** | $\le 0.0$ MB net growth | Zero buffer or context leakage across repeated switches. |
