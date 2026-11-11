# ADR-0003: GGUF as Initial Model Format & Model Support Strategy

- **Status:** Accepted
- **Decider:** Agent 0 (Lead Architect)
- **Date:** 2026-11-11
- **Consulted:** Agent 1 (Runtime), Agent 6 (QA & Security)

---

## Context

MobileLLM Engine requires a single-file, self-contained model format capable of storing model architecture hyper-parameters, tokenizer vocabularies, and quantized tensor weights in a memory-mappable layout suitable for mobile operating systems.

## Decision

We adopt **GGUF** (GPT-Generated Unified Format, specifications v2 and v3) as the initial primary model format.

### Architectural Rules for Model Support:
1. **Memory-Mapped Loading (`mmap`):** GGUF model files must be memory-mapped directly into the virtual address space. This allows the OS virtual memory manager to page tensor weights in and out of storage on demand, minimizing resident RAM and preventing slow full-file disk reads.
2. **Explicit Verification Matrix:** A `.gguf` file extension alone does **not** qualify a model as supported. Support is granted only after passing automated verification across:
   - Architecture validation (Llama, Qwen2, Gemma, Mistral).
   - Quantization type compatibility (Q4_0, Q4_K_M, Q8_0).
   - Context window verification.
   - License check (commercial vs. research use).
3. **Small Initial Target Models:** Focus initial physical-device testing on small, mobile-appropriate models:
   - Qwen 2.5 0.5B & 1.5B (Instruct, Q4_K_M)
   - Llama 3.2 1B & 3B (Instruct, Q4_K_M)
   - Gemma 2 / 3 2B (Q4_K_M)
4. **No Weight Redistribution:** MobileLLM Engine does not redistribute proprietary model weights. Model acquisition scripts and documentation point to original Hugging Face repositories with explicit license terms.

## Consequences

### Positive:
- Single-file portability: users can push a single `.gguf` file to `/data/local/tmp` or app storage via ADB.
- Embedded tokenizer vocabulary and chat template metadata inside the file.
- Direct alignment with the baseline inference backend (ADR-0002).

### Negative / Trade-offs:
- Parsing complex binary formats introduces attack surfaces (buffer overflows, path traversal, integer overflows in tensor offsets); mitigated through boundary validation and `libFuzzer` testing (Agent 6).

## Alternatives Considered

- **Safetensors + Separate JSON Configs:** Standard in Python/HuggingFace, but requires separate files for config, tokenizer, and weights, complicating mobile distribution and parsing.
- **ONNX:** Widely supported, but ONNX quantized LLM representations are fragmented across vendors and lack unified KV-cache state specifications.
