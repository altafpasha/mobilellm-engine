package ai.codesec.mobilellm.models

/**
 * Structural metadata parsed from GGUF model headers without loading weights.
 *
 * @property filepath Absolute file path of the inspected model.
 * @property architecture Neural network architecture (e.g. "llama", "smollm", "qwen2").
 * @property quantization Weight quantization format (e.g. "Q4_K_M", "Q8_0", "F16").
 * @property contextLength Maximum context length in tokens configured in model metadata.
 * @property embeddingLength Dimensionality of embedding layers.
 * @property blockCount Number of transformer layers / blocks.
 * @property headCount Number of attention heads.
 * @property tensorCount Total count of tensors stored in the GGUF file.
 * @property fileSizeBytes Physical size of the model file on disk in bytes.
 * @property sha256 Optional SHA-256 hash digest of the file for integrity verification.
 * @property isValid Whether the GGUF file format and headers are verified.
 */
data class ModelMetadata(
    val filepath: String,
    val architecture: String = "unknown",
    val quantization: String = "unknown",
    val contextLength: Int = 2048,
    val embeddingLength: Int = 0,
    val blockCount: Int = 0,
    val headCount: Int = 0,
    val tensorCount: Long = 0L,
    val fileSizeBytes: Long = 0L,
    val sha256: String = "",
    val isValid: Boolean = true
)
