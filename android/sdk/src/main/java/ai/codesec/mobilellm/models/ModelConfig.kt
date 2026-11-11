package ai.codesec.mobilellm.models

/**
 * Configuration options for loading a GGUF model into native memory.
 *
 * @property modelPath Absolute file path to the local .gguf model file.
 * @property preferredBackend Compute target preference (default: AUTO).
 * @property contextLength Maximum context length in tokens (default: 2048).
 * @property nThreads Number of CPU compute threads to allocate (default: 4).
 * @property useMmap Whether to memory-map model weights directly into virtual memory (default: true).
 * @property memoryBudgetMb Memory limit ceiling in megabytes (0 = auto-detect based on device RAM).
 */
data class ModelConfig(
    val modelPath: String,
    val preferredBackend: BackendType = BackendType.AUTO,
    val contextLength: Int = 2048,
    val nThreads: Int = 4,
    val useMmap: Boolean = true,
    val memoryBudgetMb: Float = 0f
) {
    init {
        require(modelPath.isNotBlank()) { "modelPath must not be blank" }
        require(contextLength > 0) { "contextLength must be positive" }
        require(nThreads > 0) { "nThreads must be positive" }
    }
}
