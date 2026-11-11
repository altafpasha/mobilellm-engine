package ai.codesec.mobilellm.models

/**
 * Hyperparameters controlling token sampling from raw logit distributions.
 *
 * @property temperature Randomness scale (0.0 = greedy argmax, > 0.0 = stochastic sampling).
 * @property topP Nucleus sampling cumulative probability threshold.
 * @property topK Top-K candidate count filter.
 * @property minP Minimum probability relative to the maximum token probability.
 * @property repetitionPenalty Penalty applied against recently generated tokens (1.0 = disabled).
 * @property maxTokens Maximum number of completion tokens to generate.
 * @property stopSequences List of string sequences that halt generation upon occurrence.
 */
data class SamplingParams(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val minP: Float = 0.05f,
    val repetitionPenalty: Float = 1.1f,
    val maxTokens: Int = 512,
    val stopSequences: List<String> = emptyList()
) {
    init {
        require(temperature >= 0f) { "temperature must be non-negative" }
        require(topP in 0f..1f) { "topP must be between 0.0 and 1.0" }
        require(topK >= 0) { "topK must be non-negative" }
        require(minP in 0f..1f) { "minP must be between 0.0 and 1.0" }
        require(repetitionPenalty >= 1f) { "repetitionPenalty must be >= 1.0" }
        require(maxTokens > 0) { "maxTokens must be positive" }
    }
}
