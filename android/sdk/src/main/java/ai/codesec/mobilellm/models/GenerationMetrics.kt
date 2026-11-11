package ai.codesec.mobilellm.models

/**
 * Telemetry and latency statistics measured across a complete generation run.
 *
 * @property promptTokens Number of tokens evaluated during prompt prefill.
 * @property completionTokens Number of tokens decoded during generation.
 * @property timeToFirstTokenMs Latency in milliseconds from request dispatch to emission of the first token.
 * @property tokensPerSecond Generation throughput (tokens/second) during the decode loop.
 * @property peakRssMb Peak resident set size memory in megabytes during generation.
 */
data class GenerationMetrics(
    val promptTokens: Int = 0,
    val completionTokens: Int = 0,
    val timeToFirstTokenMs: Double = 0.0,
    val tokensPerSecond: Double = 0.0,
    val peakRssMb: Double = 0.0
)

