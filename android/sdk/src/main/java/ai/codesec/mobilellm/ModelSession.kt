package ai.codesec.mobilellm

import ai.codesec.mobilellm.models.GenerationMetrics
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.mobilellm.models.TokenChunk
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

/**
 * An active conversational session tied to a loaded model in native memory.
 */
class ModelSession internal constructor(
    val engineHandle: Long,
    val sessionHandle: Long,
    val modelHandle: Long = 0L,
    private val client: MobileLLMClient? = null
) : Closeable {

    private val isClosed = AtomicBoolean(false)

    /**
     * Executes streaming text generation for the given prompt.
     *
     * Returns a cold [Flow] that emits tokens in real-time as they are decoded by the native engine.
     * Cancelling the collector coroutine automatically invokes cooperative native cancellation.
     *
     * @param prompt User instruction or conversation history to evaluate.
     * @param params Hyperparameters controlling sampling temperature, Top-P, max tokens, etc.
     * @param onMetrics Optional callback receiving final telemetry upon generation completion.
     */
    fun generate(
        prompt: String,
        params: SamplingParams = SamplingParams(),
        onMetrics: ((GenerationMetrics) -> Unit)? = null
    ): Flow<TokenChunk> = callbackFlow {
        check(!isClosed.get()) { "ModelSession has already been closed" }
        require(prompt.isNotBlank()) { "Prompt must not be blank" }

        var activeGenHandle: Long = 0
        val startTime = System.currentTimeMillis()

        val nativeCb = object : NativeCallback {
            override fun onToken(text: String, tokenId: Int, isSpecial: Boolean, logprob: Float): Boolean {
                val chunk = TokenChunk(text, tokenId, isSpecial, logprob)
                val sendResult = trySend(chunk)
                // If the channel was closed by the collector, signal native cancellation
                return sendResult.isSuccess
            }

            override fun onComplete(
                code: Int,
                message: String,
                promptTokens: Int,
                completionTokens: Int,
                ttftMs: Double,
                tps: Double,
                peakRssMb: Double
            ) {
                val durationMs = (System.currentTimeMillis() - startTime).toDouble()
                val metrics = GenerationMetrics(
                    promptTokens = promptTokens,
                    completionTokens = completionTokens,
                    timeToFirstTokenMs = ttftMs,
                    tokensPerSecond = tps,
                    peakRssMb = peakRssMb,
                    totalDurationMs = durationMs
                )
                onMetrics?.invoke(metrics)

                if (code == 0 || code == MobileLLMException.CODE_CANCELLED) {
                    channel.close()
                } else {
                    channel.close(MobileLLMException(code, message))
                }
            }
        }

        activeGenHandle = NativeBridge.nativeGenerate(
            engineHandle = engineHandle,
            sessionHandle = sessionHandle,
            prompt = prompt,
            temperature = params.temperature,
            topP = params.topP,
            topK = params.topK,
            minP = params.minP,
            repetitionPenalty = params.repetitionPenalty,
            maxTokens = params.maxTokens,
            callback = nativeCb
        )

        awaitClose {
            if (activeGenHandle != 0L) {
                NativeBridge.nativeCancel(engineHandle, activeGenHandle)
            }
        }
    }.buffer(kotlinx.coroutines.channels.Channel.UNLIMITED)

    override fun close() {
        if (isClosed.compareAndSet(false, true)) {
            NativeBridge.nativeDestroySession(engineHandle, sessionHandle)
            if (modelHandle != 0L) {
                client?.unloadModel(modelHandle)
            }
        }
    }
}

