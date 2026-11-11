package ai.codesec.mobilellm

/**
 * Internal callback interface invoked directly from the native C++ decode loop via JNI.
 */
interface NativeCallback {
    /**
     * Emits an individual token chunk. Return false to request cooperative cancellation.
     */
    fun onToken(text: String, tokenId: Int, isSpecial: Boolean, logprob: Float): Boolean

    /**
     * Signals completion of generation with final status and telemetry metrics.
     */
    fun onComplete(
        code: Int,
        message: String,
        promptTokens: Int,
        completionTokens: Int,
        ttftMs: Double,
        tps: Double,
        peakRssMb: Double
    )
}

