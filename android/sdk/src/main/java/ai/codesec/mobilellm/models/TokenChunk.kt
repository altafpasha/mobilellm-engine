package ai.codesec.mobilellm.models

/**
 * Individual token piece emitted in real-time during autoregressive generation.
 *
 * @property text The decoded UTF-8 string piece corresponding to this token.
 * @property tokenId The integer ID of the token in the model vocabulary.
 * @property isSpecial True if this token is a special control or delimiter token (e.g. BOS, EOS, EOT).
 * @property logprob Log probability associated with this token choice (0.0 if uncalculated).
 */
data class TokenChunk(
    val text: String,
    val tokenId: Int,
    val isSpecial: Boolean = false,
    val logprob: Float = 0f
)

