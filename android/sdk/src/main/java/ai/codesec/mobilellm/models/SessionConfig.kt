package ai.codesec.mobilellm.models

/**
 * Configuration options for an active generation session.
 *
 * @property maxContextTokens Maximum tokens preserved in the KV-cache for this session.
 * @property enablePrefixCaching Whether to enable virtual paged prefix cache sharing for prompt prefixes.
 */
data class SessionConfig(
    val maxContextTokens: Int = 2048,
    val enablePrefixCaching: Boolean = true
) {
    init {
        require(maxContextTokens > 0) { "maxContextTokens must be positive" }
    }
}
