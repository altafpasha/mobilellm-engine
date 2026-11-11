package ai.codesec.mobilellm

/**
 * Structured exception thrown when native MobileLLM operations fail.
 *
 * @property errorCode Native StatusCode identifying the failure mode.
 */
class MobileLLMException(
    val errorCode: Int,
    message: String,
    cause: Throwable? = null
) : RuntimeException("[$errorCode] $message", cause) {

    companion object {
        const val CODE_OK = 0
        const val CODE_INVALID_ARGUMENT = 1
        const val CODE_FILE_NOT_FOUND = 2
        const val CODE_UNSUPPORTED_FORMAT = 3
        const val CODE_OUT_OF_MEMORY = 4
        const val CODE_BACKEND_INIT_FAILED = 5
        const val CODE_BACKEND_EXEC_ERROR = 6
        const val CODE_SESSION_NOT_FOUND = 7
        const val CODE_MODEL_NOT_LOADED = 8
        const val CODE_CANCELLED = 9
        const val CODE_CONTEXT_EXCEEDED = 10
        const val CODE_TIMEOUT = 11
        const val CODE_INTERNAL_ERROR = 99
    }
}

