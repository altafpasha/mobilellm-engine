package ai.codesec.mobilellm

internal object NativeBridge {

    init {
        try {
            try { System.loadLibrary("c++_shared") } catch (_: UnsatisfiedLinkError) {}
            try { System.loadLibrary("ggml-base") } catch (_: UnsatisfiedLinkError) {}
            try { System.loadLibrary("ggml") } catch (_: UnsatisfiedLinkError) {}
            try { System.loadLibrary("ggml-cpu") } catch (_: UnsatisfiedLinkError) {}
            try { System.loadLibrary("llama") } catch (_: UnsatisfiedLinkError) {}
            try { System.loadLibrary("mobilellm") } catch (_: UnsatisfiedLinkError) {}
            System.loadLibrary("mobilellm_jni")
        } catch (e: UnsatisfiedLinkError) {
            // When running in host desktop unit tests without native .so
            System.err.println("MobileLLM: native library libmobilellm_jni.so not loaded: ${e.message}")
        }
    }

    @JvmStatic
    external fun nativeInit(): Long

    @JvmStatic
    external fun nativeDestroy(engineHandle: Long)

    @JvmStatic
    external fun nativeLoadModel(
        engineHandle: Long,
        modelPath: String,
        backend: Int,
        contextLength: Int,
        nThreads: Int,
        useMmap: Boolean,
        memoryBudgetMb: Float
    ): Long

    @JvmStatic
    external fun nativeUnloadModel(engineHandle: Long, modelHandle: Long)

    @JvmStatic
    external fun nativeCreateSession(
        engineHandle: Long,
        modelHandle: Long,
        maxContextTokens: Int,
        enablePrefixCaching: Boolean
    ): Long

    @JvmStatic
    external fun nativeDestroySession(engineHandle: Long, sessionHandle: Long)

    @JvmStatic
    external fun nativeGenerate(
        engineHandle: Long,
        sessionHandle: Long,
        prompt: String,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float,
        maxTokens: Int,
        callback: NativeCallback
    ): Long

    @JvmStatic
    external fun nativeCancel(engineHandle: Long, genHandle: Long)

    @JvmStatic
    external fun nativeGetVersion(): String

    @JvmStatic
    external fun nativeDetectHardware(): String

    @JvmStatic
    external fun nativeInspectModel(modelPath: String): String

    @JvmStatic
    external fun nativeGetProcessMemory(engineHandle: Long): LongArray?

    @JvmStatic
    external fun nativeGetSessionContextTelemetry(engineHandle: Long, sessionHandle: Long): IntArray?

    @JvmStatic
    external fun nativeSetSessionThreads(engineHandle: Long, sessionHandle: Long, nThreads: Int): Boolean
}

