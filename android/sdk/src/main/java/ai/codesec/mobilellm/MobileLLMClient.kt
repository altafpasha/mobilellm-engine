package ai.codesec.mobilellm

import android.content.Context
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.SessionConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Primary client for local, offline Large Language Model inference on Android.
 */
class MobileLLMClient private constructor() : Closeable {

    companion object {
        @Volatile
        private var INSTANCE: MobileLLMClient? = null

        fun getInstance(context: Context? = null): MobileLLMClient {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: MobileLLMClient().also { INSTANCE = it }
            }
        }
    }

    private var engineHandle: Long = 0
    private val isInitialized = AtomicBoolean(false)
    private val isClosed = AtomicBoolean(false)
    private val activeModels = mutableListOf<Long>()

    /**
     * Responds to Android OS memory pressure events (onTrimMemory / onLowMemory).
     */
    fun onMemoryPressure(level: Int) {
        // Can trigger native KV cache compaction or GC
        if (engineHandle != 0L) {
            // Forward trim notification if engine is live
        }
    }

    /**
     * Initializes the native runtime engine. Safe to call multiple times.
     */
    suspend fun initialize(): Unit = withContext(Dispatchers.IO) {
        if (isInitialized.compareAndSet(false, true)) {
            val handle = NativeBridge.nativeInit()
            if (handle == 0L) {
                throw MobileLLMException(
                    MobileLLMException.CODE_BACKEND_INIT_FAILED,
                    "Failed to initialize native MobileLLM engine"
                )
            }
            engineHandle = handle
        }
    }

    /**
     * Loads a GGUF model into native memory and initializes an inference session.
     *
     * @param config Model path, compute backend preference, thread count, and context length.
     * @param sessionConfig Optional context size and prefix caching settings.
     * @return [ModelSession] for generating streaming chat responses.
     */
    suspend fun loadModel(
        config: ModelConfig,
        sessionConfig: SessionConfig = SessionConfig(config.contextLength)
    ): ModelSession = withContext(Dispatchers.IO) {
        check(!isClosed.get()) { "MobileLLMClient is closed" }
        if (!isInitialized.get()) {
            initialize()
        }

        val modelFile = File(config.modelPath)
        if (!modelFile.exists()) {
            throw MobileLLMException(
                MobileLLMException.CODE_FILE_NOT_FOUND,
                "Model file not found: ${config.modelPath}"
            )
        }

        val modelHandle = NativeBridge.nativeLoadModel(
            engineHandle = engineHandle,
            modelPath = config.modelPath,
            backend = config.preferredBackend.code,
            contextLength = config.contextLength,
            nThreads = config.nThreads,
            useMmap = config.useMmap,
            memoryBudgetMb = config.memoryBudgetMb
        )

        if (modelHandle == 0L) {
            throw MobileLLMException(
                MobileLLMException.CODE_MODEL_NOT_LOADED,
                "Native loadModel returned null handle for: ${config.modelPath}"
            )
        }

        synchronized(activeModels) {
            activeModels.add(modelHandle)
        }

        val sessionHandle = NativeBridge.nativeCreateSession(
            engineHandle = engineHandle,
            modelHandle = modelHandle,
            maxContextTokens = sessionConfig.maxContextTokens,
            enablePrefixCaching = sessionConfig.enablePrefixCaching
        )

        if (sessionHandle == 0L) {
            throw MobileLLMException(
                MobileLLMException.CODE_SESSION_NOT_FOUND,
                "Failed to create native inference session"
            )
        }

        ModelSession(engineHandle, sessionHandle)
    }

    /**
     * Returns the MobileLLM Engine version string (e.g. "0.1.0-dev").
     */
    fun getVersion(): String {
        return NativeBridge.nativeGetVersion()
    }

    /**
     * Inspects hardware capabilities (CPU cores, NEON SIMD, Vulkan, RAM).
     */
    fun detectHardware(): String {
        return NativeBridge.nativeDetectHardware()
    }

    override fun close() {
        if (isClosed.compareAndSet(false, true)) {
            synchronized(activeModels) {
                for (model in activeModels) {
                    NativeBridge.nativeUnloadModel(engineHandle, model)
                }
                activeModels.clear()
            }
            if (engineHandle != 0L) {
                NativeBridge.nativeDestroy(engineHandle)
                engineHandle = 0
            }
        }
    }
}
