package ai.codesec.mobilellm

import android.content.Context
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.ModelMetadata
import ai.codesec.mobilellm.models.SessionConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
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
    private val initMutex = Mutex()
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
        initMutex.withLock {
            if (engineHandle == 0L && !isClosed.get()) {
                val handle = NativeBridge.nativeInit()
                if (handle == 0L) {
                    throw MobileLLMException(
                        MobileLLMException.CODE_BACKEND_INIT_FAILED,
                        "Failed to initialize native MobileLLM engine"
                    )
                }
                engineHandle = handle
                isInitialized.set(true)
            }
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
        if (engineHandle == 0L) {
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

        val sessionHandle = try {
            val handle = NativeBridge.nativeCreateSession(
                engineHandle = engineHandle,
                modelHandle = modelHandle,
                maxContextTokens = sessionConfig.maxContextTokens,
                enablePrefixCaching = sessionConfig.enablePrefixCaching
            )
            if (handle == 0L) {
                throw MobileLLMException(
                    MobileLLMException.CODE_SESSION_NOT_FOUND,
                    "Failed to create native inference session"
                )
            }
            handle
        } catch (e: Throwable) {
            synchronized(activeModels) {
                activeModels.remove(modelHandle)
            }
            NativeBridge.nativeUnloadModel(engineHandle, modelHandle)
            throw e
        }

        ModelSession(engineHandle, sessionHandle, modelHandle, this@MobileLLMClient)
    }

    /**
     * Unloads a specific model by handle from native memory, releasing its weights and memory budget.
     */
    fun unloadModel(modelHandle: Long) {
        synchronized(activeModels) {
            if (activeModels.remove(modelHandle)) {
                NativeBridge.nativeUnloadModel(engineHandle, modelHandle)
            }
        }
    }

    /**
     * Unloads all loaded models from native memory, freeing all reserved weights and memory budget.
     */
    fun unloadAllModels() {
        synchronized(activeModels) {
            for (model in activeModels) {
                NativeBridge.nativeUnloadModel(engineHandle, model)
            }
            activeModels.clear()
        }
    }

    /**
     * Inspects a GGUF model file headers and metadata without loading weights into memory.
     *
     * @param modelFile File object pointing to local .gguf file.
     * @param computeHash When true, computes SHA-256 checksum of the file.
     * @return [ModelMetadata] containing architecture, quantization, context size, etc.
     */
    suspend fun inspectModel(
        modelFile: File,
        computeHash: Boolean = false
    ): ModelMetadata = withContext(Dispatchers.IO) {
        if (!modelFile.exists()) {
            throw MobileLLMException(
                MobileLLMException.CODE_FILE_NOT_FOUND,
                "Model file not found: ${modelFile.absolutePath}"
            )
        }

        val jsonStr = try {
            NativeBridge.nativeInspectModel(modelFile.absolutePath)
        } catch (e: Exception) {
            throw MobileLLMException(
                MobileLLMException.CODE_UNSUPPORTED_FORMAT,
                "Failed to parse GGUF headers: ${e.message}"
            )
        }

        val obj = JSONObject(jsonStr)
        val sha256 = if (computeHash) {
            try {
                val md = MessageDigest.getInstance("SHA-256")
                FileInputStream(modelFile).use { fis ->
                    val buffer = ByteArray(64 * 1024)
                    var read: Int
                    while (fis.read(buffer).also { read = it } != -1) {
                        md.update(buffer, 0, read)
                    }
                }
                md.digest().joinToString("") { "%02x".format(it) }
            } catch (_: Exception) {
                ""
            }
        } else {
            ""
        }

        ModelMetadata(
            filepath = modelFile.absolutePath,
            architecture = obj.optString("architecture", "unknown"),
            quantization = obj.optString("quantization", "unknown"),
            contextLength = obj.optInt("context_length", 2048),
            embeddingLength = obj.optInt("embedding_length", 0),
            blockCount = obj.optInt("block_count", 0),
            headCount = obj.optInt("head_count", 0),
            tensorCount = obj.optLong("tensor_count", 0L),
            fileSizeBytes = obj.optLong("file_size_bytes", modelFile.length()),
            sha256 = sha256,
            isValid = obj.optBoolean("valid", true)
        )
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
