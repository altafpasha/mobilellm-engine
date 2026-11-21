package ai.codesec.mobilellm

import android.content.Context
import ai.codesec.mobilellm.models.BackendType
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
     * Safely transitions the engine compute backend to [targetBackend], draining
     * in-flight inference, releasing GPU queues, and restoring CPU thread state.
     */
    suspend fun switchBackend(targetBackend: BackendType): Unit = withContext(Dispatchers.IO) {
        check(!isClosed.get()) { "MobileLLMClient is closed" }
        initMutex.withLock {
            if (engineHandle != 0L) {
                NativeBridge.nativeSwitchBackend(engineHandle, targetBackend.code)
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

        val modelHandle = NativeBridge.nativeLoadModelWithLayers(
            engineHandle = engineHandle,
            modelPath = config.modelPath,
            backend = config.preferredBackend.code,
            contextLength = config.contextLength,
            nThreads = config.nThreads,
            useMmap = config.useMmap,
            memoryBudgetMb = config.memoryBudgetMb,
            nGpuLayers = config.nGpuLayers
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
     * Resets conversation history and clears KV cache for the given session.
     */
    fun resetSession(session: ModelSession) {
        check(!isClosed.get()) { "MobileLLMClient has already been closed" }
        session.reset()
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

    data class ProcessMemorySnapshot(
        val currentRssBytes: Long = 0L,
        val peakRssBytes: Long = 0L,
        val vmSizeBytes: Long = 0L,
        val modelAllocatedBytes: Long = 0L,
        val kvAllocatedBytes: Long = 0L
    ) {
        val currentRssMb: Double get() = currentRssBytes / (1024.0 * 1024.0)
        val peakRssMb: Double get() = peakRssBytes / (1024.0 * 1024.0)
        val vmSizeMb: Double get() = vmSizeBytes / (1024.0 * 1024.0)
        val modelAllocatedMb: Double get() = modelAllocatedBytes / (1024.0 * 1024.0)
        val kvAllocatedMb: Double get() = kvAllocatedBytes / (1024.0 * 1024.0)
    }

    /**
     * Qualcomm Adreno OpenCL GPU compute diagnostic & GEMV benchmark results.
     */
    data class OpenCLDiagnosticResult(
        val success: Boolean,
        val platform: String,
        val device: String,
        val openclVersion: String,
        val driverVersion: String,
        val computeUnits: Int,
        val maxClockMhz: Int,
        val fp16Supported: Boolean,
        val dotProductSupported: Boolean,
        val subgroupShuffleSupported: Boolean,
        val mathVerification: String,
        val perGemvMs: Double,
        val bandwidthGbps: Double,
        val projectedTokPerSec: Double,
        val rawReport: String,
        val errorMessage: String? = null
    )

    /**
     * Runs an on-device Qualcomm Adreno OpenCL hardware verification and GEMV benchmark.
     */
    suspend fun runOpenCLDiagnostic(): OpenCLDiagnosticResult = withContext(Dispatchers.IO) {
        try {
            val jsonStr = NativeBridge.nativeRunOpenCLDiagnostic()
            val obj = JSONObject(jsonStr)
            val success = obj.optBoolean("success", false)
            OpenCLDiagnosticResult(
                success = success,
                platform = obj.optString("platform", "Unknown"),
                device = obj.optString("device", "Qualcomm Adreno"),
                openclVersion = obj.optString("opencl_version", "OpenCL 3.0"),
                driverVersion = obj.optString("driver_version", "Unknown"),
                computeUnits = obj.optInt("compute_units", 0),
                maxClockMhz = obj.optInt("max_clock_mhz", 0),
                fp16Supported = obj.optBoolean("fp16_supported", false),
                dotProductSupported = obj.optBoolean("dot_product_supported", false),
                subgroupShuffleSupported = obj.optBoolean("subgroup_shuffle_supported", false),
                mathVerification = obj.optString("math_verification", if (success) "PASS" else "FAIL"),
                perGemvMs = obj.optDouble("per_gemv_ms", 0.0),
                bandwidthGbps = obj.optDouble("bandwidth_gbps", 0.0),
                projectedTokPerSec = obj.optDouble("projected_tok_s", 0.0),
                rawReport = obj.optString("raw_report", ""),
                errorMessage = if (success) null else obj.optString("error", "Diagnostic failed")
            )
        } catch (e: Throwable) {
            OpenCLDiagnosticResult(
                success = false,
                platform = "Unknown",
                device = "Adreno",
                openclVersion = "N/A",
                driverVersion = "N/A",
                computeUnits = 0,
                maxClockMhz = 0,
                fp16Supported = false,
                dotProductSupported = false,
                subgroupShuffleSupported = false,
                mathVerification = "FAIL: ${e.message}",
                perGemvMs = 0.0,
                bandwidthGbps = 0.0,
                projectedTokPerSec = 0.0,
                rawReport = e.stackTraceToString(),
                errorMessage = e.message
            )
        }
    }

    /**
     * Queries process memory telemetry (RSS, Peak RSS, VmSize, Model & KV bytes).
     */
    fun getProcessMemory(): ProcessMemorySnapshot {
        val arr = NativeBridge.nativeGetProcessMemory(engineHandle) ?: return ProcessMemorySnapshot()
        return ProcessMemorySnapshot(
            currentRssBytes = arr.getOrElse(0) { 0L },
            peakRssBytes = arr.getOrElse(1) { 0L },
            vmSizeBytes = arr.getOrElse(2) { 0L },
            modelAllocatedBytes = arr.getOrElse(3) { 0L },
            kvAllocatedBytes = arr.getOrElse(4) { 0L }
        )
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
