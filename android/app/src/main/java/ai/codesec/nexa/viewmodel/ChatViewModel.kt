package ai.codesec.nexa.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.codesec.mobilellm.MobileLLMClient
import ai.codesec.mobilellm.ModelSession
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.GenerationMetrics
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.ModelMetadata
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.mobilellm.models.SessionConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

enum class MessageSender {
    USER,
    ASSISTANT,
    SYSTEM
}

enum class NexaScreen {
    CHAT,
    MODELS,
    DASHBOARD
}

data class MediaAttachment(
    val id: String = UUID.randomUUID().toString(),
    val uri: String,
    val mimeType: String,
    val name: String,
    val sizeBytes: Long = 0L,
    val isAudio: Boolean = false
)

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val sender: MessageSender,
    val text: String,
    val attachments: List<MediaAttachment> = emptyList(),
    val metrics: GenerationMetrics? = null,
    val isStreaming: Boolean = false,
    val timestamp: Long = System.currentTimeMillis()
)

data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val modelName: String?,
    val messages: List<ChatMessage> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis()
)

enum class DownloadStatus {
    QUEUED, DOWNLOADING, PAUSED, COMPLETED, FAILED, CANCELLED
}

data class DownloadItem(
    val id: String,
    val modelName: String,
    val downloadUrl: String,
    val targetFileName: String,
    val totalBytes: Long,
    val downloadedBytes: Long = 0L,
    val speedBytesPerSec: Double = 0.0,
    val progress: Float = 0.0f,
    val status: DownloadStatus = DownloadStatus.QUEUED,
    val errorMessage: String? = null
)

data class DiscoverModelItem(
    val id: String,
    val name: String,
    val architecture: String,
    val parameters: String,
    val quantization: String,
    val downloadSizeMb: Long,
    val ramRequiredMb: Long,
    val contextLength: Int,
    val description: String,
    val downloadUrl: String,
    val filename: String,
    val category: String,
    val license: String = "Apache-2.0",
    val recommendedDevice: String = "Snapdragon 8 Gen 2 / 8 Gen 3"
)

val DEFAULT_DISCOVER_CATALOG = listOf(
    DiscoverModelItem(
        id = "smollm2-135m",
        name = "SmolLM2-135M-Instruct",
        architecture = "Llama",
        parameters = "135M",
        quantization = "Q4_K_M",
        downloadSizeMb = 98L,
        ramRequiredMb = 350L,
        contextLength = 2048,
        description = "Ultra-compact mobile language model. Exceptional token generation speed (150+ tok/s on ARM NEON).",
        downloadUrl = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf",
        filename = "smollm2-135m-instruct-q4_k_m.gguf",
        category = "Ultra-Fast",
        license = "Apache-2.0",
        recommendedDevice = "Any ARM64 Device"
    ),
    DiscoverModelItem(
        id = "qwen2.5-0.5b",
        name = "Qwen2.5-0.5B-Instruct",
        architecture = "Qwen2",
        parameters = "0.5B",
        quantization = "Q4_K_M",
        downloadSizeMb = 398L,
        ramRequiredMb = 850L,
        contextLength = 4096,
        description = "Alibaba Cloud's ultra-efficient sub-1B model with great instruction following and multi-turn capabilities.",
        downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        category = "Balanced",
        license = "Qwen License",
        recommendedDevice = "Snapdragon 7/8 Series"
    ),
    DiscoverModelItem(
        id = "qwen2.5-1.5b",
        name = "Qwen2.5-1.5B-Instruct",
        architecture = "Qwen2",
        parameters = "1.5B",
        quantization = "Q4_K_M",
        downloadSizeMb = 1117L,
        ramRequiredMb = 1850L,
        contextLength = 4096,
        description = "Balanced coding, reasoning, and multi-lingual instruction model. Excellent benchmark accuracy on mobile.",
        downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        category = "Reasoning",
        license = "Qwen License",
        recommendedDevice = "Snapdragon 8 Gen 2 / 8 Gen 3"
    ),
    DiscoverModelItem(
        id = "llama-3.2-1b",
        name = "Llama-3.2-1B-Instruct",
        architecture = "Llama",
        parameters = "1.2B",
        quantization = "Q4_K_M",
        downloadSizeMb = 810L,
        ramRequiredMb = 1400L,
        contextLength = 4096,
        description = "Meta's official lightweight edge model designed for high-throughput mobile summarization and chat.",
        downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        category = "Balanced",
        license = "Llama 3.2 Community",
        recommendedDevice = "Snapdragon 8 Gen 1/2/3"
    ),
    DiscoverModelItem(
        id = "gemma-2-2b",
        name = "Gemma-2-2B-IT",
        architecture = "Gemma2",
        parameters = "2.2B",
        quantization = "Q4_K_M",
        downloadSizeMb = 1640L,
        ramRequiredMb = 2400L,
        contextLength = 4096,
        description = "Google DeepMind's state-of-the-art small model with sliding window attention and exceptional reasoning.",
        downloadUrl = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        filename = "gemma-2-2b-it-Q4_K_M.gguf",
        category = "Reasoning",
        license = "Gemma Terms of Use",
        recommendedDevice = "Snapdragon 8 Gen 2 / 12GB+ RAM"
    ),
    DiscoverModelItem(
        id = "deepseek-r1-1.5b",
        name = "DeepSeek-R1-Distill-1.5B",
        architecture = "Qwen2",
        parameters = "1.5B",
        quantization = "Q4_K_M",
        downloadSizeMb = 1120L,
        ramRequiredMb = 1850L,
        contextLength = 4096,
        description = "DeepSeek R1 reasoning traces distilled into a compact 1.5B model for step-by-step mathematical reasoning.",
        downloadUrl = "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        filename = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        category = "Reasoning",
        license = "MIT",
        recommendedDevice = "Snapdragon 8 Gen 2 / 8 Gen 3"
    )
)

data class BenchmarkRun(
    val threads: Int,
    val promptTokens: Int,
    val completionTokens: Int,
    val promptTokensPerSecond: Double,
    val tokensPerSecond: Double,
    val timeToFirstTokenMs: Double,
    val totalDurationMs: Double,
    val peakRssMb: Double,
    val thermalStatusName: String,
    val timestamp: Long = System.currentTimeMillis()
)

data class ChatUiState(
    val currentScreen: NexaScreen = NexaScreen.CHAT,
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val isModelLoaded: Boolean = false,
    val loadedModelName: String? = null,
    val availableModels: List<File> = emptyList(),
    val modelMetadataMap: Map<String, ModelMetadata> = emptyMap(),
    val samplingParams: SamplingParams = SamplingParams(),
    val preferredBackend: BackendType = BackendType.CPU_NEON,
    val gpuLayers: Int = -1,
    val threadCount: Int = run {
        val totalCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        if (totalCores >= 8) 4 else minOf(4, totalCores)
    },
    val engineVersion: String = "0.1.0-dev",
    val hardwareInfo: String = "Detecting...",
    val lastMetrics: GenerationMetrics? = null,
    val generationDiagnostics: String = "",
    val systemTotalRamMb: Long = 0L,
    val systemAvailRamMb: Long = 0L,
    val isLowMemory: Boolean = false,
    val nativeRssMb: Double = 0.0,
    val nativePeakRssMb: Double = 0.0,
    val nativeVmSizeMb: Double = 0.0,
    val modelMemoryMb: Double = 0.0,
    val kvCacheMemoryMb: Double = 0.0,
    val contextCapacity: Int = 4096,
    val usedContextTokens: Int = 0,
    val remainingContextTokens: Int = 4096,
    val isContextAvailable: Boolean = false,
    val thermalStatus: Int = 0,
    val thermalStatusName: String = "Normal (0)",
    val isThermalThrottling: Boolean = false,
    val jvmHeapMb: Double = 0.0,
    val freeStorageBytes: Long = 0L,
    val modelsStorageBytes: Long = 0L,
    val errorMessage: String? = null,
    val importProgress: Float? = null,
    val benchmarkHistory: List<BenchmarkRun> = emptyList(),
    val isBenchmarking: Boolean = false,

    // On-Device AI Chat Experience States
    val isStatsDashboardVisible: Boolean = true,
    val isModelSelectorOpen: Boolean = false,
    val isInferenceSettingsOpen: Boolean = false,
    val isChatHistoryOpen: Boolean = false,
    val isRuntimeInfoOpen: Boolean = false,
    val isModelDetailsOpen: Boolean = false,
    val selectedModelFileForDetails: File? = null,
    val selectedDiscoverModelForDetails: DiscoverModelItem? = null,
    val pendingAttachments: List<MediaAttachment> = emptyList(),
    val conversations: List<Conversation> = emptyList(),
    val activeConversationId: String? = null,
    val activeDownloads: Map<String, DownloadItem> = emptyMap(),
    val discoverCatalog: List<DiscoverModelItem> = DEFAULT_DISCOVER_CATALOG,
    val openclResult: MobileLLMClient.OpenCLDiagnosticResult? = null,
    val isOpenclBenchmarking: Boolean = false
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val TAG = "ChatViewModel"
    private val client: MobileLLMClient = MobileLLMClient.getInstance(application)
    private var activeSession: ModelSession? = null
    private var generationJob: Job? = null
    private var benchmarkJob: Job? = null
    private var loadModelJob: Job? = null
    private val loadMutex = Mutex()
    private var thermalListener: PowerManager.OnThermalStatusChangedListener? = null

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val modelsDirs: List<File>
        get() {
            val app = getApplication<Application>()
            val dirs = mutableListOf<File>()
            dirs.add(File(app.filesDir, "models").apply { mkdirs() })
            app.getExternalFilesDir("models")?.let { dirs.add(it.apply { mkdirs() }) }
            try {
                val downloads = android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
                if (downloads != null && downloads.exists() && downloads.canRead() && !dirs.contains(downloads)) {
                    dirs.add(downloads)
                }
            } catch (_: Exception) {}
            try {
                val directSdDownload = File("/sdcard/Download")
                if (directSdDownload.exists() && directSdDownload.canRead() && !dirs.contains(directSdDownload)) {
                    dirs.add(directSdDownload)
                }
            } catch (_: Exception) {}
            return dirs
        }

    private val modelsDir: File
        get() = modelsDirs.first()

    init {
        initThermalMonitoring(application)
        viewModelScope.launch {
            try {
                client.initialize()
                val version = client.getVersion()
                val hw = client.detectHardware()
                _uiState.update {
                    it.copy(
                        engineVersion = version,
                        hardwareInfo = hw
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize engine", e)
                _uiState.update { it.copy(errorMessage = "Engine init failed: ${e.message}") }
            } finally {
                refreshSystemStats()
                refreshModelsList()
            }
        }
    }

    private fun initThermalMonitoring(app: Application) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val powerManager = app.getSystemService(Context.POWER_SERVICE) as? PowerManager
            if (powerManager != null) {
                try {
                    val initialStatus = powerManager.currentThermalStatus
                    updateThermalStatus(initialStatus)
                    val listener = PowerManager.OnThermalStatusChangedListener { status ->
                        updateThermalStatus(status)
                    }
                    powerManager.addThermalStatusListener(app.mainExecutor, listener)
                    thermalListener = listener
                } catch (e: Exception) {
                    Log.w(TAG, "Could not initialize thermal status listener: ${e.message}")
                }
            }
        } else {
            _uiState.update {
                it.copy(
                    thermalStatus = 0,
                    thermalStatusName = "Unsupported (< API 29)",
                    isThermalThrottling = false
                )
            }
        }
    }

    private fun updateThermalStatus(status: Int) {
        val name = when (status) {
            PowerManager.THERMAL_STATUS_NONE -> "Normal (0 - None)"
            PowerManager.THERMAL_STATUS_LIGHT -> "Light (1 - Light Throttling)"
            PowerManager.THERMAL_STATUS_MODERATE -> "Moderate (2 - Moderate Throttling)"
            PowerManager.THERMAL_STATUS_SEVERE -> "Severe (3 - Performance Impacted)"
            PowerManager.THERMAL_STATUS_CRITICAL -> "Critical (4 - Significant Throttling)"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency (5 - Shutdown Imminent)"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown (6 - Thermal Shutdown)"
            else -> "Status $status"
        }
        val isThrottling = status >= PowerManager.THERMAL_STATUS_MODERATE
        _uiState.update {
            it.copy(
                thermalStatus = status,
                thermalStatusName = name,
                isThermalThrottling = isThrottling
            )
        }
    }

    fun setScreen(screen: NexaScreen) {
        _uiState.update { it.copy(currentScreen = screen) }
        if (screen == NexaScreen.MODELS || screen == NexaScreen.DASHBOARD) {
            refreshSystemStats()
            refreshModelsList()
        }
    }

    fun refreshSystemStats() {
        val app = getApplication<Application>()
        val actManager = app.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo().also { actManager?.getMemoryInfo(it) }

        val totalRamMb = (memInfo.totalMem) / (1024 * 1024)
        val availRamMb = (memInfo.availMem) / (1024 * 1024)
        val lowMem = memInfo.lowMemory

        val runtime = Runtime.getRuntime()
        val jvmHeapMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0)

        val freeSpace = modelsDir.freeSpace
        val modelsSize = _uiState.value.availableModels.sumOf { it.length() }

        val procMem = try {
            client.getProcessMemory()
        } catch (e: Exception) {
            MobileLLMClient.ProcessMemorySnapshot()
        }

        val contextTelem = try {
            activeSession?.getContextTelemetry()
        } catch (e: Exception) {
            null
        }

        _uiState.update {
            it.copy(
                systemTotalRamMb = totalRamMb,
                systemAvailRamMb = availRamMb,
                isLowMemory = lowMem,
                jvmHeapMb = jvmHeapMb,
                nativeRssMb = procMem.currentRssMb,
                nativePeakRssMb = procMem.peakRssMb,
                nativeVmSizeMb = procMem.vmSizeMb,
                modelMemoryMb = procMem.modelAllocatedMb,
                kvCacheMemoryMb = procMem.kvAllocatedMb,
                contextCapacity = contextTelem?.contextCapacity ?: it.contextCapacity,
                usedContextTokens = contextTelem?.usedTokens ?: it.usedContextTokens,
                remainingContextTokens = contextTelem?.remainingCapacity ?: it.remainingContextTokens,
                isContextAvailable = contextTelem?.isAvailable ?: it.isContextAvailable,
                freeStorageBytes = freeSpace,
                modelsStorageBytes = modelsSize
            )
        }
    }

    fun refreshModelsList() {
        val files = modelsDirs.flatMap { dir ->
            dir.listFiles { f -> f.extension.equals("gguf", ignoreCase = true) }?.toList() ?: emptyList()
        }.filter { it.canRead() && it.length() > 0 }.distinctBy { it.name }
        val totalBytes = files.sumOf { it.length() }
        val freeBytes = modelsDir.freeSpace

        _uiState.update {
            it.copy(
                availableModels = files,
                modelsStorageBytes = totalBytes,
                freeStorageBytes = freeBytes
            )
        }

        // Asynchronously inspect headers for each model
        viewModelScope.launch(Dispatchers.IO) {
            val metaMap = _uiState.value.modelMetadataMap.toMutableMap()
            for (file in files) {
                if (!metaMap.containsKey(file.name)) {
                    try {
                        val meta = client.inspectModel(file, computeHash = false)
                        metaMap[file.name] = meta
                    } catch (e: Exception) {
                        Log.w(TAG, "Could not inspect model ${file.name}: ${e.message}")
                    }
                }
            }
            _uiState.update { it.copy(modelMetadataMap = metaMap) }
        }

        // Auto-load first available model if none is currently active and not already loading
        if (!_uiState.value.isModelLoaded && activeSession == null && files.isNotEmpty() && loadModelJob?.isActive != true) {
            loadModel(files.first())
        }
    }

    fun deleteModel(file: File) {
        if (file.name == _uiState.value.loadedModelName) {
            unloadModel()
        }
        if (file.exists()) {
            file.delete()
        }
        refreshModelsList()
        refreshSystemStats()
    }

    fun importModel(uri: Uri, contentResolver: ContentResolver) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                var displayName = "model_${System.currentTimeMillis()}.gguf"
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (nameIdx >= 0 && cursor.moveToFirst()) {
                        displayName = cursor.getString(nameIdx)
                    }
                }

                _uiState.update { it.copy(importProgress = 0f) }

                val targetFile = File(modelsDir, displayName)
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(targetFile).use { output ->
                        val buffer = ByteArray(64 * 1024)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                        }
                    }
                }
                targetFile.setReadable(true, false)

                // Verify file header validity
                val isValidGguf = try {
                    val meta = client.inspectModel(targetFile, computeHash = false)
                    meta.isValid
                } catch (e: Exception) {
                    targetFile.delete()
                    throw IllegalArgumentException("Invalid GGUF model: ${e.message}")
                }

                if (!isValidGguf) {
                    targetFile.delete()
                    throw IllegalArgumentException("File is not a valid GGUF model.")
                }

                withContext(Dispatchers.Main) {
                    _uiState.update { it.copy(importProgress = null) }
                    refreshModelsList()
                    loadModel(targetFile)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model import failed", e)
                withContext(Dispatchers.Main) {
                    _uiState.update {
                        it.copy(
                            importProgress = null,
                            errorMessage = "Import failed: ${e.message}"
                        )
                    }
                }
            }
        }
    }

    fun loadModel(file: File) {
        loadModelJob?.cancel()
        loadModelJob = viewModelScope.launch {
            loadMutex.withLock {
                try {
                    // Ensure previous session and model weights are completely unloaded to release memory budget
                    activeSession?.close()
                    activeSession = null
                    client.unloadAllModels()
                    _uiState.update { it.copy(isModelLoaded = false, loadedModelName = null) }
                    refreshSystemStats()

                    val config = ModelConfig(
                        modelPath = file.absolutePath,
                        preferredBackend = _uiState.value.preferredBackend,
                        contextLength = 2048,
                        nThreads = _uiState.value.threadCount,
                        nGpuLayers = _uiState.value.gpuLayers
                    )
                    val sessionConfig = SessionConfig(
                        maxContextTokens = 2048,
                        enablePrefixCaching = true
                    )

                    val session = client.loadModel(config, sessionConfig)
                    activeSession = session
                    _uiState.update {
                        it.copy(
                            isModelLoaded = true,
                            loadedModelName = file.name,
                            errorMessage = null
                        )
                    }
                    refreshSystemStats()
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load model", e)
                    _uiState.update {
                        it.copy(
                            isModelLoaded = false,
                            errorMessage = "Load model error: ${e.message}"
                        )
                    }
                }
            }
        }
    }

    fun unloadModel() {
        activeSession?.close()
        activeSession = null
        client.unloadAllModels()
        _uiState.update {
            it.copy(
                isModelLoaded = false,
                loadedModelName = null
            )
        }
        refreshSystemStats()
    }

    fun regenerateLastResponse() {
        val messages = _uiState.value.messages
        val lastUserIdx = messages.indexOfLast { it.sender == MessageSender.USER }
        if (lastUserIdx == -1) return

        val userPrompt = messages[lastUserIdx].text
        // Prune messages down to before the last user message
        val pruned = messages.subList(0, lastUserIdx)
        _uiState.update { it.copy(messages = pruned) }

        sendMessage(userPrompt)
    }

    fun sendMessage(promptText: String) {
        val trimmed = promptText.trim()
        if (trimmed.isEmpty()) return

        val session = activeSession
        if (session == null) {
            _uiState.update { it.copy(errorMessage = "Please import and load a GGUF model first.") }
            return
        }

        val attachments = _uiState.value.pendingAttachments
        val userMsg = ChatMessage(
            sender = MessageSender.USER,
            text = trimmed,
            attachments = attachments
        )
        val assistantMsgId = UUID.randomUUID().toString()
        val assistantMsg = ChatMessage(
            id = assistantMsgId,
            sender = MessageSender.ASSISTANT,
            text = "",
            isStreaming = true
        )

        _uiState.update {
            it.copy(
                messages = it.messages + userMsg + assistantMsg,
                isGenerating = true,
                errorMessage = null,
                pendingAttachments = emptyList()
            )
        }

        generationJob = viewModelScope.launch(Dispatchers.Default) {
            var fullText = ""
            var completedMetrics: GenerationMetrics? = null

            val promptForModel = if (!trimmed.startsWith("<|")) {
                "<|im_start|>user\n$trimmed<|im_end|>\n<|im_start|>assistant\n"
            } else {
                trimmed
            }

            session.generate(
                prompt = promptForModel,
                params = _uiState.value.samplingParams,
                onMetrics = { metrics ->
                    completedMetrics = metrics
                    _uiState.update { it.copy(lastMetrics = metrics) }
                }
            ).catch { e ->
                Log.e(TAG, "Error during generation stream", e)
                _uiState.update { state ->
                    state.copy(
                        isGenerating = false,
                        errorMessage = "Generation error: ${e.message}",
                        messages = state.messages.map { msg ->
                            if (msg.id == assistantMsgId) {
                                msg.copy(isStreaming = false)
                            } else msg
                        }
                    )
                }
            }.collect { chunk ->
                fullText += chunk.text
                _uiState.update { state ->
                    state.copy(
                        messages = state.messages.map { msg ->
                            if (msg.id == assistantMsgId) {
                                msg.copy(text = fullText)
                            } else msg
                        }
                    )
                }
            }

            val diagJson = try {
                session.getGenerationDiagnostics()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to read generation diagnostics", e)
                null
            }

            _uiState.update { state ->
                state.copy(
                    isGenerating = false,
                    lastMetrics = completedMetrics,
                    generationDiagnostics = diagJson ?: state.generationDiagnostics,
                    messages = state.messages.map { msg ->
                        if (msg.id == assistantMsgId) {
                            msg.copy(
                                text = fullText,
                                isStreaming = false,
                                metrics = completedMetrics
                            )
                        } else msg
                    }
                )
            }
            refreshSystemStats()
        }
    }

    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        _uiState.update { state ->
            state.copy(
                isGenerating = false,
                messages = state.messages.map { msg ->
                    if (msg.isStreaming) msg.copy(isStreaming = false) else msg
                }
            )
        }
    }

    fun updateSettings(params: SamplingParams, backend: BackendType, threads: Int, layers: Int) {
        val prevBackend = _uiState.value.preferredBackend
        val prevLayers = _uiState.value.gpuLayers
        val prevThreads = _uiState.value.threadCount

        val maxCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val clampedThreads = threads.coerceIn(1, maxCores)
        val finalLayers = if (backend == BackendType.CPU_NEON) 0 else layers

        _uiState.update {
            it.copy(
                samplingParams = params,
                preferredBackend = backend,
                threadCount = clampedThreads,
                gpuLayers = finalLayers
            )
        }

        val engineChanged = (prevBackend != backend) || (prevLayers != finalLayers)
        if (engineChanged && _uiState.value.isModelLoaded) {
            stopGeneration()
            val loadedName = _uiState.value.loadedModelName
            val currentFile = _uiState.value.availableModels.firstOrNull { it.name == loadedName }
            if (currentFile != null && currentFile.exists()) {
                Log.i(TAG, "Engine configuration changed (backend: $prevBackend -> $backend, layers: $prevLayers -> $finalLayers). Reloading model: ${currentFile.name}")
                loadModel(currentFile)
            }
        } else if (prevThreads != clampedThreads && activeSession != null) {
            viewModelScope.launch(Dispatchers.Default) {
                activeSession?.setThreads(clampedThreads)
            }
        }
    }

    fun updateSamplingParams(params: SamplingParams) {
        _uiState.update { it.copy(samplingParams = params) }
    }

    fun updateBackend(backend: BackendType) {
        val previous = _uiState.value.preferredBackend
        _uiState.update { it.copy(preferredBackend = backend) }
        if (previous != backend && _uiState.value.isModelLoaded) {
            val loadedName = _uiState.value.loadedModelName
            val currentFile = _uiState.value.availableModels.firstOrNull { it.name == loadedName }
            if (currentFile != null && currentFile.exists()) {
                Log.i(TAG, "Backend changed from $previous to $backend. Reloading model: ${currentFile.name}")
                loadModel(currentFile)
            }
        }
    }

    fun updateGpuLayers(layers: Int) {
        val previous = _uiState.value.gpuLayers
        _uiState.update { it.copy(gpuLayers = layers) }
        if (previous != layers && _uiState.value.isModelLoaded) {
            val loadedName = _uiState.value.loadedModelName
            val currentFile = _uiState.value.availableModels.firstOrNull { it.name == loadedName }
            if (currentFile != null && currentFile.exists()) {
                Log.i(TAG, "GPU layers changed from $previous to $layers. Reloading model: ${currentFile.name}")
                loadModel(currentFile)
            }
        }
    }

    fun updateThreads(threads: Int) {
        val maxCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val clamped = threads.coerceIn(1, maxCores)
        _uiState.update { it.copy(threadCount = clamped) }
        viewModelScope.launch(Dispatchers.Default) {
            val session = activeSession
            if (session != null) {
                val success = session.setThreads(clamped)
                Log.d(TAG, "Dynamically updated CPU threads to $clamped (result: $success)")
            }
        }
    }

    fun stopBenchmark() {
        benchmarkJob?.cancel()
        benchmarkJob = null
        stopGeneration()
        _uiState.update {
            it.copy(
                isBenchmarking = false,
                errorMessage = "Benchmark cancelled by user."
            )
        }
    }

    /**
     * Agent 12B: Qualcomm Adreno OpenCL Hardware Benchmark & GEMV Verification.
     * Runs 128-bit coalesced FP16 GEMV kernel directly on Adreno GPU via dynamic libOpenCL.so.
     */
    fun runOpenCLBenchmark() {
        if (_uiState.value.isOpenclBenchmarking) return
        viewModelScope.launch(Dispatchers.Default) {
            _uiState.update { it.copy(isOpenclBenchmarking = true) }
            val result = client.runOpenCLDiagnostic()
            _uiState.update {
                it.copy(
                    isOpenclBenchmarking = false,
                    openclResult = result
                )
            }
        }
    }

    /**
     * Agent 14: Thermal-Aware Benchmark Execution
     *
     * In accordance with Section 8 Thermal Safety guidelines:
     * - Rejects benchmark execution when thermal status is SEVERE, CRITICAL, or EMERGENCY.
     * - Default mode runs 1 Warmup (16 tok) + 1 Measured Run (32 tok) on the user's selected backend.
     * - Extended mode (sweep across 2T..8T) applies only to CPU and enforces adaptive cooldown.
     * - Aborts active benchmark if device thermals escalate to SEVERE or CRITICAL mid-run.
     * - Releases resources and resets thread state upon completion or cancellation.
     */
    fun runBenchmark(extended: Boolean = false) {
        val session = activeSession
        if (session == null || _uiState.value.isGenerating || _uiState.value.isBenchmarking) {
            return
        }

        // Section 8: Thermal Status Gate - detect severity before launching
        val currentThermal = _uiState.value.thermalStatus
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && currentThermal >= PowerManager.THERMAL_STATUS_SEVERE) {
            _uiState.update {
                it.copy(
                    errorMessage = "Benchmark blocked for thermal safety: Device thermal status is ${_uiState.value.thermalStatusName}. Please allow the device to cool down."
                )
            }
            return
        }

        benchmarkJob = viewModelScope.launch(Dispatchers.Default) {
            _uiState.update { it.copy(isBenchmarking = true, errorMessage = null) }
            val maxCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            val isVulkan = _uiState.value.preferredBackend == BackendType.VULKAN

            // Section 8: Default benchmark configuration:
            // Warmup: 1, Measured runs: 1, Output tokens: 32, Threads: 4 (or clamped maxCores)
            val threadConfigs = if (extended && !isVulkan) {
                listOf(2, 4, 6, 8).filter { it <= maxCores }.ifEmpty { listOf(maxCores) }
            } else {
                listOf(minOf(4, maxCores))
            }

            val benchmarkPrompt = "Explain the fundamental principles of artificial intelligence and machine learning in simple, clear terms for mobile engineers."

            try {
                for (threads in threadConfigs) {
                    // Check cancellation
                    if (!coroutineContext[Job]!!.isActive) break

                    // Section 8: Mid-run thermal severity check
                    val liveThermal = _uiState.value.thermalStatus
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && liveThermal >= PowerManager.THERMAL_STATUS_SEVERE) {
                        _uiState.update {
                            it.copy(
                                errorMessage = "Benchmark safely aborted: Device thermal state reached ${_uiState.value.thermalStatusName}."
                            )
                        }
                        break
                    }

                    updateThreads(threads)

                    // 1. Warmup iteration (16 tokens) per Section 8
                    try {
                        val warmupParams = SamplingParams(maxTokens = 16, temperature = 0.0f)
                        session.generate(
                            prompt = benchmarkPrompt,
                            params = warmupParams
                        ).collect {}
                    } catch (e: Exception) {
                        Log.w(TAG, "Benchmark warmup failed: ${e.message}")
                    }

                    // Section 8: Cooldown between benchmark runs (2000 ms)
                    delay(2000)

                    // Check cancellation after cooldown
                    if (!coroutineContext[Job]!!.isActive) break

                    // 2. Measured iteration (32 output tokens) per Section 8
                    var receivedMetrics: GenerationMetrics? = null
                    try {
                        val sampling = SamplingParams(maxTokens = 32, temperature = 0.0f)
                        session.generate(
                            prompt = benchmarkPrompt,
                            params = sampling,
                            onMetrics = { m -> receivedMetrics = m }
                        ).collect {}
                    } catch (e: Exception) {
                        Log.w(TAG, "Benchmark failed for thread count $threads: ${e.message}")
                    }

                    val finalMetrics = receivedMetrics
                    if (finalMetrics != null) {
                        val run = BenchmarkRun(
                            threads = threads,
                            promptTokens = finalMetrics.promptTokens,
                            completionTokens = finalMetrics.completionTokens,
                            promptTokensPerSecond = finalMetrics.promptTokensPerSecond,
                            tokensPerSecond = finalMetrics.tokensPerSecond,
                            timeToFirstTokenMs = finalMetrics.timeToFirstTokenMs,
                            totalDurationMs = finalMetrics.totalDurationMs,
                            peakRssMb = finalMetrics.peakRssMb,
                            thermalStatusName = _uiState.value.thermalStatusName
                        )
                        _uiState.update { state ->
                            state.copy(
                                benchmarkHistory = state.benchmarkHistory + run,
                                lastMetrics = finalMetrics
                            )
                        }
                    }

                    // Post-iteration cooldown
                    delay(1500)
                }
            } finally {
                _uiState.update { it.copy(isBenchmarking = false) }
                benchmarkJob = null
                refreshSystemStats()
            }
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(messages = emptyList()) }
    }

    fun dismissError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    // =========================================================================
    // ON-DEVICE MODEL DOWNLOAD MANAGER
    // =========================================================================

    private val downloadJobs = mutableMapOf<String, Job>()

    fun startModelDownload(item: DiscoverModelItem) {
        val targetFile = File(modelsDir, item.filename)
        if (targetFile.exists() && targetFile.length() > 0) {
            refreshModelsList()
            return
        }

        val downloadId = item.id
        val downloadItem = DownloadItem(
            id = downloadId,
            modelName = item.name,
            downloadUrl = item.downloadUrl,
            targetFileName = item.filename,
            totalBytes = item.downloadSizeMb * 1024L * 1024L,
            status = DownloadStatus.DOWNLOADING
        )

        _uiState.update {
            it.copy(activeDownloads = it.activeDownloads + (downloadId to downloadItem))
        }

        downloadJobs[downloadId]?.cancel()
        downloadJobs[downloadId] = viewModelScope.launch(Dispatchers.IO) {
            val tempFile = File(modelsDir, "${item.filename}.download")
            var downloaded = if (tempFile.exists()) tempFile.length() else 0L
            var connection: java.net.HttpURLConnection? = null
            try {
                val url = java.net.URL(item.downloadUrl)
                connection = url.openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 15000
                connection.readTimeout = 30000
                connection.instanceFollowRedirects = true
                if (downloaded > 0) {
                    connection.setRequestProperty("Range", "bytes=$downloaded-")
                }
                connection.connect()

                val responseCode = connection.responseCode
                val isPartial = responseCode == java.net.HttpURLConnection.HTTP_PARTIAL
                val isOk = responseCode == java.net.HttpURLConnection.HTTP_OK

                if (!isPartial && !isOk) {
                    throw java.io.IOException("HTTP error response: $responseCode")
                }

                val contentLength = connection.contentLengthLong
                val totalBytes = if (contentLength > 0) {
                    if (isPartial) contentLength + downloaded else contentLength
                } else {
                    item.downloadSizeMb * 1024L * 1024L
                }

                val append = isPartial && downloaded > 0
                val outputStream = java.io.FileOutputStream(tempFile, append)
                val inputStream = connection.inputStream

                val buffer = ByteArray(64 * 1024)
                var bytesRead: Int
                var lastSpeedCalcTime = System.currentTimeMillis()
                var bytesSinceLastSpeedCalc = 0L

                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    downloaded += bytesRead
                    bytesSinceLastSpeedCalc += bytesRead

                    val now = System.currentTimeMillis()
                    val timeDelta = now - lastSpeedCalcTime
                    if (timeDelta >= 500) {
                        val speed = (bytesSinceLastSpeedCalc.toDouble() / (timeDelta.toDouble() / 1000.0))
                        val progress = if (totalBytes > 0) (downloaded.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f) else 0f
                        _uiState.update { state ->
                            val current = state.activeDownloads[downloadId] ?: downloadItem
                            state.copy(
                                activeDownloads = state.activeDownloads + (downloadId to current.copy(
                                    downloadedBytes = downloaded,
                                    totalBytes = totalBytes,
                                    speedBytesPerSec = speed,
                                    progress = progress,
                                    status = DownloadStatus.DOWNLOADING
                                ))
                            )
                        }
                        lastSpeedCalcTime = now
                        bytesSinceLastSpeedCalc = 0L
                    }
                }
                outputStream.flush()
                outputStream.close()
                inputStream.close()

                if (tempFile.exists()) {
                    tempFile.renameTo(targetFile)
                }

                _uiState.update { state ->
                    val current = state.activeDownloads[downloadId] ?: downloadItem
                    state.copy(
                        activeDownloads = state.activeDownloads + (downloadId to current.copy(
                            downloadedBytes = totalBytes,
                            progress = 1.0f,
                            status = DownloadStatus.COMPLETED
                        ))
                    )
                }
                refreshModelsList()
                refreshSystemStats()
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) {
                    _uiState.update { state ->
                        val current = state.activeDownloads[downloadId] ?: downloadItem
                        state.copy(
                            activeDownloads = state.activeDownloads + (downloadId to current.copy(status = DownloadStatus.PAUSED))
                        )
                    }
                } else {
                    Log.e(TAG, "Download failed for ${item.name}", e)
                    _uiState.update { state ->
                        val current = state.activeDownloads[downloadId] ?: downloadItem
                        state.copy(
                            activeDownloads = state.activeDownloads + (downloadId to current.copy(
                                status = DownloadStatus.FAILED,
                                errorMessage = e.message
                            ))
                        )
                    }
                }
            } finally {
                connection?.disconnect()
            }
        }
    }

    fun pauseModelDownload(downloadId: String) {
        downloadJobs[downloadId]?.cancel()
        _uiState.update { state ->
            val item = state.activeDownloads[downloadId] ?: return@update state
            state.copy(activeDownloads = state.activeDownloads + (downloadId to item.copy(status = DownloadStatus.PAUSED)))
        }
    }

    fun cancelModelDownload(downloadId: String) {
        downloadJobs[downloadId]?.cancel()
        val item = _uiState.value.activeDownloads[downloadId]
        if (item != null) {
            val tempFile = File(modelsDir, "${item.targetFileName}.download")
            if (tempFile.exists()) tempFile.delete()
        }
        _uiState.update { state ->
            state.copy(activeDownloads = state.activeDownloads - downloadId)
        }
    }

    // =========================================================================
    // CHAT HISTORY & CONVERSATION MANAGEMENT
    // =========================================================================

    fun newConversation() {
        val currentMsgs = _uiState.value.messages
        val activeId = _uiState.value.activeConversationId
        var convs = _uiState.value.conversations

        if (currentMsgs.isNotEmpty()) {
            val title = currentMsgs.firstOrNull { it.sender == MessageSender.USER }?.text?.take(36) ?: "Chat ${convs.size + 1}"
            val existing = convs.find { it.id == activeId }
            if (existing != null) {
                convs = convs.map { if (it.id == activeId) it.copy(messages = currentMsgs, updatedAt = System.currentTimeMillis()) else it }
            } else {
                val newConv = Conversation(
                    id = activeId ?: UUID.randomUUID().toString(),
                    title = title,
                    modelName = _uiState.value.loadedModelName,
                    messages = currentMsgs
                )
                convs = listOf(newConv) + convs
            }
        }

        val freshId = UUID.randomUUID().toString()
        _uiState.update {
            it.copy(
                activeConversationId = freshId,
                messages = emptyList(),
                conversations = convs,
                isChatHistoryOpen = false
            )
        }
    }

    fun switchConversation(id: String) {
        val target = _uiState.value.conversations.find { it.id == id } ?: return
        _uiState.update {
            it.copy(
                activeConversationId = target.id,
                messages = target.messages,
                isChatHistoryOpen = false
            )
        }
    }

    fun deleteConversation(id: String) {
        _uiState.update { state ->
            val updated = state.conversations.filter { it.id != id }
            val newActive = if (state.activeConversationId == id) null else state.activeConversationId
            val newMsgs = if (state.activeConversationId == id) emptyList() else state.messages
            state.copy(
                conversations = updated,
                activeConversationId = newActive,
                messages = newMsgs
            )
        }
    }

    fun renameConversation(id: String, newTitle: String) {
        _uiState.update { state ->
            state.copy(
                conversations = state.conversations.map {
                    if (it.id == id) it.copy(title = newTitle, updatedAt = System.currentTimeMillis()) else it
                }
            )
        }
    }

    // =========================================================================
    // MESSAGE ACTIONS & ATTACHMENTS
    // =========================================================================

    fun deleteMessage(id: String) {
        _uiState.update { state ->
            state.copy(messages = state.messages.filter { it.id != id })
        }
    }

    fun editAndResendMessage(id: String, newText: String) {
        val messages = _uiState.value.messages
        val idx = messages.indexOfFirst { it.id == id }
        if (idx == -1) return
        val pruned = messages.subList(0, idx)
        _uiState.update { it.copy(messages = pruned) }
        sendMessage(newText)
    }

    fun addAttachment(attachment: MediaAttachment) {
        if (_uiState.value.pendingAttachments.size >= 10) return
        _uiState.update { it.copy(pendingAttachments = it.pendingAttachments + attachment) }
    }

    fun removeAttachment(id: String) {
        _uiState.update { it.copy(pendingAttachments = it.pendingAttachments.filter { it.id != id }) }
    }

    fun clearAttachments() {
        _uiState.update { it.copy(pendingAttachments = emptyList()) }
    }

    // =========================================================================
    // UI BOTTOM SHEETS & PANELS TOGGLE
    // =========================================================================

    fun toggleStatsDashboard() {
        _uiState.update { it.copy(isStatsDashboardVisible = !it.isStatsDashboardVisible) }
    }

    fun setStatsDashboardVisible(visible: Boolean) {
        _uiState.update { it.copy(isStatsDashboardVisible = visible) }
    }

    fun setModelSelectorOpen(open: Boolean) {
        _uiState.update { it.copy(isModelSelectorOpen = open) }
    }

    fun setInferenceSettingsOpen(open: Boolean) {
        _uiState.update { it.copy(isInferenceSettingsOpen = open) }
    }

    fun setChatHistoryOpen(open: Boolean) {
        _uiState.update { it.copy(isChatHistoryOpen = open) }
    }

    fun setRuntimeInfoOpen(open: Boolean) {
        _uiState.update { it.copy(isRuntimeInfoOpen = open) }
    }

    fun setModelDetailsOpen(open: Boolean, file: File? = null, discoverItem: DiscoverModelItem? = null) {
        _uiState.update {
            it.copy(
                isModelDetailsOpen = open,
                selectedModelFileForDetails = file,
                selectedDiscoverModelForDetails = discoverItem
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopGeneration()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && thermalListener != null) {
            val powerManager = getApplication<Application>().getSystemService(Context.POWER_SERVICE) as? PowerManager
            try {
                thermalListener?.let { powerManager?.removeThermalStatusListener(it) }
            } catch (e: Exception) {
                Log.w(TAG, "Failed removing thermal listener: ${e.message}")
            }
            thermalListener = null
        }
        activeSession?.close()
    }
}
