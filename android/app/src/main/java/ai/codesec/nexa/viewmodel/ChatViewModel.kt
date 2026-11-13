package ai.codesec.nexa.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.ContentResolver
import android.content.Context
import android.net.Uri
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
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val sender: MessageSender,
    val text: String,
    val metrics: GenerationMetrics? = null,
    val isStreaming: Boolean = false,
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
    val threadCount: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    val engineVersion: String = "0.1.0-dev",
    val hardwareInfo: String = "Detecting...",
    val lastMetrics: GenerationMetrics? = null,
    val systemTotalRamMb: Long = 0L,
    val systemAvailRamMb: Long = 0L,
    val isLowMemory: Boolean = false,
    val nativeRssMb: Double = 0.0,
    val jvmHeapMb: Double = 0.0,
    val freeStorageBytes: Long = 0L,
    val modelsStorageBytes: Long = 0L,
    val errorMessage: String? = null,
    val importProgress: Float? = null
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val TAG = "ChatViewModel"
    private val client: MobileLLMClient = MobileLLMClient.getInstance(application)
    private var activeSession: ModelSession? = null
    private var generationJob: Job? = null

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val modelsDir: File
        get() = File(getApplication<Application>().filesDir, "models").apply { mkdirs() }

    init {
        refreshModelsList()
        refreshSystemStats()
        initializeEngine()
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

        _uiState.update {
            it.copy(
                systemTotalRamMb = totalRamMb,
                systemAvailRamMb = availRamMb,
                isLowMemory = lowMem,
                jvmHeapMb = jvmHeapMb,
                freeStorageBytes = freeSpace,
                modelsStorageBytes = modelsSize
            )
        }
    }

    private fun initializeEngine() {
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
            }
        }
    }

    fun refreshModelsList() {
        val files = modelsDir.listFiles { f -> f.extension.equals("gguf", ignoreCase = true) }
            ?.toList() ?: emptyList()
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

        // Auto-load first available model if none is currently active
        if (!_uiState.value.isModelLoaded && activeSession == null && files.isNotEmpty()) {
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
        viewModelScope.launch {
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
                    nThreads = _uiState.value.threadCount
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

        val userMsg = ChatMessage(
            sender = MessageSender.USER,
            text = trimmed
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
                errorMessage = null
            )
        }

        generationJob = viewModelScope.launch {
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

            _uiState.update { state ->
                state.copy(
                    isGenerating = false,
                    lastMetrics = completedMetrics,
                    nativeRssMb = completedMetrics?.peakRssMb ?: state.nativeRssMb,
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

    fun updateSamplingParams(params: SamplingParams) {
        _uiState.update { it.copy(samplingParams = params) }
    }

    fun updateBackend(backend: BackendType) {
        _uiState.update { it.copy(preferredBackend = backend) }
    }

    fun updateThreads(threads: Int) {
        _uiState.update { it.copy(threadCount = threads) }
        // If a model is currently loaded, reload it with the new thread count
        val loadedName = _uiState.value.loadedModelName
        if (loadedName != null && _uiState.value.isModelLoaded) {
            val file = File(modelsDir, loadedName)
            if (file.exists()) {
                loadModel(file)
            }
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(messages = emptyList()) }
    }

    fun dismissError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        stopGeneration()
        activeSession?.close()
    }
}
