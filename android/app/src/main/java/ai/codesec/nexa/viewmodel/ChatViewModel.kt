package ai.codesec.nexa.viewmodel

import android.app.Application
import android.content.ContentResolver
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

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val sender: MessageSender,
    val text: String,
    val metrics: GenerationMetrics? = null,
    val isStreaming: Boolean = false,
    val timestamp: Long = System.currentTimeMillis()
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val isModelLoaded: Boolean = false,
    val loadedModelName: String? = null,
    val availableModels: List<File> = emptyList(),
    val samplingParams: SamplingParams = SamplingParams(),
    val preferredBackend: BackendType = BackendType.CPU_NEON,
    val threadCount: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    val engineVersion: String = "0.1.0-dev",
    val hardwareInfo: String = "Detecting...",
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
        initializeEngine()
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
        _uiState.update { it.copy(availableModels = files) }
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
                activeSession?.close()
                activeSession = null
                _uiState.update { it.copy(isModelLoaded = false, loadedModelName = null) }

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
        _uiState.update {
            it.copy(
                isModelLoaded = false,
                loadedModelName = null
            )
        }
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

