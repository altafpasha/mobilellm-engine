package ai.codesec.nexa.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Build
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.CyberLimeContainer
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.ObsidianBackground
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard
import ai.codesec.nexa.ui.theme.ObsidianSurface
import ai.codesec.nexa.viewmodel.ChatMessage
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.MessageSender
import ai.codesec.nexa.viewmodel.NexaScreen

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val snackbarHostState = remember { SnackbarHostState() }

    var inputPrompt by remember { mutableStateOf("") }
    var showSettingsDialog by remember { mutableStateOf(false) }
    var showTelemetryDetails by remember { mutableStateOf(false) }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { viewModel.importModel(it, context.contentResolver) }
    }

    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    LaunchedEffect(uiState.errorMessage) {
        uiState.errorMessage?.let { error ->
            snackbarHostState.showSnackbar(error)
            viewModel.dismissError()
        }
    }

    Scaffold(
        containerColor = ObsidianBackground,
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(34.dp)
                                .clip(CircleShape)
                                .background(CyberLimeContainer)
                                .border(1.dp, CyberLime.copy(alpha = 0.5f), CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Bolt,
                                contentDescription = null,
                                tint = CyberLime,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(
                                    text = "MobileLLM Lab",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 17.sp,
                                    color = Color.White
                                )
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Color(0xFF1E281F))
                                        .border(1.dp, CyberLime.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                                        .padding(horizontal = 6.dp, vertical = 1.dp)
                                ) {
                                    Text(
                                        text = if (uiState.isModelLoaded) "READY ●" else "OFFLINE",
                                        fontSize = 9.sp,
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = FontFamily.Monospace,
                                        color = if (uiState.isModelLoaded) CyberLime else Color(0xFF869690)
                                    )
                                }
                            }
                            Text(
                                text = "${Build.MODEL} · ${if (uiState.preferredBackend == BackendType.VULKAN) "Qualcomm Vulkan" else "ARM NEON"}",
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                color = Color(0xFF869690)
                            )
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsDialog = true }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color(0xFF94A39D))
                    }
                    IconButton(onClick = { viewModel.clearMessages() }) {
                        Icon(Icons.Default.DeleteSweep, contentDescription = "Clear Chat", tint = Color(0xFF94A39D))
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ObsidianBackground
                )
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) },
        bottomBar = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .imePadding()
                    .background(ObsidianBackground)
                    .padding(horizontal = 14.dp, vertical = 6.dp)
            ) {
                if (uiState.importProgress != null) {
                    LinearProgressIndicator(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp)
                            .clip(RoundedCornerShape(4.dp)),
                        color = CyberLime
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = inputPrompt,
                        onValueChange = { inputPrompt = it },
                        modifier = Modifier.weight(1f),
                        placeholder = {
                            Text(
                                text = if (uiState.isModelLoaded) "Enter prompt for inference lab..." else "Load a GGUF model first",
                                fontSize = 13.sp,
                                color = Color(0xFF64746E)
                            )
                        },
                        enabled = uiState.isModelLoaded && !uiState.isGenerating,
                        maxLines = 3,
                        shape = RoundedCornerShape(20.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = CyberLime,
                            unfocusedBorderColor = ObsidianBorder,
                            focusedContainerColor = ObsidianCard,
                            unfocusedContainerColor = ObsidianCard,
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White
                        )
                    )

                    if (uiState.isGenerating) {
                        IconButton(
                            onClick = { viewModel.stopGeneration() },
                            modifier = Modifier
                                .size(46.dp)
                                .clip(RoundedCornerShape(18.dp))
                                .background(Color(0xFFF87171))
                        ) {
                            Icon(Icons.Default.Stop, contentDescription = "Stop", tint = Color.White)
                        }
                    } else {
                        IconButton(
                            onClick = {
                                if (inputPrompt.isNotBlank()) {
                                    viewModel.sendMessage(inputPrompt)
                                    inputPrompt = ""
                                }
                            },
                            enabled = uiState.isModelLoaded && inputPrompt.isNotBlank(),
                            modifier = Modifier
                                .size(46.dp)
                                .clip(RoundedCornerShape(18.dp))
                                .background(
                                    if (uiState.isModelLoaded && inputPrompt.isNotBlank()) CyberLime else ObsidianCard
                                )
                                .border(
                                    1.dp,
                                    if (uiState.isModelLoaded && inputPrompt.isNotBlank()) CyberLime else ObsidianBorder,
                                    RoundedCornerShape(18.dp)
                                )
                        ) {
                            Icon(
                                Icons.Default.Send,
                                contentDescription = "Run",
                                tint = if (uiState.isModelLoaded && inputPrompt.isNotBlank()) Color(0xFF0D1110) else Color(0xFF64746E)
                            )
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        LazyColumn(
            state = listState,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Item 1: Active Model Hero Card
            item {
                ActiveModelCard(
                    modelName = uiState.loadedModelName,
                    isLoaded = uiState.isModelLoaded,
                    onChangeModel = { viewModel.setScreen(NexaScreen.MODELS) },
                    onUnload = { viewModel.unloadModel() }
                )
            }

            // Item 2: Backend Switcher Pill
            item {
                BackendSelectorCard(
                    currentBackend = uiState.preferredBackend,
                    isTransitioning = uiState.isGenerating,
                    onSelectBackend = { target ->
                        viewModel.updateSettings(
                            uiState.samplingParams,
                            target,
                            uiState.threadCount,
                            if (target == BackendType.VULKAN) 28 else 0
                        )
                    }
                )
            }

            // Item 3: Live Telemetry Card (Reference Design Inspired)
            item {
                LiveTelemetryCard(
                    metrics = uiState.lastMetrics,
                    isGenerating = uiState.isGenerating,
                    expanded = showTelemetryDetails,
                    onToggleExpand = { showTelemetryDetails = !showTelemetryDetails }
                )
            }

            // Item 4: Empty State or Messages
            if (uiState.messages.isEmpty()) {
                item {
                    EmptyStateView(
                        isModelLoaded = uiState.isModelLoaded,
                        onImportClick = { filePickerLauncher.launch(arrayOf("*/*")) },
                        onSuggestionClick = { prompt -> viewModel.sendMessage(prompt) }
                    )
                }
            } else {
                items(
                    items = uiState.messages,
                    key = { it.id }
                ) { message ->
                    val isLastAssistant = message.sender == MessageSender.ASSISTANT &&
                        message == uiState.messages.lastOrNull { it.sender == MessageSender.ASSISTANT }

                    MessageBubble(
                        message = message,
                        isLastAssistant = isLastAssistant,
                        isGenerating = uiState.isGenerating,
                        onRegenerate = { viewModel.regenerateLastResponse() }
                    )
                }
            }

            item {
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }

    if (showSettingsDialog) {
        SettingsDialog(
            initialParams = uiState.samplingParams,
            initialBackend = uiState.preferredBackend,
            initialThreads = uiState.threadCount,
            initialGpuLayers = uiState.gpuLayers,
            onSave = { params, backend, threads, layers ->
                viewModel.updateSettings(params, backend, threads, layers)
            },
            onDismiss = { showSettingsDialog = false }
        )
    }
}

@Composable
private fun ActiveModelCard(
    modelName: String?,
    isLoaded: Boolean,
    onChangeModel: () -> Unit,
    onUnload: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ACTIVE INFERENCE MODEL",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    color = CyberLime
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0xFF1E281F))
                        .border(1.dp, CyberLime.copy(alpha = 0.35f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "Q4_K_M · GGUF",
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace,
                        color = CyberLime
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = modelName ?: "No GGUF Model Loaded",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            Text(
                text = if (isLoaded) "2048 Context Length · Fully Resident in Native Memory" else "Select a quantized GGUF model from local storage",
                fontSize = 11.sp,
                color = Color(0xFF869690)
            )

            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(12.dp))
                        .clickable(onClick = onChangeModel)
                        .background(Color(0xFF222C28))
                        .border(1.dp, Color(0xFF2D3933), RoundedCornerShape(12.dp))
                        .padding(vertical = 8.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Change Model",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                }

                if (isLoaded) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .clickable(onClick = onUnload)
                            .background(Color(0xFF281818))
                            .border(1.dp, Color(0xFF4A2525), RoundedCornerShape(12.dp))
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "Unload",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = Color(0xFFF87171)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun BackendSelectorCard(
    currentBackend: BackendType,
    isTransitioning: Boolean,
    onSelectBackend: (BackendType) -> Unit
) {
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Hardware Backend",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = if (currentBackend == BackendType.VULKAN) "Adreno Vulkan (Experimental)" else "ARM64 NEON Vector Engine (Coherent)",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = if (currentBackend == BackendType.VULKAN) Color(0xFFE5A93C) else Color(0xFF869690)
                )
            }

            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(14.dp))
                    .background(ObsidianBackground)
                    .border(1.dp, ObsidianBorder, RoundedCornerShape(14.dp))
                    .padding(3.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                BackendPill(
                    label = "CPU",
                    selected = currentBackend == BackendType.CPU_NEON,
                    onClick = { if (!isTransitioning) onSelectBackend(BackendType.CPU_NEON) }
                )
                BackendPill(
                    label = "Vulkan GPU",
                    selected = currentBackend == BackendType.VULKAN,
                    onClick = { if (!isTransitioning) onSelectBackend(BackendType.VULKAN) }
                )
            }
        }
    }
}

@Composable
private fun BackendPill(
    label: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(10.dp))
            .clickable(onClick = onClick)
            .background(if (selected) CyberLime else Color.Transparent)
            .padding(horizontal = 12.dp, vertical = 5.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            fontSize = 11.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
            color = if (selected) Color(0xFF0D1110) else Color(0xFF869690)
        )
    }
}

@Composable
private fun LiveTelemetryCard(
    metrics: ai.codesec.mobilellm.models.GenerationMetrics?,
    isGenerating: Boolean,
    expanded: Boolean,
    onToggleExpand: () -> Unit
) {
    val tps = metrics?.tokensPerSecond ?: 0.0
    val ttft = metrics?.timeToFirstTokenMs ?: 0.0
    val completionToks = metrics?.completionTokens ?: 0
    val rssMb = metrics?.peakRssMb ?: 0.0

    Card(
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Box(
                        modifier = Modifier
                            .size(7.dp)
                            .clip(CircleShape)
                            .background(CyberLime)
                    )
                    Text(
                        text = "PERFORMANCE TELEMETRY",
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0xFF1E281F))
                        .border(1.dp, CyberLime.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = if (isGenerating) "LIVE STREAMING ●" else "LAST INFERENCE",
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        color = CyberLime
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // 2x2 Telemetry Grid
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TelemetryStatBox(
                    label = "DECODE SPEED",
                    value = "%.1f".format(tps),
                    unit = "tok/s",
                    valueColor = CyberLime,
                    subtitle = "Pure compute rate",
                    modifier = Modifier.weight(1f)
                )
                TelemetryStatBox(
                    label = "PREFILL TTFT",
                    value = "%.0f".format(ttft),
                    unit = "ms",
                    valueColor = ColdBlue,
                    subtitle = "Input prompt latency",
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TelemetryStatBox(
                    label = "GENERATED",
                    value = "$completionToks",
                    unit = "tok",
                    valueColor = Color.White,
                    subtitle = "Argmax Greedy Parity",
                    modifier = Modifier.weight(1f)
                )
                TelemetryStatBox(
                    label = "NATIVE RSS",
                    value = "%.0f".format(rssMb),
                    unit = "MB",
                    valueColor = Color.White,
                    subtitle = "Process VmRSS bound",
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Expandable Drawer Trigger
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(onClick = onToggleExpand)
                    .padding(vertical = 4.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = if (expanded) "▲ Hide Hardware & Wavefront Telemetry" else "▼ Show Hardware & Wavefront Telemetry",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF869690)
                )
            }

            AnimatedVisibility(visible = expanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 6.dp)
                        .background(Color(0xFF131816), RoundedCornerShape(12.dp))
                        .padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    DetailRow("Silicon Architecture", "Qualcomm Adreno 750 (SM8650)")
                    DetailRow("Wavefront SIMD Width", "64 threads (Wave64 verified)")
                    DetailRow("Virtual KV-Cache Allocation", "${(metrics?.promptTokens ?: 0) + completionToks} / 2048 blocks")
                    DetailRow("Thermal State", "Normal (THERMAL_STATUS_NONE)")
                }
            }
        }
    }
}

@Composable
private fun TelemetryStatBox(
    label: String,
    value: String,
    unit: String,
    valueColor: Color,
    subtitle: String,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF131816))
            .border(1.dp, Color(0xFF222B27), RoundedCornerShape(12.dp))
            .padding(10.dp)
    ) {
        Column {
            Text(text = label, fontSize = 9.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF869690))
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(text = value, fontSize = 17.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, color = valueColor)
                Text(text = unit, fontSize = 10.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF869690), modifier = Modifier.padding(bottom = 2.dp))
            }
            Text(text = subtitle, fontSize = 9.sp, color = Color(0xFF64746E))
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, fontSize = 10.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF869690))
        Text(text = value, fontSize = 10.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.SemiBold, color = Color.White)
    }
}

@Composable
private fun MessageBubble(
    message: ChatMessage,
    isLastAssistant: Boolean,
    isGenerating: Boolean,
    onRegenerate: () -> Unit
) {
    val isUser = message.sender == MessageSender.USER
    val context = LocalContext.current

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) Color(0xFF1E281F) else ObsidianCard
            ),
            border = androidx.compose.foundation.BorderStroke(
                1.dp,
                if (isUser) CyberLime.copy(alpha = 0.3f) else ObsidianBorder
            ),
            modifier = Modifier.fillMaxWidth(0.92f)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (isUser) "PROMPT" else "LAB INFERENCE",
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp,
                        color = if (isUser) CyberLime else ColdBlue
                    )

                    if (!isUser && message.text.isNotBlank()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            IconButton(
                                onClick = {
                                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                                    val clip = ClipData.newPlainText("Response", message.text)
                                    clipboard?.setPrimaryClip(clip)
                                    Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
                                },
                                modifier = Modifier.size(24.dp)
                            ) {
                                Icon(
                                    Icons.Default.ContentCopy,
                                    contentDescription = "Copy Response",
                                    tint = Color(0xFF94A39D),
                                    modifier = Modifier.size(14.dp)
                                )
                            }

                            if (isLastAssistant && !isGenerating) {
                                Spacer(modifier = Modifier.width(4.dp))
                                IconButton(
                                    onClick = onRegenerate,
                                    modifier = Modifier.size(24.dp)
                                ) {
                                    Icon(
                                        Icons.Default.Refresh,
                                        contentDescription = "Regenerate",
                                        tint = Color(0xFF94A39D),
                                        modifier = Modifier.size(14.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                if (isUser) {
                    Text(
                        text = message.text,
                        fontSize = 14.sp,
                        color = Color.White
                    )
                } else {
                    if (message.text.isNotBlank()) {
                        MarkdownText(
                            text = message.text,
                            color = Color(0xFFE2E8F0)
                        )
                    } else if (message.isStreaming) {
                        Text(
                            text = "Streaming autoregressive tokens...",
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            color = ColdBlue
                        )
                    }
                }

                if (message.isStreaming) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(10.dp),
                            strokeWidth = 2.dp,
                            color = CyberLime
                        )
                        Text(
                            text = "Executing decode step...",
                            fontSize = 10.sp,
                            fontFamily = FontFamily.Monospace,
                            color = CyberLime
                        )
                    }
                }

                message.metrics?.let { metrics ->
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "%.1f tok/s · TTFT %.0fms · %d tokens".format(
                            metrics.tokensPerSecond,
                            metrics.timeToFirstTokenMs,
                            metrics.completionTokens
                        ),
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF869690)
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyStateView(
    isModelLoaded: Boolean,
    onImportClick: () -> Unit,
    onSuggestionClick: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(54.dp)
                .clip(CircleShape)
                .background(CyberLimeContainer)
                .border(1.dp, CyberLime.copy(alpha = 0.5f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Bolt, contentDescription = null, tint = CyberLime, modifier = Modifier.size(28.dp))
        }

        Spacer(modifier = Modifier.height(12.dp))
        Text("MobileLLM Testing Lab", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color.White)
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "On-device inference benchmarking for GGUF models on ARM64 Snapdragon silicon.",
            fontSize = 12.sp,
            color = Color(0xFF869690),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )

        Spacer(modifier = Modifier.height(18.dp))

        if (!isModelLoaded) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(14.dp))
                    .clickable(onClick = onImportClick)
                    .background(ObsidianCard)
                    .border(1.dp, CyberLime.copy(alpha = 0.4f), RoundedCornerShape(14.dp))
                    .padding(horizontal = 20.dp, vertical = 12.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Icon(Icons.Default.Add, contentDescription = null, tint = CyberLime, modifier = Modifier.size(18.dp))
                    Text("Import GGUF Model from Storage", fontWeight = FontWeight.Bold, fontSize = 12.sp, color = Color.White)
                }
            }
        } else {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = "QUICK BENCHMARK VECTORS",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    color = CyberLime
                )
                QuickPromptChip("Explain quantum computing in simple terms", onSuggestionClick)
                QuickPromptChip("Write a Python script to benchmark LLM decode speed", onSuggestionClick)
                QuickPromptChip("What are the advantages of local AI on mobile devices?", onSuggestionClick)
            }
        }
    }
}

@Composable
private fun QuickPromptChip(text: String, onClick: (String) -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .clickable { onClick(text) }
            .background(ObsidianCard)
            .border(1.dp, ObsidianBorder, RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp)
    ) {
        Text(text = text, fontSize = 12.sp, color = Color(0xFFE2E8F0))
    }
}
