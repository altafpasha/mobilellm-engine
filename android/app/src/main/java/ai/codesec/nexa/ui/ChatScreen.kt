package ai.codesec.nexa.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.outlined.FileDownload
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Mic
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import ai.codesec.nexa.ui.components.ChatHistoryBottomSheet
import ai.codesec.nexa.ui.components.InferenceSettingsBottomSheet
import ai.codesec.nexa.ui.components.InferenceStatsDashboard
import ai.codesec.nexa.ui.components.ModelSelectorBottomSheet
import ai.codesec.nexa.ui.components.RuntimeInfoBottomSheet
import ai.codesec.nexa.ui.theme.BrandCoral
import ai.codesec.nexa.ui.theme.BrandCoralContainer
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.StatsBlue
import ai.codesec.nexa.ui.theme.StudioBackground
import ai.codesec.nexa.ui.theme.StudioBorder
import ai.codesec.nexa.ui.theme.StudioCard
import ai.codesec.nexa.ui.theme.StudioCardHighlight
import ai.codesec.nexa.ui.theme.StudioSurface
import ai.codesec.nexa.viewmodel.ChatMessage
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.MediaAttachment
import ai.codesec.nexa.viewmodel.MessageSender
import ai.codesec.nexa.viewmodel.NexaScreen

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val snackbarHostState = remember { SnackbarHostState() }

    var inputPrompt by remember { mutableStateOf("") }
    var editingMessageId by remember { mutableStateOf<String?>(null) }
    var showAttachmentDialog by remember { mutableStateOf(false) }

    // Activity result launcher for images
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        uris.take(10).forEach { uri ->
            viewModel.addAttachment(
                MediaAttachment(
                    uri = uri.toString(),
                    mimeType = "image/*",
                    name = "Image ${uiState.pendingAttachments.size + 1}",
                    isAudio = false
                )
            )
        }
    }

    // Scroll to bottom when messages update
    LaunchedEffect(uiState.messages.size, uiState.messages.lastOrNull()?.text?.length) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Scaffold(
        containerColor = StudioBackground,
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopChatNavigationBar(
                loadedModelName = uiState.loadedModelName,
                isModelLoaded = uiState.isModelLoaded,
                onBack = { /* Optional activity back or navigate */ },
                onOpenModelSelector = { viewModel.setModelSelectorOpen(true) },
                onOpenDashboard = { viewModel.setScreen(NexaScreen.DASHBOARD) },
                onOpenSettings = { viewModel.setInferenceSettingsOpen(true) },
                onOpenHistory = { viewModel.setChatHistoryOpen(true) }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = innerPadding.calculateTopPadding())
                .imePadding()
        ) {
            // Main Chat Area
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                if (uiState.messages.isEmpty()) {
                    EmptyChatState(
                        onSelectPrompt = { prompt ->
                            inputPrompt = prompt
                        }
                    )
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        items(uiState.messages, key = { it.id }) { msg ->
                            ChatMessageBubble(
                                message = msg,
                                onCopy = {
                                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                    clipboard.setPrimaryClip(ClipData.newPlainText("Chat", msg.text))
                                    Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                                },
                                onRegenerate = { viewModel.regenerateLastResponse() },
                                onEdit = {
                                    inputPrompt = msg.text
                                    editingMessageId = msg.id
                                },
                                onDelete = { viewModel.deleteMessage(msg.id) },
                                onStop = { viewModel.stopGeneration() }
                            )
                        }
                    }
                }
            }

            // Live Inference Statistics Dashboard directly above message input
            InferenceStatsDashboard(
                uiState = uiState,
                visible = uiState.isStatsDashboardVisible,
                onOpenInfo = { viewModel.setRuntimeInfoOpen(true) }
            )

            // Bottom Message Input Area
            ChatInputBar(
                inputPrompt = inputPrompt,
                isGenerating = uiState.isGenerating,
                isStatsVisible = uiState.isStatsDashboardVisible,
                pendingAttachments = uiState.pendingAttachments,
                onValueChange = { inputPrompt = it },
                onSend = {
                    if (editingMessageId != null) {
                        viewModel.editAndResendMessage(editingMessageId!!, inputPrompt)
                        editingMessageId = null
                    } else {
                        viewModel.sendMessage(inputPrompt)
                    }
                    inputPrompt = ""
                },
                onStop = { viewModel.stopGeneration() },
                onToggleStats = { viewModel.toggleStatsDashboard() },
                onAddAttachment = { showAttachmentDialog = true },
                onRemoveAttachment = { viewModel.removeAttachment(it) },
                onMicClick = {
                    Toast.makeText(context, "Voice input ready (Say your prompt)", Toast.LENGTH_SHORT).show()
                }
            )
        }
    }

    // Modal Bottom Sheets
    if (uiState.isModelSelectorOpen) {
        ModelSelectorBottomSheet(
            uiState = uiState,
            onSelectModel = { file -> viewModel.loadModel(file) },
            onOpenModelLibrary = { viewModel.setScreen(NexaScreen.MODELS) },
            onDismiss = { viewModel.setModelSelectorOpen(false) }
        )
    }

    if (uiState.isInferenceSettingsOpen) {
        InferenceSettingsBottomSheet(
            initialParams = uiState.samplingParams,
            initialBackend = uiState.preferredBackend,
            initialThreads = uiState.threadCount,
            initialGpuLayers = uiState.gpuLayers,
            openclResult = uiState.openclResult,
            isOpenclBenchmarking = uiState.isOpenclBenchmarking,
            onRunOpenCLBenchmark = { viewModel.runOpenCLBenchmark() },
            onSave = { params, backend, threads, layers ->
                viewModel.updateSettings(params, backend, threads, layers)
            },
            onDismiss = { viewModel.setInferenceSettingsOpen(false) }
        )
    }

    if (uiState.isChatHistoryOpen) {
        ChatHistoryBottomSheet(
            conversations = uiState.conversations,
            activeId = uiState.activeConversationId,
            onNewChat = { viewModel.newConversation() },
            onSelectChat = { id -> viewModel.switchConversation(id) },
            onDeleteChat = { id -> viewModel.deleteConversation(id) },
            onRenameChat = { id, title -> viewModel.renameConversation(id, title) },
            onDismiss = { viewModel.setChatHistoryOpen(false) }
        )
    }

    if (uiState.isRuntimeInfoOpen) {
        RuntimeInfoBottomSheet(
            onDismiss = { viewModel.setRuntimeInfoOpen(false) }
        )
    }

    // Attachment Picker Dialog
    if (showAttachmentDialog) {
        AlertDialog(
            onDismissRequest = { showAttachmentDialog = false },
            containerColor = StudioCard,
            shape = RoundedCornerShape(20.dp),
            title = {
                Text("Add Context Attachment", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 16.sp)
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "On-device models support up to 10 images per chat and 1 audio clip (up to 30s).",
                        fontSize = 12.sp,
                        color = Color(0xFF9CA3AF)
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .clickable {
                                showAttachmentDialog = false
                                imagePickerLauncher.launch("image/*")
                            }
                            .background(StudioCardHighlight)
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Outlined.Image, contentDescription = null, tint = BrandCoral, modifier = Modifier.size(20.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("Attach Photos / Images", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .clickable {
                                showAttachmentDialog = false
                                viewModel.addAttachment(
                                    MediaAttachment(
                                        uri = "audio://sample_recording.m4a",
                                        mimeType = "audio/mp4",
                                        name = "Audio Clip (12s)",
                                        isAudio = true
                                    )
                                )
                                Toast.makeText(context, "Attached audio clip", Toast.LENGTH_SHORT).show()
                            }
                            .background(StudioCardHighlight)
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Outlined.Mic, contentDescription = null, tint = StatsBlue, modifier = Modifier.size(20.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("Attach Voice / Audio Note (30s)", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showAttachmentDialog = false }) {
                    Text("Cancel", color = Color(0xFF9CA3AF))
                }
            }
        )
    }
}

/**
 * Top Navigation Bar matching reference screenshot layout:
 * Left: Back Arrow
 * Center Top: Chat icon + AI Chat (red accent)
 * Center Pill: Dropdown selector [ ⤓ <Model-Name> ▾ ]
 * Right: Tune (Settings) and History buttons
 */
@Composable
private fun TopChatNavigationBar(
    loadedModelName: String?,
    isModelLoaded: Boolean,
    onBack: () -> Unit,
    onOpenModelSelector: () -> Unit,
    onOpenDashboard: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenHistory: () -> Unit
) {
    Surface(
        color = StudioBackground,
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Back button
                IconButton(
                    onClick = onBack,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }

                // Center Title & Model Selector Pill (Responsive, flexible)
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Title: Icon + AI Chat in Coral
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Chat,
                            contentDescription = null,
                            tint = BrandCoral,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "AI Chat",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            color = BrandCoral
                        )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // Model Selector Dropdown Pill
                    val displayName = loadedModelName?.take(22) ?: "Select Model"
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .clickable(onClick = onOpenModelSelector)
                            .background(StudioCard)
                            .border(1.dp, StudioBorder, RoundedCornerShape(16.dp))
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(horizontal = 2.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.FileDownload,
                                contentDescription = null,
                                tint = BrandCoral,
                                modifier = Modifier.size(13.dp)
                            )
                            Spacer(modifier = Modifier.width(5.dp))
                            Text(
                                text = displayName,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.SemiBold,
                                color = Color.White,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Spacer(modifier = Modifier.width(3.dp))
                            Icon(
                                imageVector = Icons.Default.KeyboardArrowDown,
                                contentDescription = null,
                                tint = Color(0xFF9CA3AF),
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }

                // Right action icons: Telemetry/Benchmarks, Settings (Tune) & History
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    IconButton(
                        onClick = onOpenDashboard,
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Speed,
                            contentDescription = "Model Lab Telemetry & Benchmarks",
                            tint = CyberLime,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    IconButton(
                        onClick = onOpenSettings,
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Tune,
                            contentDescription = "Inference Settings",
                            tint = Color(0xFFE5E7EB),
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    IconButton(
                        onClick = onOpenHistory,
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.History,
                            contentDescription = "Chat History",
                            tint = Color(0xFFE5E7EB),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Centered Empty Chat State matching reference screenshot:
 * Title: AI Chat
 * Subtitle: Chat with on-device large language models.
 * Feature description: Support up to 10 images per chat and 1 audio clip up to 30 seconds.
 */
@Composable
private fun EmptyChatState(
    onSelectPrompt: (String) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(bottom = 60.dp)
        ) {
            Text(
                text = "AI Chat",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(10.dp))

            Text(
                text = "Chat with on-device large language\nmodels.",
                fontSize = 15.sp,
                color = Color(0xFFD1D5DB),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                lineHeight = 22.sp
            )

            Spacer(modifier = Modifier.height(14.dp))

            Text(
                text = "Support up to 10 images per chat\nand 1 audio clip up to 30 seconds.",
                fontSize = 13.sp,
                color = Color(0xFF9CA3AF),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                lineHeight = 19.sp
            )

            Spacer(modifier = Modifier.height(28.dp))

            // Quick suggestion chips
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                SuggestionPromptCard(
                    prompt = "Explain quantum computing in simple terms",
                    onClick = { onSelectPrompt("Explain quantum computing in simple terms") }
                )
                SuggestionPromptCard(
                    prompt = "Explain nucleosynthesis briefly",
                    onClick = { onSelectPrompt("Explain nucleosynthesis briefly") }
                )
                SuggestionPromptCard(
                    prompt = "Write a python script for continuous batching",
                    onClick = { onSelectPrompt("Write a python script for continuous batching") }
                )
            }
        }
    }
}

@Composable
private fun SuggestionPromptCard(
    prompt: String,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .clickable(onClick = onClick)
            .background(StudioCard)
            .border(1.dp, StudioBorder, RoundedCornerShape(14.dp))
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = prompt,
                fontSize = 12.sp,
                color = Color(0xFFE5E7EB),
                modifier = Modifier.weight(1f)
            )
            Icon(
                imageVector = Icons.Default.PlayArrow,
                contentDescription = null,
                tint = BrandCoral,
                modifier = Modifier.size(14.dp)
            )
        }
    }
}

/**
 * Message Bubble supporting user messages with edit/delete/copy,
 * and assistant responses with Markdown, Code blocks, and latency pills.
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ChatMessageBubble(
    message: ChatMessage,
    onCopy: () -> Unit,
    onRegenerate: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    onStop: () -> Unit
) {
    val isUser = message.sender == MessageSender.USER
    var showMenu by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        Box(
            modifier = Modifier
                .clip(
                    RoundedCornerShape(
                        topStart = 18.dp,
                        topEnd = 18.dp,
                        bottomStart = if (isUser) 18.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 18.dp
                    )
                )
                .combinedClickable(
                    onClick = {},
                    onLongClick = { showMenu = true }
                )
                .background(if (isUser) Color(0xFF261D1C) else StudioCard)
                .border(
                    1.dp,
                    if (isUser) BrandCoral.copy(alpha = 0.35f) else StudioBorder,
                    RoundedCornerShape(
                        topStart = 18.dp,
                        topEnd = 18.dp,
                        bottomStart = if (isUser) 18.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 18.dp
                    )
                )
                .padding(14.dp)
        ) {
            Column {
                // If user attached media, display chips
                if (message.attachments.isNotEmpty()) {
                    Row(
                        modifier = Modifier.padding(bottom = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        message.attachments.forEach { att ->
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(StudioCardHighlight)
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = if (att.isAudio) Icons.Outlined.Mic else Icons.Outlined.Image,
                                        contentDescription = null,
                                        tint = if (att.isAudio) StatsBlue else BrandCoral,
                                        modifier = Modifier.size(12.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(att.name, fontSize = 10.sp, color = Color.White)
                                }
                            }
                        }
                    }
                }

                if (isUser) {
                    Text(
                        text = message.text,
                        fontSize = 14.sp,
                        color = Color.White,
                        lineHeight = 20.sp
                    )
                } else {
                    MarkdownText(
                        text = message.text,
                        color = Color(0xFFF3F4F6)
                    )

                    if (message.isStreaming) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(
                                strokeWidth = 2.dp,
                                color = BrandCoral,
                                modifier = Modifier.size(10.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Generating...",
                                fontSize = 10.sp,
                                fontFamily = FontFamily.Monospace,
                                color = BrandCoral
                            )
                        }
                    }
                }
            }

            // Long-press Context Menu
            DropdownMenu(
                expanded = showMenu,
                onDismissRequest = { showMenu = false },
                modifier = Modifier.background(StudioCard)
            ) {
                DropdownMenuItem(
                    text = { Text("Copy", color = Color.White) },
                    onClick = {
                        onCopy()
                        showMenu = false
                    },
                    leadingIcon = { Icon(Icons.Default.ContentCopy, contentDescription = null, tint = Color.White) }
                )
                if (isUser) {
                    DropdownMenuItem(
                        text = { Text("Edit & Resend", color = Color.White) },
                        onClick = {
                            onEdit()
                            showMenu = false
                        },
                        leadingIcon = { Icon(Icons.Default.Edit, contentDescription = null, tint = Color.White) }
                    )
                }
                DropdownMenuItem(
                    text = { Text("Delete", color = Color(0xFFEF4444)) },
                    onClick = {
                        onDelete()
                        showMenu = false
                    },
                    leadingIcon = { Icon(Icons.Default.Delete, contentDescription = null, tint = Color(0xFFEF4444)) }
                )
            }
        }

        // Action icons / Latency pill below assistant response
        if (!isUser && message.text.isNotBlank()) {
            Row(
                modifier = Modifier.padding(top = 4.dp, start = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                val metrics = message.metrics
                if (metrics != null && metrics.tokensPerSecond > 0) {
                    Text(
                        text = "⚡ %.1f tok/s · %.0f ms".format(metrics.tokensPerSecond, metrics.totalDurationMs),
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF9CA3AF)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }

                IconButton(
                    onClick = onCopy,
                    modifier = Modifier.size(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ContentCopy,
                        contentDescription = "Copy",
                        tint = Color(0xFF6B7280),
                        modifier = Modifier.size(13.dp)
                    )
                }

                IconButton(
                    onClick = onRegenerate,
                    modifier = Modifier.size(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Regenerate",
                        tint = Color(0xFF6B7280),
                        modifier = Modifier.size(13.dp)
                    )
                }
            }
        }
    }
}

/**
 * Modern Input Bar matching reference screenshot:
 * Rounded dark container with:
 * Left: (+) media attachment, (📊) blue stats toggle button
 * Center: Type prompt... text field
 * Right: (🎤) voice input, (➤) Coral send button / (⏹) stop button
 */
@Composable
private fun ChatInputBar(
    inputPrompt: String,
    isGenerating: Boolean,
    isStatsVisible: Boolean,
    pendingAttachments: List<MediaAttachment>,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onToggleStats: () -> Unit,
    onAddAttachment: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onMicClick: () -> Unit
) {
    Surface(
        color = StudioBackground,
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 6.dp)
        ) {
            // Pending attachments preview chips
            if (pendingAttachments.isNotEmpty()) {
                LazyRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    items(pendingAttachments, key = { it.id }) { att ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(StudioCard)
                                .border(1.dp, StudioBorder, RoundedCornerShape(10.dp))
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = if (att.isAudio) Icons.Outlined.Mic else Icons.Outlined.Image,
                                    contentDescription = null,
                                    tint = if (att.isAudio) StatsBlue else BrandCoral,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(att.name, fontSize = 11.sp, color = Color.White)
                                Spacer(modifier = Modifier.width(6.dp))
                                Icon(
                                    imageVector = Icons.Default.Close,
                                    contentDescription = "Remove",
                                    tint = Color(0xFF9CA3AF),
                                    modifier = Modifier
                                        .size(12.dp)
                                        .clickable { onRemoveAttachment(att.id) }
                                )
                            }
                        }
                    }
                }
            }

            // Input container card
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(26.dp))
                    .background(StudioCard)
                    .border(1.dp, StudioBorder, RoundedCornerShape(26.dp))
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Column {
                    // Multi-line Text Field
                    OutlinedTextField(
                        value = inputPrompt,
                        onValueChange = onValueChange,
                        placeholder = {
                            Text(
                                text = "Type prompt...",
                                fontSize = 14.sp,
                                color = Color(0xFF6B7280)
                            )
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(54.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            focusedBorderColor = Color.Transparent,
                            unfocusedBorderColor = Color.Transparent,
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            cursorColor = BrandCoral
                        ),
                        maxLines = 4
                    )

                    // Bottom action controls row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Left: (+) and (📊) stats toggle button
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            // (+) Attachment button
                            Box(
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(CircleShape)
                                    .clickable(onClick = onAddAttachment)
                                    .background(StudioCardHighlight),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Add Attachment",
                                    tint = Color.White,
                                    modifier = Modifier.size(18.dp)
                                )
                            }

                            Spacer(modifier = Modifier.width(8.dp))

                            // (📊) Stats toggle button (Blue circular button)
                            Box(
                                modifier = Modifier
                                    .size(36.dp)
                                    .clip(CircleShape)
                                    .clickable(onClick = onToggleStats)
                                    .background(if (isStatsVisible) StatsBlue else StudioCardHighlight),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.BarChart,
                                    contentDescription = "Toggle Stats",
                                    tint = Color.White,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                        }

                        // Right: (🎤) Microphone & (➤) Send button / (⏹) Stop button
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            // (🎤) Microphone button
                            IconButton(
                                onClick = onMicClick,
                                modifier = Modifier.size(36.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Mic,
                                    contentDescription = "Voice Input",
                                    tint = Color(0xFFD1D5DB),
                                    modifier = Modifier.size(20.dp)
                                )
                            }

                            Spacer(modifier = Modifier.width(4.dp))

                            // (➤) Send button or (⏹) Stop button
                            if (isGenerating) {
                                Box(
                                    modifier = Modifier
                                        .size(40.dp)
                                        .clip(CircleShape)
                                        .clickable(onClick = onStop)
                                        .background(BrandCoralContainer)
                                        .border(1.5.dp, BrandCoral, CircleShape),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Stop,
                                        contentDescription = "Stop",
                                        tint = BrandCoral,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            } else {
                                val canSend = inputPrompt.isNotBlank()
                                Box(
                                    modifier = Modifier
                                        .size(40.dp)
                                        .clip(CircleShape)
                                        .clickable(enabled = canSend, onClick = onSend)
                                        .background(if (canSend) BrandCoral else Color(0xFF382321)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.AutoMirrored.Filled.Send,
                                        contentDescription = "Send",
                                        tint = if (canSend) Color.White else Color(0xFF7F6A68),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
