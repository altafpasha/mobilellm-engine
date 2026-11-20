package ai.codesec.nexa.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.mobilellm.models.ModelMetadata
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CrimsonError
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.CyberLimeContainer
import ai.codesec.nexa.ui.theme.CyberLimeDark
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.ObsidianBackground
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard
import ai.codesec.nexa.ui.theme.ObsidianCardHighlight
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.NexaScreen
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelManagerScreen(
    viewModel: ChatViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    var modelToDelete by remember { mutableStateOf<File?>(null) }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { viewModel.importModel(it, context.contentResolver) }
    }

    LaunchedEffect(Unit) {
        viewModel.refreshModelsList()
        viewModel.refreshSystemStats()
    }

    Scaffold(
        containerColor = ObsidianBackground,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Model Laboratory",
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp,
                            color = Color(0xFFF1F5F3)
                        )
                        Text(
                            text = "Local GGUF Weights & Storage",
                            fontSize = 11.sp,
                            color = Color(0xFF94A39D)
                        )
                    }
                },
                navigationIcon = {
                    IconButton(
                        onClick = onBack,
                        modifier = Modifier
                            .padding(8.dp)
                            .size(38.dp)
                            .background(ObsidianCard, CircleShape)
                            .border(1.dp, ObsidianBorder, CircleShape)
                    ) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = { filePickerLauncher.launch(arrayOf("*/*")) },
                        modifier = Modifier
                            .padding(8.dp)
                            .size(38.dp)
                            .background(CyberLime, CircleShape)
                    ) {
                        Icon(
                            Icons.Default.Add,
                            contentDescription = "Import Model",
                            tint = Color(0xFF0D1110),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ObsidianBackground
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp, vertical = 6.dp)
        ) {
            if (uiState.importProgress != null) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = CyberLime,
                    trackColor = ObsidianCardHighlight
                )
            }

            // Storage Overview Card
            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .background(CyberLimeContainer, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.SdStorage,
                                contentDescription = null,
                                tint = CyberLime,
                                modifier = Modifier.size(22.dp)
                            )
                        }
                        Spacer(modifier = Modifier.width(14.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "%.1f MB Models Installed".format(uiState.modelsStorageBytes / (1024.0 * 1024.0)),
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp,
                                color = Color.White
                            )
                            Text(
                                text = "%.1f GB Free Storage · %d Model(s)".format(
                                    uiState.freeStorageBytes / (1024.0 * 1024.0 * 1024.0),
                                    uiState.availableModels.size
                                ),
                                fontSize = 11.sp,
                                color = Color(0xFF94A39D)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    val usedRatio = if (uiState.modelsStorageBytes + uiState.freeStorageBytes > 0) {
                        (uiState.modelsStorageBytes.toFloat() / (uiState.modelsStorageBytes + uiState.freeStorageBytes)).coerceIn(0.02f, 1f)
                    } else 0.05f

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(ObsidianBackground)
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(usedRatio)
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(Brush.horizontalGradient(listOf(CyberLimeDark, CyberLime)))
                        )
                    }
                }
            }

            // Memory Warning Banner if system RAM is low
            if (uiState.isLowMemory || (uiState.systemAvailRamMb > 0 && uiState.systemAvailRamMb < 500)) {
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    color = CrimsonError.copy(alpha = 0.15f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, CrimsonError.copy(alpha = 0.4f)),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 6.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Warning, contentDescription = null, tint = CrimsonError, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "Low RAM Warning: Only ${uiState.systemAvailRamMb} MB available. Large models may be killed by Android LMK.",
                            color = CrimsonError,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            if (uiState.availableModels.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            "No GGUF Models Found",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color.White
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            "Tap + to import GGUF model files to device storage",
                            fontSize = 12.sp,
                            color = Color(0xFF94A39D)
                        )
                        Spacer(modifier = Modifier.height(18.dp))
                        Button(
                            onClick = { filePickerLauncher.launch(arrayOf("*/*")) },
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = CyberLime,
                                contentColor = Color(0xFF0D1110)
                            )
                        ) {
                            Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Import Model File", fontWeight = FontWeight.Bold)
                        }
                        Spacer(modifier = Modifier.height(18.dp))
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 8.dp),
                            shape = RoundedCornerShape(14.dp),
                            colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                            border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder)
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Text("Push weights via ADB:", fontWeight = FontWeight.SemiBold, fontSize = 12.sp, color = CyberLime)
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    "adb push <model>.gguf /sdcard/Android/data/ai.codesec.nexa.debug/files/models/",
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp,
                                    color = Color(0xFF94A39D)
                                )
                            }
                        }
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(uiState.availableModels, key = { it.absolutePath }) { file ->
                        val metadata = uiState.modelMetadataMap[file.name]
                        val isActive = file.name == uiState.loadedModelName && uiState.isModelLoaded

                        ObsidianModelCard(
                            file = file,
                            metadata = metadata,
                            isActive = isActive,
                            availRamMb = uiState.systemAvailRamMb,
                            onLoad = {
                                viewModel.loadModel(file)
                                viewModel.setScreen(NexaScreen.CHAT)
                            },
                            onUnload = { viewModel.unloadModel() },
                            onDelete = { modelToDelete = file }
                        )
                    }
                    item {
                        Spacer(modifier = Modifier.height(40.dp))
                    }
                }
            }
        }
    }

    modelToDelete?.let { file ->
        AlertDialog(
            onDismissRequest = { modelToDelete = null },
            containerColor = ObsidianCard,
            title = { Text("Delete Model?", color = Color.White, fontWeight = FontWeight.Bold) },
            text = {
                Text(
                    "Are you sure you want to delete '${file.name}'? This will free %.1f MB of storage.".format(file.length() / (1024.0 * 1024.0)),
                    color = Color(0xFF94A39D)
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.deleteModel(file)
                        modelToDelete = null
                    },
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = CrimsonError)
                ) {
                    Text("Delete", fontWeight = FontWeight.Bold, color = Color.White)
                }
            },
            dismissButton = {
                TextButton(onClick = { modelToDelete = null }) {
                    Text("Cancel", color = Color(0xFF94A39D))
                }
            }
        )
    }
}

@Composable
private fun ObsidianModelCard(
    file: File,
    metadata: ModelMetadata?,
    isActive: Boolean,
    availRamMb: Long,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onDelete: () -> Unit
) {
    val estimatedRamMb = (file.length() * 1.2) / (1024 * 1024)
    val memoryRisk = availRamMb > 0 && estimatedRamMb > (availRamMb * 0.7)

    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isActive) ObsidianCardHighlight else ObsidianCard
        ),
        border = androidx.compose.foundation.BorderStroke(
            1.dp,
            if (isActive) CyberLime else ObsidianBorder
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = file.name,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        color = Color.White
                    )
                    Text(
                        text = "%.1f MB on disk".format(file.length() / (1024.0 * 1024.0)),
                        fontSize = 11.sp,
                        color = Color(0xFF94A39D)
                    )
                }

                if (isActive) {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = CyberLime
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Icon(
                                Icons.Default.CheckCircle,
                                contentDescription = null,
                                tint = Color(0xFF0D1110),
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "ACTIVE",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.ExtraBold,
                                color = Color(0xFF0D1110)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Metadata Badges
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                ObsidianBadgePill(label = metadata?.architecture?.uppercase() ?: "GGUF", color = CyberLime)
                ObsidianBadgePill(label = metadata?.quantization ?: "Q4_K_M", color = ColdBlue)
                ObsidianBadgePill(label = "${metadata?.contextLength ?: 2048} ctx", color = Color(0xFF94A39D))
                if ((metadata?.blockCount ?: 0) > 0) {
                    ObsidianBadgePill(label = "${metadata?.blockCount}L", color = MintSuccess)
                }
            }

            if (metadata?.sha256?.isNotEmpty() == true) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "SHA256: ${metadata.sha256.take(12)}...",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF6B7280)
                )
            }

            if (memoryRisk && !isActive) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "⚠️ Estimated RAM (~%.0f MB) exceeds recommended headroom".format(estimatedRamMb),
                    fontSize = 11.sp,
                    color = Color(0xFFF59E0B),
                    fontWeight = FontWeight.Medium
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = onDelete,
                    modifier = Modifier
                        .size(36.dp)
                        .background(ObsidianBackground, CircleShape)
                        .border(1.dp, ObsidianBorder, CircleShape)
                ) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Delete Model",
                        tint = CrimsonError,
                        modifier = Modifier.size(16.dp)
                    )
                }

                if (isActive) {
                    Button(
                        onClick = onUnload,
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = ObsidianCardHighlight,
                            contentColor = Color.White
                        ),
                        modifier = Modifier.border(1.dp, ObsidianBorder, RoundedCornerShape(12.dp))
                    ) {
                        Text("Unload Weights", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                    }
                } else {
                    Button(
                        onClick = onLoad,
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = CyberLime,
                            contentColor = Color(0xFF0D1110)
                        )
                    ) {
                        Text("Load to Engine", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

@Composable
private fun ObsidianBadgePill(label: String, color: Color) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = ObsidianBackground,
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder)
    ) {
        Text(
            text = label,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            color = color,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }
}
