package ai.codesec.nexa.ui

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
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import androidx.compose.runtime.mutableIntStateOf
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
import ai.codesec.nexa.ui.components.ModelDetailsBottomSheet
import ai.codesec.nexa.ui.theme.BrandCoral
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.StatsBlue
import ai.codesec.nexa.ui.theme.StudioBackground
import ai.codesec.nexa.ui.theme.StudioBorder
import ai.codesec.nexa.ui.theme.StudioCard
import ai.codesec.nexa.ui.theme.StudioCardHighlight
import ai.codesec.nexa.ui.theme.StudioSurface
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.DiscoverModelItem
import ai.codesec.nexa.viewmodel.DownloadItem
import ai.codesec.nexa.viewmodel.DownloadStatus
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelManagerScreen(
    viewModel: ChatViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    var selectedTab by remember { mutableIntStateOf(0) } // 0: My Models, 1: Discover, 2: Downloads
    var modelToDelete by remember { mutableStateOf<File?>(null) }
    var searchQuery by remember { mutableStateOf("") }
    var selectedCategory by remember { mutableStateOf("All") }

    var selectedDetailFile by remember { mutableStateOf<File?>(null) }
    var selectedDetailDiscover by remember { mutableStateOf<DiscoverModelItem?>(null) }

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
        containerColor = StudioBackground,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Model Library",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Text(
                            text = "On-device weights & downloads",
                            fontSize = 11.sp,
                            color = Color(0xFF9CA3AF)
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                },
                actions = {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .clickable { filePickerLauncher.launch(arrayOf("*/*")) }
                            .background(StudioCard)
                            .border(1.dp, StudioBorder, RoundedCornerShape(12.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Add, contentDescription = null, tint = BrandCoral, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Import GGUF", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                    Spacer(modifier = Modifier.width(10.dp))
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = StudioBackground)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = innerPadding.calculateTopPadding())
        ) {
            // Tab row: My Models, Discover, Downloads
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                LibraryTabPill(
                    label = "My Models (${uiState.availableModels.size})",
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    modifier = Modifier.weight(1f)
                )
                LibraryTabPill(
                    label = "Discover",
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    modifier = Modifier.weight(1f)
                )
                val activeCount = uiState.activeDownloads.count { it.value.status == DownloadStatus.DOWNLOADING }
                val downloadLabel = if (activeCount > 0) "Downloads ($activeCount)" else "Downloads"
                LibraryTabPill(
                    label = downloadLabel,
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    modifier = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Tab Content
            when (selectedTab) {
                0 -> MyModelsTab(
                    uiState = uiState,
                    onLoadModel = { viewModel.loadModel(it) },
                    onUnloadModel = { viewModel.unloadModel() },
                    onDeleteModel = { modelToDelete = it },
                    onOpenDetails = { file ->
                        selectedDetailFile = file
                        selectedDetailDiscover = null
                    }
                )
                1 -> DiscoverModelsTab(
                    catalog = uiState.discoverCatalog,
                    installedNames = uiState.availableModels.map { it.name },
                    activeDownloads = uiState.activeDownloads,
                    searchQuery = searchQuery,
                    selectedCategory = selectedCategory,
                    onSearchChange = { searchQuery = it },
                    onCategoryChange = { selectedCategory = it },
                    onDownload = { viewModel.startModelDownload(it) },
                    onOpenDetails = { item ->
                        selectedDetailDiscover = item
                        selectedDetailFile = null
                    }
                )
                2 -> DownloadsTab(
                    downloads = uiState.activeDownloads.values.toList(),
                    onPause = { viewModel.pauseModelDownload(it) },
                    onCancel = { viewModel.cancelModelDownload(it) }
                )
            }
        }
    }

    // Delete confirmation dialog
    if (modelToDelete != null) {
        AlertDialog(
            onDismissRequest = { modelToDelete = null },
            containerColor = StudioCard,
            shape = RoundedCornerShape(20.dp),
            title = {
                Text("Delete Model", fontWeight = FontWeight.Bold, color = Color.White, fontSize = 16.sp)
            },
            text = {
                Text(
                    "Are you sure you want to delete ${modelToDelete?.name}? This will free up ${modelToDelete?.length()?.div(1024 * 1024)} MB of device storage.",
                    color = Color(0xFFD1D5DB),
                    fontSize = 13.sp
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        modelToDelete?.let { viewModel.deleteModel(it) }
                        modelToDelete = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF4444))
                ) {
                    Text("Delete", color = Color.White)
                }
            },
            dismissButton = {
                TextButton(onClick = { modelToDelete = null }) {
                    Text("Cancel", color = Color(0xFF9CA3AF))
                }
            }
        )
    }

    // Model details bottom sheet
    if (selectedDetailFile != null || selectedDetailDiscover != null) {
        ModelDetailsBottomSheet(
            file = selectedDetailFile,
            metadata = selectedDetailFile?.let { uiState.modelMetadataMap[it.name] },
            discoverItem = selectedDetailDiscover,
            isLoaded = selectedDetailFile?.name == uiState.loadedModelName,
            onLoadModel = { viewModel.loadModel(it) },
            onUnloadModel = { viewModel.unloadModel() },
            onDeleteModel = { modelToDelete = it },
            onDownloadModel = { viewModel.startModelDownload(it) },
            onDismiss = {
                selectedDetailFile = null
                selectedDetailDiscover = null
            }
        )
    }
}

@Composable
private fun LibraryTabPill(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .background(if (selected) BrandCoral else StudioCard)
            .border(1.dp, if (selected) BrandCoral else StudioBorder, RoundedCornerShape(12.dp))
            .padding(vertical = 8.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
            color = if (selected) Color.White else Color(0xFF9CA3AF)
        )
    }
}

/**
 * Tab 1: My Models (installed local weights)
 */
@Composable
private fun MyModelsTab(
    uiState: ai.codesec.nexa.viewmodel.ChatUiState,
    onLoadModel: (File) -> Unit,
    onUnloadModel: () -> Unit,
    onDeleteModel: (File) -> Unit,
    onOpenDetails: (File) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Storage summary card
        item {
            StorageSummaryCard(
                usedBytes = uiState.modelsStorageBytes,
                freeBytes = uiState.freeStorageBytes
            )
        }

        if (uiState.availableModels.isEmpty()) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 40.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(Icons.Default.Folder, contentDescription = null, tint = Color(0xFF6B7280), modifier = Modifier.size(44.dp))
                        Spacer(modifier = Modifier.height(10.dp))
                        Text("No GGUF models in storage", fontSize = 14.sp, color = Color(0xFF9CA3AF))
                    }
                }
            }
        } else {
            items(uiState.availableModels, key = { it.name }) { file ->
                val isLoaded = file.name == uiState.loadedModelName
                val meta = uiState.modelMetadataMap[file.name]
                val sizeMb = file.length() / (1024 * 1024)

                InstalledModelCard(
                    file = file,
                    sizeMb = sizeMb,
                    metadata = meta,
                    isLoaded = isLoaded,
                    onLoad = { onLoadModel(file) },
                    onUnload = onUnloadModel,
                    onDelete = { onDeleteModel(file) },
                    onDetails = { onOpenDetails(file) }
                )
            }
        }

        item {
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun StorageSummaryCard(
    usedBytes: Long,
    freeBytes: Long
) {
    val usedMb = usedBytes / (1024 * 1024)
    val freeGb = "%.1f GB".format(freeBytes / (1024.0 * 1024.0 * 1024.0))

    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = StudioCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.SdStorage, contentDescription = null, tint = StatsBlue, modifier = Modifier.size(24.dp))
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text("Device Storage", fontSize = 12.sp, color = Color(0xFF9CA3AF))
                    Text("$freeGb Free Headroom", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Color.White)
                }
            }
            Text(
                text = "${usedMb} MB in Models",
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                color = BrandCoral
            )
        }
    }
}

@Composable
private fun InstalledModelCard(
    file: File,
    sizeMb: Long,
    metadata: ModelMetadata?,
    isLoaded: Boolean,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onDelete: () -> Unit,
    onDetails: () -> Unit
) {
    val borderColor = if (isLoaded) BrandCoral else StudioBorder
    val containerColor = if (isLoaded) Color(0xFF241C1B) else StudioCard

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(containerColor)
            .border(1.dp, borderColor, RoundedCornerShape(16.dp))
            .padding(14.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = file.name,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    val arch = metadata?.architecture ?: "GGUF"
                    val quant = metadata?.quantization ?: "Q4_K_M"
                    Text(
                        text = "$arch · $quant · ${sizeMb} MB · 2048 ctx",
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF9CA3AF)
                    )
                }

                if (isLoaded) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color(0xFF381B18))
                            .border(1.dp, BrandCoral.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text("LOADED", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = BrandCoral)
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Action row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (isLoaded) {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .clickable(onClick = onUnload)
                                .background(StudioCardHighlight)
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Text("Unload", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = BrandCoral)
                        }
                    } else {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .clickable(onClick = onLoad)
                                .background(BrandCoral)
                                .padding(horizontal = 14.dp, vertical = 6.dp)
                        ) {
                            Text("Load Model", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color.White)
                        }
                    }

                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .clickable(onClick = onDetails)
                            .background(StudioCardHighlight)
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Text("Details", fontSize = 11.sp, color = Color(0xFFD1D5DB))
                    }
                }

                IconButton(
                    onClick = onDelete,
                    modifier = Modifier.size(30.dp)
                ) {
                    Icon(Icons.Default.Delete, contentDescription = "Delete", tint = Color(0xFFEF4444), modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

/**
 * Tab 2: Discover Models (Curated mobile catalog)
 */
@Composable
private fun DiscoverModelsTab(
    catalog: List<DiscoverModelItem>,
    installedNames: List<String>,
    activeDownloads: Map<String, DownloadItem>,
    searchQuery: String,
    selectedCategory: String,
    onSearchChange: (String) -> Unit,
    onCategoryChange: (String) -> Unit,
    onDownload: (DiscoverModelItem) -> Unit,
    onOpenDetails: (DiscoverModelItem) -> Unit
) {
    val categories = listOf("All", "Ultra-Fast", "Reasoning", "Balanced")

    val filtered = remember(catalog, searchQuery, selectedCategory) {
        catalog.filter { item ->
            val matchesCategory = selectedCategory == "All" || item.category.equals(selectedCategory, ignoreCase = true)
            val matchesSearch = searchQuery.isBlank() ||
                item.name.contains(searchQuery, ignoreCase = true) ||
                item.description.contains(searchQuery, ignoreCase = true)
            matchesCategory && matchesSearch
        }
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Search bar
        item {
            OutlinedTextField(
                value = searchQuery,
                onValueChange = onSearchChange,
                placeholder = { Text("Search on-device model catalog...", fontSize = 12.sp, color = Color(0xFF6B7280)) },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = Color(0xFF9CA3AF), modifier = Modifier.size(16.dp)) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp),
                shape = RoundedCornerShape(14.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = StudioCard,
                    unfocusedContainerColor = StudioCard,
                    focusedBorderColor = BrandCoral,
                    unfocusedBorderColor = StudioBorder,
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                ),
                singleLine = true
            )
        }

        // Category filter chips
        item {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                items(categories) { cat ->
                    val isSelected = cat == selectedCategory
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .clickable { onCategoryChange(cat) }
                            .background(if (isSelected) BrandCoral else StudioCard)
                            .border(1.dp, if (isSelected) BrandCoral else StudioBorder, RoundedCornerShape(10.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = cat,
                            fontSize = 11.sp,
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                            color = if (isSelected) Color.White else Color(0xFF9CA3AF)
                        )
                    }
                }
            }
        }

        // Discover cards
        items(filtered, key = { it.id }) { item ->
            val isDownloaded = installedNames.contains(item.filename)
            val activeDownload = activeDownloads[item.id]

            DiscoverModelCard(
                item = item,
                isDownloaded = isDownloaded,
                downloadItem = activeDownload,
                onDownload = { onDownload(item) },
                onDetails = { onOpenDetails(item) }
            )
        }

        item {
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun DiscoverModelCard(
    item: DiscoverModelItem,
    isDownloaded: Boolean,
    downloadItem: DownloadItem?,
    onDownload: () -> Unit,
    onDetails: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = StudioCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = item.name,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "${item.architecture} · ${item.parameters} · ${item.quantization} · ${item.downloadSizeMb} MB",
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = BrandCoral
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(StudioCardHighlight)
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(item.category, fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color(0xFFD1D5DB))
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = item.description,
                fontSize = 12.sp,
                lineHeight = 16.sp,
                color = Color(0xFF9CA3AF)
            )

            Spacer(modifier = Modifier.height(10.dp))

            // Action row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "RAM req: ~${item.ramRequiredMb} MB",
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF6B7280)
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .clickable(onClick = onDetails)
                            .background(StudioCardHighlight)
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Text("Details", fontSize = 11.sp, color = Color(0xFFD1D5DB))
                    }

                    if (isDownloaded) {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(Color(0xFF1E281F))
                                .padding(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("Installed ✓", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = MintSuccess)
                        }
                    } else if (downloadItem?.status == DownloadStatus.DOWNLOADING) {
                        val pct = (downloadItem.progress * 100).toInt()
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .background(StudioCardHighlight)
                                .padding(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text("Downloading $pct%", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = BrandCoral)
                        }
                    } else {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(10.dp))
                                .clickable(onClick = onDownload)
                                .background(BrandCoral)
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Download, contentDescription = null, tint = Color.White, modifier = Modifier.size(13.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Get", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color.White)
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Tab 3: Downloads Manager
 */
@Composable
private fun DownloadsTab(
    downloads: List<DownloadItem>,
    onPause: (String) -> Unit,
    onCancel: (String) -> Unit
) {
    if (downloads.isEmpty()) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(vertical = 40.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Default.Download, contentDescription = null, tint = Color(0xFF6B7280), modifier = Modifier.size(44.dp))
                Spacer(modifier = Modifier.height(10.dp))
                Text("No active or queued downloads", fontSize = 14.sp, color = Color(0xFF9CA3AF))
            }
        }
    } else {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            items(downloads, key = { it.id }) { item ->
                ActiveDownloadCard(
                    item = item,
                    onPause = { onPause(item.id) },
                    onCancel = { onCancel(item.id) }
                )
            }
        }
    }
}

@Composable
private fun ActiveDownloadCard(
    item: DownloadItem,
    onPause: () -> Unit,
    onCancel: () -> Unit
) {
    val downloadedMb = item.downloadedBytes / (1024 * 1024)
    val totalMb = item.totalBytes / (1024 * 1024)
    val speedMb = "%.1f MB/s".format(item.speedBytesPerSec / (1024.0 * 1024.0))

    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = StudioCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = item.modelName,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "Status: ${item.status.name} · $speedMb",
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = BrandCoral
                    )
                }

                Row {
                    if (item.status == DownloadStatus.DOWNLOADING) {
                        IconButton(onClick = onPause, modifier = Modifier.size(30.dp)) {
                            Icon(Icons.Default.Pause, contentDescription = "Pause", tint = Color.White, modifier = Modifier.size(16.dp))
                        }
                    }
                    IconButton(onClick = onCancel, modifier = Modifier.size(30.dp)) {
                        Icon(Icons.Default.Cancel, contentDescription = "Cancel", tint = Color(0xFFEF4444), modifier = Modifier.size(16.dp))
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Progress bar
            LinearProgressIndicator(
                progress = { item.progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp)),
                color = BrandCoral,
                trackColor = StudioCardHighlight
            )

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "$downloadedMb MB / $totalMb MB (${(item.progress * 100).toInt()}%)",
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF9CA3AF)
                )
                if (item.speedBytesPerSec > 0 && item.totalBytes > item.downloadedBytes) {
                    val remainingSec = ((item.totalBytes - item.downloadedBytes) / item.speedBytesPerSec).toInt()
                    Text(
                        text = "${remainingSec}s remaining",
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF6B7280)
                    )
                }
            }
        }
    }
}
