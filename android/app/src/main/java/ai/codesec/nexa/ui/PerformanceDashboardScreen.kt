package ai.codesec.nexa.ui

import android.os.Build
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CrimsonError
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.CyberLimeDark
import ai.codesec.nexa.ui.theme.CyberLimeContainer
import ai.codesec.nexa.ui.theme.CyberLimeGlow
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.ObsidianBackground
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard
import ai.codesec.nexa.ui.theme.ObsidianCardHighlight
import ai.codesec.nexa.ui.theme.ObsidianSurface
import ai.codesec.nexa.viewmodel.BenchmarkRun
import ai.codesec.nexa.viewmodel.ChatViewModel
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PerformanceDashboardScreen(
    viewModel: ChatViewModel,
    onBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val metrics = uiState.lastMetrics
    val maxCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
    var threadSliderValue by remember(uiState.threadCount) { mutableFloatStateOf(uiState.threadCount.toFloat()) }

    LaunchedEffect(Unit) {
        viewModel.refreshSystemStats()
    }

    Scaffold(
        containerColor = ObsidianBackground,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Model Lab Telemetry",
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp,
                            color = Color(0xFFF1F5F3)
                        )
                        Text(
                            text = "Hardware Profiling & Benchmarks",
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
                        onClick = { viewModel.refreshSystemStats() },
                        modifier = Modifier
                            .padding(8.dp)
                            .size(38.dp)
                            .background(ObsidianCard, CircleShape)
                            .border(1.dp, ObsidianBorder, CircleShape)
                    ) {
                        Icon(
                            Icons.Default.Refresh,
                            contentDescription = "Refresh Telemetry",
                            tint = CyberLime,
                            modifier = Modifier.size(18.dp)
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
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            // ==========================================
            // HERO ARC GAUGE CARD: INFERENCE VELOCITY
            // ==========================================
            ThroughputArcGaugeCard(
                tokPerSec = metrics?.tokensPerSecond ?: 0.0,
                isGenerating = uiState.isGenerating,
                totalTokens = metrics?.completionTokens ?: 0,
                activeBackend = uiState.preferredBackend
            )

            // ==========================================
            // SECTION A: INFERENCE KPIS (2x2 Grid)
            // ==========================================
            SectionTitle(title = "INFERENCE VELOCITY & LATENCY")

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                TelemetryMetricCard(
                    title = "Prefill Rate",
                    value = if (metrics != null && metrics.promptTokensPerSecond > 0)
                        "%.1f tok/s".format(metrics.promptTokensPerSecond)
                    else "N/A",
                    subtitle = "Prompt ingest rate",
                    accentColor = CyberLime,
                    modifier = Modifier.weight(1f)
                )
                TelemetryMetricCard(
                    title = "TTFT",
                    value = if (metrics != null && metrics.timeToFirstTokenMs > 0)
                        "%.0f ms".format(metrics.timeToFirstTokenMs)
                    else "N/A",
                    subtitle = "Queue + prompt prefill",
                    accentColor = ColdBlue,
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                TelemetryMetricCard(
                    title = "Prefill Exec",
                    value = if (metrics != null && metrics.prefillTimeMs > 0)
                        "%.0f ms".format(metrics.prefillTimeMs)
                    else "N/A",
                    subtitle = "Compute evaluation",
                    accentColor = Color(0xFFF1F5F3),
                    modifier = Modifier.weight(1f)
                )
                TelemetryMetricCard(
                    title = "Decode Exec",
                    value = if (metrics != null && metrics.decodeTimeMs > 0)
                        "%.0f ms".format(metrics.decodeTimeMs)
                    else "N/A",
                    subtitle = "Window + streaming",
                    accentColor = Color(0xFFF1F5F3),
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                TelemetryMetricCard(
                    title = "Total Latency",
                    value = if (metrics != null && metrics.totalDurationMs > 0)
                        "%.2f s".format(metrics.totalDurationMs / 1000.0)
                    else "N/A",
                    subtitle = "Enqueue to complete",
                    accentColor = ColdBlue,
                    modifier = Modifier.weight(1f)
                )
                TelemetryMetricCard(
                    title = "Evaluated",
                    value = if (metrics != null)
                        "${metrics.promptTokens} in / ${metrics.completionTokens} out"
                    else "N/A",
                    subtitle = "Context / generated",
                    accentColor = MintSuccess,
                    modifier = Modifier.weight(1f)
                )
            }

            // ==========================================
            // SECTION B: MEMORY & RESIDENT FOOTPRINT
            // ==========================================
            SectionTitle(title = "PROCESS & PHYSICAL RAM FOOTPRINT")

            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(32.dp)
                                    .background(CyberLimeContainer, CircleShape),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    Icons.Default.Memory,
                                    contentDescription = null,
                                    tint = CyberLime,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                "System & Process Memory",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color.White
                            )
                        }

                        val ramPct = if (uiState.systemTotalRamMb > 0) {
                            (((uiState.systemTotalRamMb - uiState.systemAvailRamMb).toFloat() / uiState.systemTotalRamMb) * 100).toInt()
                        } else 0
                        Surface(
                            shape = RoundedCornerShape(12.dp),
                            color = if (ramPct > 85) CrimsonError.copy(alpha = 0.2f) else CyberLimeContainer
                        ) {
                            Text(
                                text = "$ramPct% Utilized",
                                color = if (ramPct > 85) CrimsonError else CyberLime,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    val usedRamPercent = if (uiState.systemTotalRamMb > 0) {
                        ((uiState.systemTotalRamMb - uiState.systemAvailRamMb).toFloat() / uiState.systemTotalRamMb)
                    } else 0f

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            "Device Physical RAM",
                            fontSize = 12.sp,
                            color = Color(0xFF94A39D)
                        )
                        Text(
                            "${uiState.systemAvailRamMb} MB free / ${uiState.systemTotalRamMb} MB",
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            color = Color(0xFFF1F5F3)
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp))
                            .background(ObsidianBackground)
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(usedRamPercent.coerceIn(0f, 1f))
                                .height(8.dp)
                                .clip(RoundedCornerShape(4.dp))
                                .background(
                                    Brush.horizontalGradient(
                                        listOf(CyberLimeDark, if (usedRamPercent > 0.85f) CrimsonError else CyberLime)
                                    )
                                )
                        )
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    val displayRss = if (uiState.nativeRssMb > 0) "%.1f MB".format(uiState.nativeRssMb) else "N/A"
                    val displayPeakRss = if (uiState.nativePeakRssMb > 0)
                        "%.1f MB".format(uiState.nativePeakRssMb)
                    else if (metrics != null && metrics.peakRssMb > 0)
                        "%.1f MB".format(metrics.peakRssMb)
                    else "N/A"
                    val displayVmSize = if (uiState.nativeVmSizeMb > 0) "%.1f MB".format(uiState.nativeVmSizeMb) else "N/A"
                    val displayModelMem = if (uiState.modelMemoryMb > 0) "%.1f MB".format(uiState.modelMemoryMb) else "Internal (llama.cpp)"

                    ObsidianDashboardRow(label = "Native Process RSS", value = displayRss, valueColor = CyberLime)
                    ObsidianDashboardRow(label = "Native RSS Peak (HWM)", value = displayPeakRss, valueColor = Color.White)
                    ObsidianDashboardRow(label = "Virtual Memory (VmSize)", value = displayVmSize)
                    ObsidianDashboardRow(label = "JVM Heap Allocated", value = "%.1f MB".format(uiState.jvmHeapMb))
                    ObsidianDashboardRow(label = "Model Memory Map", value = displayModelMem)

                    val kvStatusText = if (uiState.isContextAvailable && uiState.contextCapacity > 0) {
                        "${uiState.usedContextTokens} / ${uiState.contextCapacity} tok (${uiState.remainingContextTokens} free)"
                    } else if (uiState.isModelLoaded) {
                        "Managed dynamically (llama.cpp)"
                    } else {
                        "No active session"
                    }
                    ObsidianDashboardRow(label = "KV-Cache Capacity", value = kvStatusText, valueColor = ColdBlue)

                    ObsidianDashboardRow(
                        label = "OS Low Memory Warning",
                        value = if (uiState.isLowMemory) "ACTIVE (Critical)" else "Normal",
                        valueColor = if (uiState.isLowMemory) CrimsonError else MintSuccess
                    )
                }
            }

            // ==========================================
            // SECTION C: HARDWARE & THREAD CONFIG
            // ==========================================
            SectionTitle(title = "CPU THREAD TUNING & THERMALS")

            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(32.dp)
                                    .background(ColdBlue.copy(alpha = 0.15f), CircleShape),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    Icons.Default.Tune,
                                    contentDescription = null,
                                    tint = ColdBlue,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                "Execution Threads",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color.White
                            )
                        }

                        Text(
                            "${threadSliderValue.roundToInt()} Threads",
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 14.sp,
                            color = CyberLime
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Slider(
                        value = threadSliderValue,
                        onValueChange = { threadSliderValue = it },
                        onValueChangeFinished = {
                            viewModel.updateThreads(threadSliderValue.roundToInt())
                        },
                        valueRange = 1f..maxCores.toFloat(),
                        steps = (maxCores - 2).coerceAtLeast(0),
                        colors = SliderDefaults.colors(
                            thumbColor = CyberLime,
                            activeTrackColor = CyberLime,
                            inactiveTrackColor = ObsidianCardHighlight
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )

                    // Quick thread preset pills
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(2, 4, 6, 8).filter { it <= maxCores }.forEach { threads ->
                            val isSelected = uiState.threadCount == threads
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(if (isSelected) CyberLime else ObsidianCardHighlight)
                                    .clickable {
                                        threadSliderValue = threads.toFloat()
                                        viewModel.updateThreads(threads)
                                    }
                                    .padding(vertical = 8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "${threads}T",
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = FontFamily.Monospace,
                                    color = if (isSelected) Color(0xFF0D1110) else Color(0xFF94A39D)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    val backendLabel = when (uiState.preferredBackend) {
                        BackendType.VULKAN -> "Vulkan GPU (Adreno 740/750)"
                        BackendType.CPU_NEON -> "CPU ARM NEON (128-bit)"
                        else -> "Auto Backend"
                    }
                    val accelLabel = when (uiState.preferredBackend) {
                        BackendType.VULKAN -> "Direct SPIR-V / mul_mat_vec"
                        else -> "128-bit Vector SIMD"
                    }
                    val gpuLayerLabel = if (uiState.gpuLayers >= 0) "${uiState.gpuLayers} / 28 Layers" else "Auto Offload"

                    ObsidianDashboardRow(label = "Hardware Cores", value = "$maxCores Cores Available")
                    ObsidianDashboardRow(label = "Selected Backend", value = backendLabel, valueColor = CyberLime)
                    ObsidianDashboardRow(label = "Acceleration Engine", value = accelLabel)
                    ObsidianDashboardRow(label = "GPU Offload Target", value = gpuLayerLabel)
                    ObsidianDashboardRow(
                        label = "Device Thermal State",
                        value = uiState.thermalStatusName,
                        valueColor = if (uiState.isThermalThrottling) CrimsonError else MintSuccess
                    )

                    if (uiState.isThermalThrottling) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(CrimsonError.copy(alpha = 0.15f), RoundedCornerShape(10.dp))
                                .border(1.dp, CrimsonError.copy(alpha = 0.4f), RoundedCornerShape(10.dp))
                                .padding(10.dp)
                        ) {
                            Icon(Icons.Default.Warning, contentDescription = null, tint = CrimsonError, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Device is thermally throttled. CPU/GPU clocks are reduced to manage temperature.",
                                fontSize = 11.sp,
                                color = CrimsonError
                            )
                        }
                    }
                }
            }

            // ==========================================
            // SECTION D: AUTOMATED BENCHMARK SUITE
            // ==========================================
            SectionTitle(title = "AUTOMATED BENCHMARK SUITE")

            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Thermal-Aware Benchmark",
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp,
                                color = Color.White
                            )
                            Text(
                                if (uiState.preferredBackend == BackendType.VULKAN)
                                    "Safe GPU benchmark (1 Warmup + 1 Measured 32-tok run)"
                                else
                                    "Standard: 32 tokens | Extended: Multi-Thread Sweep",
                                fontSize = 11.sp,
                                color = Color(0xFF94A39D)
                            )
                        }

                        val isDeviceSeverelyHot = uiState.thermalStatus >= 3

                        if (uiState.isBenchmarking) {
                            Button(
                                onClick = { viewModel.stopBenchmark() },
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = CrimsonError,
                                    contentColor = Color.White
                                )
                            ) {
                                Text("Stop", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                            }
                        } else {
                            Button(
                                onClick = { viewModel.runBenchmark(extended = false) },
                                enabled = uiState.isModelLoaded && !uiState.isGenerating && !isDeviceSeverelyHot,
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = CyberLime,
                                    contentColor = Color(0xFF0D1110),
                                    disabledContainerColor = ObsidianCardHighlight,
                                    disabledContentColor = Color(0xFF6B7280)
                                )
                            ) {
                                if (isDeviceSeverelyHot) {
                                    Icon(Icons.Default.Warning, contentDescription = null, tint = CrimsonError, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Hot (Cooldown)", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                } else {
                                    Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Run (32T)", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    if (uiState.benchmarkHistory.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(ObsidianBackground, RoundedCornerShape(14.dp))
                                .border(1.dp, ObsidianBorder, RoundedCornerShape(14.dp))
                                .padding(vertical = 18.dp, horizontal = 12.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = if (uiState.isModelLoaded)
                                    "Tap 'Run Suite' to benchmark throughput across CPU thread configurations."
                                else
                                    "Load a model in the Lab to run multi-thread benchmarks.",
                                fontSize = 12.sp,
                                color = Color(0xFF94A39D)
                            )
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            uiState.benchmarkHistory.takeLast(6).reversed().forEach { run ->
                                ObsidianBenchmarkCard(run)
                            }
                        }
                    }
                }
            }

            // ==========================================
            // SECTION E: ENVIRONMENT CONTRACTS
            // ==========================================
            SectionTitle(title = "DEVICE & ENVIRONMENT INFO")

            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    ObsidianDashboardRow(label = "Hardware Target", value = "${Build.MANUFACTURER.uppercase()} ${Build.MODEL}", valueColor = Color.White)
                    ObsidianDashboardRow(label = "Android Release", value = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                    ObsidianDashboardRow(label = "MobileLLM Core", value = uiState.engineVersion, valueColor = CyberLime)
                    ObsidianDashboardRow(label = "llama.cpp Baseline", value = "b4260 (NEON/Vulkan)")
                    ObsidianDashboardRow(label = "Active Model", value = uiState.loadedModelName ?: "None", valueColor = ColdBlue)
                    ObsidianDashboardRow(label = "Security Posture", value = "100% Offline / Zero Remote Telemetry", valueColor = MintSuccess)
                }
            }

            // ==========================================
            // SECTION F: STRUCTURED ENGINE DIAGNOSTICS
            // ==========================================
            SectionTitle(title = "STRUCTURED ENGINE DIAGNOSTICS")

            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = ObsidianCard),
                border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Text(
                        text = "PER-GENERATION RECORD",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.ExtraBold,
                        color = CyberLime
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    if (uiState.generationDiagnostics.isBlank()) {
                        Text(
                            text = "Run an inference in the Lab to capture structured diagnostics: backend " +
                                "identity, GPU offload evidence, decomposed stage timings, KV usage, and " +
                                "per-token latency percentiles.",
                            fontSize = 12.sp,
                            color = Color(0xFF94A39D)
                        )
                    } else {
                        Text(
                            text = uiState.generationDiagnostics,
                            fontSize = 10.sp,
                            lineHeight = 15.sp,
                            fontFamily = FontFamily.Monospace,
                            color = Color(0xFF94A39D)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(30.dp))
        }
    }
}

/**
 * Circular Arc Throughput Gauge inspired by modern AI telemetry dashboards.
 */
@Composable
private fun ThroughputArcGaugeCard(
    tokPerSec: Double,
    isGenerating: Boolean,
    totalTokens: Int,
    activeBackend: BackendType
) {
    Card(
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "INFERENCE THROUGHPUT",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = 1.2.sp,
                    color = Color(0xFF94A39D)
                )

                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (activeBackend == BackendType.VULKAN) ColdBlue.copy(alpha = 0.15f) else CyberLimeContainer
                ) {
                    Text(
                        text = if (activeBackend == BackendType.VULKAN) "VULKAN GPU" else "ARM NEON",
                        color = if (activeBackend == BackendType.VULKAN) ColdBlue else CyberLime,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.ExtraBold,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Gauge Canvas Arc
            val animatedToks by animateFloatAsState(
                targetValue = tokPerSec.toFloat(),
                animationSpec = tween(durationMillis = 600),
                label = "toks"
            )

            // Normalized to 0..50 tok/s scale for arc representation
            val progress = (animatedToks / 45f).coerceIn(0.04f, 1f)

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(190.dp)
                    .padding(8.dp)
            ) {
                Canvas(modifier = Modifier.size(170.dp)) {
                    val strokeWidth = 14.dp.toPx()
                    val arcSize = size.width - strokeWidth
                    val topLeft = Offset(strokeWidth / 2, strokeWidth / 2)

                    // Background track: 220 degree arc from 160 deg to 380 deg
                    drawArc(
                        color = Color(0xFF222C28),
                        startAngle = 140f,
                        sweepAngle = 260f,
                        useCenter = false,
                        topLeft = topLeft,
                        size = Size(arcSize, arcSize),
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
                    )

                    // Foreground gradient arc
                    val sweep = progress * 260f
                    drawArc(
                        brush = Brush.sweepGradient(
                            listOf(
                                CyberLimeDark,
                                CyberLime,
                                MintSuccess
                            )
                        ),
                        startAngle = 140f,
                        sweepAngle = sweep,
                        useCenter = false,
                        topLeft = topLeft,
                        size = Size(arcSize, arcSize),
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
                    )
                }

                // Center Content
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(top = 10.dp)
                ) {
                    Text(
                        text = if (tokPerSec > 0) "%.1f".format(tokPerSec) else "0.0",
                        fontSize = 38.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = FontFamily.Monospace,
                        color = Color.White
                    )
                    Text(
                        text = "TOK / SEC",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        color = CyberLime
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (isGenerating)
                    "Generating tokens via on-device LLM..."
                else if (totalTokens > 0)
                    "Last generation completed $totalTokens tokens"
                else
                    "Run prompt inference to benchmark throughput",
                fontSize = 12.sp,
                color = Color(0xFF94A39D)
            )
        }
    }
}

@Composable
private fun SectionTitle(title: String) {
    Text(
        text = title,
        fontSize = 11.sp,
        fontWeight = FontWeight.ExtraBold,
        letterSpacing = 1.2.sp,
        color = CyberLime,
        modifier = Modifier.padding(top = 8.dp, bottom = 2.dp)
    )
}

@Composable
private fun TelemetryMetricCard(
    title: String,
    value: String,
    subtitle: String,
    accentColor: Color,
    modifier: Modifier = Modifier
) {
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = modifier
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = title.uppercase(),
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.8.sp,
                color = Color(0xFF94A39D)
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = value,
                fontSize = 18.sp,
                fontWeight = FontWeight.ExtraBold,
                fontFamily = FontFamily.Monospace,
                color = accentColor
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = subtitle,
                fontSize = 10.sp,
                color = Color(0xFF6B7280)
            )
        }
    }
}

@Composable
private fun ObsidianBenchmarkCard(run: BenchmarkRun) {
    Card(
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = ObsidianBackground),
        border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(CyberLime, CircleShape)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${run.threads} CPU Threads",
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        color = Color.White
                    )
                }

                Text(
                    text = "%.1f tok/s".format(run.tokensPerSecond),
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 14.sp,
                    color = CyberLime
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Prefill: %.1f tps · TTFT: %.0f ms".format(run.promptTokensPerSecond, run.timeToFirstTokenMs),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF94A39D)
                )
                Text(
                    text = "Peak: %.1f MB".format(run.peakRssMb),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF94A39D)
                )
            }
        }
    }
}

@Composable
private fun ObsidianDashboardRow(
    label: String,
    value: String,
    valueColor: Color = Color(0xFFF1F5F3)
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            color = Color(0xFF94A39D)
        )
        Text(
            text = value,
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}
