package ai.codesec.nexa.ui

import android.os.Build
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
        topBar = {
            TopAppBar(
                title = { Text("Performance & Telemetry", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refreshSystemStats() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh Telemetry")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {

            // ==========================================
            // SECTION A: INFERENCE PERFORMANCE
            // ==========================================
            SectionHeader(title = "A. INFERENCE PERFORMANCE")

            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "DECODE GENERATION THROUGHPUT",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = if (metrics != null && metrics.tokensPerSecond > 0)
                            "%.1f tok/s".format(metrics.tokensPerSecond)
                        else
                            "N/A",
                        fontSize = 36.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (uiState.isGenerating)
                            "Evaluating Prompt & Streaming Tokens..."
                        else if (metrics != null && metrics.completionTokens > 0)
                            "Measured over ${metrics.completionTokens} output tokens"
                        else
                            "Run inference to measure decode speed",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Prefill Speed",
                    value = if (metrics != null && metrics.promptTokensPerSecond > 0)
                        "%.1f tok/s".format(metrics.promptTokensPerSecond)
                    else "N/A",
                    subtitle = "Prompt ingest rate",
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "TTFT",
                    value = if (metrics != null && metrics.timeToFirstTokenMs > 0)
                        "%.0f ms".format(metrics.timeToFirstTokenMs)
                    else "N/A",
                    subtitle = "Time to first token",
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Total Latency",
                    value = if (metrics != null && metrics.totalDurationMs > 0)
                        "%.2f s".format(metrics.totalDurationMs / 1000.0)
                    else "N/A",
                    subtitle = "Enqueue to complete",
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Tokens Evaluated",
                    value = if (metrics != null)
                        "${metrics.promptTokens} in / ${metrics.completionTokens} out"
                    else "N/A",
                    subtitle = "Context / generated",
                    modifier = Modifier.weight(1f)
                )
            }

            // ==========================================
            // SECTION B: MEMORY MONITORING
            // ==========================================
            SectionHeader(title = "B. MEMORY MONITORING")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Memory, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Process & System Memory", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    val usedRamPercent = if (uiState.systemTotalRamMb > 0) {
                        ((uiState.systemTotalRamMb - uiState.systemAvailRamMb).toFloat() / uiState.systemTotalRamMb)
                    } else 0f

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Device Physical RAM", style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${uiState.systemAvailRamMb} MB free / ${uiState.systemTotalRamMb} MB",
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(
                        progress = usedRamPercent.coerceIn(0f, 1f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp),
                        color = if (usedRamPercent > 0.85f) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    val displayRss = if (uiState.nativeRssMb > 0)
                        "%.1f MB".format(uiState.nativeRssMb)
                    else "N/A"

                    val displayPeakRss = if (uiState.nativePeakRssMb > 0)
                        "%.1f MB".format(uiState.nativePeakRssMb)
                    else if (metrics != null && metrics.peakRssMb > 0)
                        "%.1f MB".format(metrics.peakRssMb)
                    else "N/A"

                    val displayVmSize = if (uiState.nativeVmSizeMb > 0)
                        "%.1f MB".format(uiState.nativeVmSizeMb)
                    else "N/A"

                    val displayModelMem = if (uiState.modelMemoryMb > 0)
                        "%.1f MB".format(uiState.modelMemoryMb)
                    else "Internal to llama.cpp"

                    DashboardRow(label = "Native Process RSS", value = displayRss)
                    DashboardRow(label = "Native RSS (Peak HWM)", value = displayPeakRss)
                    DashboardRow(label = "Virtual Memory (VmSize)", value = displayVmSize)
                    DashboardRow(label = "JVM Heap Allocated", value = "%.1f MB".format(uiState.jvmHeapMb))
                    DashboardRow(label = "Model Weights Allocation", value = displayModelMem)

                    val kvStatusText = if (uiState.isContextAvailable && uiState.contextCapacity > 0) {
                        "${uiState.usedContextTokens} / ${uiState.contextCapacity} tokens (${uiState.remainingContextTokens} free)"
                    } else if (uiState.isModelLoaded) {
                        "Managed internally (llama.cpp)"
                    } else {
                        "No active session"
                    }
                    DashboardRow(label = "KV-Cache Capacity", value = kvStatusText)

                    DashboardRow(
                        label = "OS Low Memory Warning",
                        value = if (uiState.isLowMemory) "ACTIVE (Low RAM)" else "Normal",
                        valueColor = if (uiState.isLowMemory) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Note: JVM Heap and Native Process RSS are monitored independently via /proc/self/status and OS Runtime.",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.75f)
                    )
                }
            }

            // ==========================================
            // SECTION C: HARDWARE MONITORING
            // ==========================================
            SectionHeader(title = "C. HARDWARE MONITORING & CPU TUNING")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Tune, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("CPU Threads & Thermal Throttling", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Active Inference Threads", style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${threadSliderValue.roundToInt()} Threads",
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.secondary
                        )
                    }

                    Slider(
                        value = threadSliderValue,
                        onValueChange = { threadSliderValue = it },
                        onValueChangeFinished = {
                            viewModel.updateThreads(threadSliderValue.roundToInt())
                        },
                        valueRange = 1f..maxCores.toFloat(),
                        steps = (maxCores - 2).coerceAtLeast(0),
                        modifier = Modifier.fillMaxWidth()
                    )

                    // Quick thread presets
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(2, 4, 6, 8).filter { it <= maxCores }.forEach { threads ->
                            FilterChip(
                                selected = uiState.threadCount == threads,
                                onClick = {
                                    threadSliderValue = threads.toFloat()
                                    viewModel.updateThreads(threads)
                                },
                                label = { Text("${threads}T", fontSize = 12.sp) },
                                colors = FilterChipDefaults.filterChipColors()
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    DashboardRow(
                        label = "Thermal Status",
                        value = uiState.thermalStatusName,
                        valueColor = if (uiState.isThermalThrottling) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )

                    if (uiState.isThermalThrottling) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                                .padding(8.dp)
                        ) {
                            Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Device is experiencing thermal throttling. Inference throughput may be reduced.",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    DashboardRow(label = "CPU Core Count", value = "$maxCores Cores Available")
                    DashboardRow(label = "Compute Acceleration", value = "ARM NEON 128-bit Vectorized")
                    DashboardRow(label = "Backend Engine", value = "llama.cpp CPU NEON")
                }
            }

            // ==========================================
            // SECTION D: RUNTIME INFORMATION
            // ==========================================
            SectionHeader(title = "D. RUNTIME INFORMATION")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Speed, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Device & Software Environment", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    DashboardRow(label = "Device", value = "${Build.MANUFACTURER.uppercase()} ${Build.MODEL}")
                    DashboardRow(label = "OS Platform", value = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                    DashboardRow(label = "MobileLLM Core", value = uiState.engineVersion)
                    DashboardRow(label = "llama.cpp Version", value = "b4260 (NEON Optimized)")
                    DashboardRow(label = "Active Model", value = uiState.loadedModelName ?: "None Loaded")
                    DashboardRow(label = "Offline Execution", value = "100% On-Device (Zero Network)", valueColor = MaterialTheme.colorScheme.primary)
                }
            }

            // ==========================================
            // SECTION E: BENCHMARK HISTORY & THREAD COMPARISON
            // ==========================================
            SectionHeader(title = "E. AUTOMATED BENCHMARK & THREAD COMPARISON")

            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text("CPU Thread Benchmark", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                            Text("Evaluates standard prompt across 2, 4, 6, 8 threads", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }

                        Button(
                            onClick = { viewModel.runBenchmark() },
                            enabled = uiState.isModelLoaded && !uiState.isGenerating && !uiState.isBenchmarking,
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                        ) {
                            if (uiState.isBenchmarking) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Running...", fontSize = 12.sp)
                            } else {
                                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Run", fontSize = 12.sp)
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    if (uiState.benchmarkHistory.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 12.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = if (uiState.isModelLoaded)
                                    "Tap 'Run' to benchmark generation speed across CPU thread configurations."
                                else
                                    "Load a model first to run the automated thread benchmark.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            uiState.benchmarkHistory.takeLast(6).reversed().forEach { run ->
                                BenchmarkRunCard(run)
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(20.dp))
        }
    }
}

@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        fontSize = 12.sp,
        fontWeight = FontWeight.ExtraBold,
        letterSpacing = 1.sp,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(top = 8.dp, bottom = 2.dp)
    )
}

@Composable
private fun MetricCard(
    title: String,
    value: String,
    subtitle: String,
    modifier: Modifier = Modifier
) {
    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = modifier
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = title.uppercase(),
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                fontSize = 19.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = subtitle,
                fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun BenchmarkRunCard(run: BenchmarkRun) {
    Card(
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "${run.threads} CPU Threads",
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = "%.1f tok/s".format(run.tokensPerSecond),
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Prefill: %.1f tps | TTFT: %.0f ms".format(run.promptTokensPerSecond, run.timeToFirstTokenMs),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "Peak: %.1f MB".format(run.peakRssMb),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun DashboardRow(
    label: String,
    value: String,
    valueColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}
