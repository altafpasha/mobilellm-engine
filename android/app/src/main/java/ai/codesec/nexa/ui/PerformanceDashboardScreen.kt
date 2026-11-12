package ai.codesec.nexa.ui

import android.os.Build
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
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
                title = { Text("Performance Dashboard", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refreshSystemStats() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh Metrics")
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
            // Inference Speed Hero Card
            Card(
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f)
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "TOKEN GENERATION THROUGHPUT",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = if (metrics != null && metrics.tokensPerSecond > 0)
                            "%.1f tok/s".format(metrics.tokensPerSecond)
                        else
                            "N/A",
                        fontSize = 38.sp,
                        fontWeight = FontWeight.ExtraBold,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (uiState.isGenerating) "Evaluating & Streaming Tokens..." else "Measured during last generation run",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                    )
                }
            }

            // Latency & Token Metrics Grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "TTFT",
                    value = if (metrics != null && metrics.timeToFirstTokenMs > 0) "%.0f ms".format(metrics.timeToFirstTokenMs) else "N/A",
                    subtitle = "Time to first token",
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Total Time",
                    value = if (metrics != null && metrics.totalDurationMs > 0) "%.2f s".format(metrics.totalDurationMs / 1000.0) else "N/A",
                    subtitle = "Complete inference",
                    modifier = Modifier.weight(1f)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MetricCard(
                    title = "Prompt Tokens",
                    value = if (metrics != null) "${metrics.promptTokens}" else "N/A",
                    subtitle = "Evaluated context",
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Output Tokens",
                    value = if (metrics != null) "${metrics.completionTokens}" else "N/A",
                    subtitle = "Generated completion",
                    modifier = Modifier.weight(1f)
                )
            }

            // Memory Utilization Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Memory, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Memory & Virtual Address Space", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    val usedRamPercent = if (uiState.systemTotalRamMb > 0) {
                        ((uiState.systemTotalRamMb - uiState.systemAvailRamMb).toFloat() / uiState.systemTotalRamMb)
                    } else 0f

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("System RAM", style = MaterialTheme.typography.bodyMedium)
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
                        modifier = Modifier.fillMaxWidth().height(6.dp),
                        color = if (usedRamPercent > 0.85f) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    DashboardRow(
                        label = "Native RSS (Peak)",
                        value = if (metrics != null && metrics.peakRssMb > 0) "%.1f MB".format(metrics.peakRssMb) else "N/A"
                    )
                    DashboardRow(
                        label = "JVM Heap Allocated",
                        value = "%.1f MB".format(uiState.jvmHeapMb)
                    )
                    DashboardRow(
                        label = "OS Low Memory State",
                        value = if (uiState.isLowMemory) "CRITICAL" else "NORMAL",
                        valueColor = if (uiState.isLowMemory) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )
                }
            }

            // Dynamic Thread & Hardware Tuning Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Tune, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Hardware Thread Allocation", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("CPU Inference Threads", style = MaterialTheme.typography.bodyMedium)
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

                    Text(
                        text = "Device has $maxCores physical/logical cores. Setting threads higher than CPU efficiency cores may cause throttling.",
                        fontSize = 11.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Hardware & Diagnostics Card
            Card(
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Speed, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Device & Runtime Information", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    DashboardRow(label = "Device", value = "${Build.MANUFACTURER.uppercase()} ${Build.MODEL}")
                    DashboardRow(label = "OS Version", value = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                    DashboardRow(label = "MobileLLM Core", value = uiState.engineVersion)
                    DashboardRow(label = "Compute SIMD", value = "ARM NEON 128-bit Vectorized")
                    DashboardRow(label = "Active Backend", value = "LlamaCpp_CPU_NEON")
                    DashboardRow(label = "Offline Security", value = "100% On-Device (Zero Network)", valueColor = MaterialTheme.colorScheme.primary)
                    DashboardRow(label = "Active Model", value = uiState.loadedModelName ?: "None Loaded")
                }
            }
        }
    }
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
                fontSize = 20.sp,
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
