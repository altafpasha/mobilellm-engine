package ai.codesec.nexa.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.nexa.ui.theme.BrandCoral
import ai.codesec.nexa.ui.theme.BrandCoralContainer
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.StatsBlue
import ai.codesec.nexa.ui.theme.StudioBorder
import ai.codesec.nexa.ui.theme.StudioCard
import ai.codesec.nexa.ui.theme.StudioCardHighlight
import ai.codesec.nexa.ui.theme.StudioSurface
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InferenceSettingsBottomSheet(
    initialParams: SamplingParams,
    initialBackend: BackendType,
    initialThreads: Int,
    initialGpuLayers: Int,
    onSave: (SamplingParams, BackendType, Int, Int) -> Unit,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var temperature by remember { mutableFloatStateOf(initialParams.temperature) }
    var topP by remember { mutableFloatStateOf(initialParams.topP) }
    var topK by remember { mutableIntStateOf(initialParams.topK) }
    var repetitionPenalty by remember { mutableFloatStateOf(initialParams.repetitionPenalty) }
    var maxTokens by remember { mutableIntStateOf(initialParams.maxTokens) }
    var selectedBackend by remember { mutableStateOf(initialBackend) }
    var threads by remember { mutableIntStateOf(initialThreads) }
    var gpuLayers by remember { mutableIntStateOf(if (initialGpuLayers >= 0) initialGpuLayers else 28) }

    val maxCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = StudioSurface,
        dragHandle = null,
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 18.dp)
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Inference & Runtime Settings",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "Sampling parameters & hardware scheduling",
                        fontSize = 12.sp,
                        color = Color(0xFF9CA3AF)
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(
                        onClick = {
                            temperature = 0.7f
                            topP = 0.9f
                            topK = 40
                            repetitionPenalty = 1.1f
                            maxTokens = 512
                            selectedBackend = BackendType.CPU_NEON
                            threads = 4
                            gpuLayers = 28
                        },
                        modifier = Modifier
                            .size(32.dp)
                            .background(StudioCard, CircleShape)
                            .border(1.dp, StudioBorder, CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Default.RestartAlt,
                            contentDescription = "Reset",
                            tint = Color(0xFF9CA3AF),
                            modifier = Modifier.size(16.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    IconButton(
                        onClick = onDismiss,
                        modifier = Modifier
                            .size(32.dp)
                            .background(StudioCard, CircleShape)
                            .border(1.dp, StudioBorder, CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(18.dp))

            // Section 1: Hardware Compute Engine
            Text(
                text = "COMPUTE ACCELERATION BACKEND",
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                color = BrandCoral
            )
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                val isCpu = selectedBackend == BackendType.CPU_NEON
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(14.dp))
                        .clickable { selectedBackend = BackendType.CPU_NEON }
                        .background(if (isCpu) BrandCoralContainer else StudioCard)
                        .border(1.dp, if (isCpu) BrandCoral else StudioBorder, RoundedCornerShape(14.dp))
                        .padding(12.dp)
                ) {
                    Column {
                        Text(
                            text = "CPU ARM NEON",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            color = if (isCpu) Color.White else Color(0xFF9CA3AF)
                        )
                        Text(
                            text = "Recommended · 117-153 tok/s",
                            fontSize = 10.sp,
                            color = if (isCpu) BrandCoral else Color(0xFF6B7280)
                        )
                    }
                }

                val isGpu = selectedBackend == BackendType.VULKAN
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(14.dp))
                        .clickable { selectedBackend = BackendType.VULKAN }
                        .background(if (isGpu) StudioCardHighlight else StudioCard)
                        .border(1.dp, if (isGpu) Color(0xFFF59E0B) else StudioBorder, RoundedCornerShape(14.dp))
                        .padding(12.dp)
                ) {
                    Column {
                        Text(
                            text = "Vulkan GPU",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            color = if (isGpu) Color.White else Color(0xFF9CA3AF)
                        )
                        Text(
                            text = "Experimental (Adreno)",
                            fontSize = 10.sp,
                            color = if (isGpu) Color(0xFFF59E0B) else Color(0xFF6B7280)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Thread count slider
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "CPU Execution Threads",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White
                )
                Text(
                    text = "${threads}T (of $maxCores cores)",
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                    color = BrandCoral
                )
            }
            Slider(
                value = threads.toFloat(),
                onValueChange = { threads = it.roundToInt() },
                valueRange = 1f..maxCores.toFloat(),
                steps = (maxCores - 2).coerceAtLeast(0),
                colors = SliderDefaults.colors(
                    thumbColor = BrandCoral,
                    activeTrackColor = BrandCoral,
                    inactiveTrackColor = StudioCardHighlight
                )
            )

            if (selectedBackend == BackendType.VULKAN) {
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "GPU Offload Layers",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                    Text(
                        text = "$gpuLayers / 28 Layers",
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFFF59E0B)
                    )
                }
                Slider(
                    value = gpuLayers.toFloat(),
                    onValueChange = { gpuLayers = it.roundToInt() },
                    valueRange = 1f..28f,
                    steps = 26,
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFFF59E0B),
                        activeTrackColor = Color(0xFFF59E0B),
                        inactiveTrackColor = StudioCardHighlight
                    )
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Section 2: Generation Sampling Settings
            Text(
                text = "TOKEN GENERATION SAMPLING",
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                color = BrandCoral
            )
            Spacer(modifier = Modifier.height(8.dp))

            // Temperature
            SettingSliderRow(
                label = "Temperature",
                valueText = "%.2f".format(temperature),
                value = temperature,
                onValueChange = { temperature = it },
                range = 0.0f..1.5f
            )

            // Top P
            SettingSliderRow(
                label = "Top-P (Nucleus)",
                valueText = "%.2f".format(topP),
                value = topP,
                onValueChange = { topP = it },
                range = 0.1f..1.0f
            )

            // Top K
            SettingSliderRow(
                label = "Top-K Filtering",
                valueText = "$topK",
                value = topK.toFloat(),
                onValueChange = { topK = it.roundToInt() },
                range = 1f..100f
            )

            // Repetition Penalty
            SettingSliderRow(
                label = "Repetition Penalty",
                valueText = "%.2f".format(repetitionPenalty),
                value = repetitionPenalty,
                onValueChange = { repetitionPenalty = it },
                range = 1.0f..1.5f
            )

            // Max Tokens
            SettingSliderRow(
                label = "Max Generation Tokens",
                valueText = "$maxTokens tokens",
                value = maxTokens.toFloat(),
                onValueChange = { maxTokens = it.roundToInt() },
                range = 64f..2048f
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Save button
            Button(
                onClick = {
                    val finalParams = initialParams.copy(
                        temperature = temperature,
                        topP = topP,
                        topK = topK,
                        repetitionPenalty = repetitionPenalty,
                        maxTokens = maxTokens
                    )
                    val finalLayers = if (selectedBackend == BackendType.CPU_NEON) 0 else gpuLayers
                    onSave(finalParams, selectedBackend, threads, finalLayers)
                    onDismiss()
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = ButtonDefaults.buttonColors(containerColor = BrandCoral),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text(
                    text = "Apply Settings",
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    color = Color.White
                )
            }

            Spacer(modifier = Modifier.height(10.dp))
        }
    }
}

@Composable
private fun SettingSliderRow(
    label: String,
    valueText: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    range: ClosedFloatingPointRange<Float>
) {
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                fontSize = 12.sp,
                color = Color.White
            )
            Text(
                text = valueText,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
                color = Color(0xFF9CA3AF)
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            colors = SliderDefaults.colors(
                thumbColor = BrandCoral,
                activeTrackColor = BrandCoral,
                inactiveTrackColor = StudioCardHighlight
            )
        )
    }
}
