package ai.codesec.nexa.ui

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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.ObsidianBackground
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard
import ai.codesec.nexa.ui.theme.ObsidianCardHighlight
import kotlin.math.roundToInt

@Composable
fun SettingsDialog(
    initialParams: SamplingParams,
    initialBackend: BackendType,
    initialThreads: Int,
    initialGpuLayers: Int = -1,
    onSave: (SamplingParams, BackendType, Int, Int) -> Unit,
    onDismiss: () -> Unit
) {
    var temperature by remember { mutableFloatStateOf(initialParams.temperature) }
    var topP by remember { mutableFloatStateOf(initialParams.topP) }
    var repetitionPenalty by remember { mutableFloatStateOf(initialParams.repetitionPenalty) }
    var maxTokens by remember { mutableIntStateOf(initialParams.maxTokens) }
    var selectedBackend by remember { mutableStateOf(initialBackend) }
    var threads by remember { mutableIntStateOf(initialThreads) }
    var gpuLayers by remember { mutableIntStateOf(if (initialGpuLayers >= 0) initialGpuLayers else 28) }

    val maxThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = ObsidianCard,
        shape = RoundedCornerShape(24.dp),
        title = {
            Text(
                "Inference Engine Config",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = Color.White
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    "Compute Acceleration Backend",
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    color = CyberLime
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
                            .clip(RoundedCornerShape(12.dp))
                            .background(if (isCpu) CyberLime else ObsidianBackground)
                            .border(1.dp, if (isCpu) CyberLime else ObsidianBorder, RoundedCornerShape(12.dp))
                            .clickable { selectedBackend = BackendType.CPU_NEON }
                            .padding(vertical = 10.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "CPU (NEON)",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            color = if (isCpu) Color(0xFF0D1110) else Color(0xFF94A39D)
                        )
                    }

                    val isGpu = selectedBackend == BackendType.VULKAN
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(12.dp))
                            .background(if (isGpu) ColdBlue else ObsidianBackground)
                            .border(1.dp, if (isGpu) ColdBlue else ObsidianBorder, RoundedCornerShape(12.dp))
                            .clickable { selectedBackend = BackendType.VULKAN }
                            .padding(vertical = 10.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "GPU (Vulkan)",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            color = if (isGpu) Color(0xFF0D1110) else Color(0xFF94A39D)
                        )
                    }
                }

                if (selectedBackend == BackendType.VULKAN) {
                    Spacer(modifier = Modifier.height(12.dp))
                    SettingSlider(
                        label = "GPU Offload Layers",
                        valueStr = "$gpuLayers / 28",
                        value = gpuLayers.toFloat(),
                        range = 0f..28f,
                        steps = 27,
                        accentColor = ColdBlue,
                        onValueChange = { gpuLayers = it.roundToInt() }
                    )
                }

                Spacer(modifier = Modifier.height(14.dp))
                Divider(color = ObsidianBorder)
                Spacer(modifier = Modifier.height(14.dp))

                SettingSlider(
                    label = "Compute Threads",
                    valueStr = "$threads Threads",
                    value = threads.toFloat(),
                    range = 1f..maxThreads.toFloat(),
                    steps = maxThreads - 1,
                    accentColor = CyberLime,
                    onValueChange = { threads = it.roundToInt() }
                )

                SettingSlider(
                    label = "Temperature",
                    valueStr = "%.2f".format(temperature),
                    value = temperature,
                    range = 0.0f..2.0f,
                    accentColor = CyberLime,
                    onValueChange = { temperature = it }
                )

                SettingSlider(
                    label = "Top-P Sampling",
                    valueStr = "%.2f".format(topP),
                    value = topP,
                    range = 0.05f..1.0f,
                    accentColor = CyberLime,
                    onValueChange = { topP = it }
                )

                SettingSlider(
                    label = "Repetition Penalty",
                    valueStr = "%.2f".format(repetitionPenalty),
                    value = repetitionPenalty,
                    range = 1.0f..2.0f,
                    accentColor = CyberLime,
                    onValueChange = { repetitionPenalty = it }
                )

                SettingSlider(
                    label = "Max Output Tokens",
                    valueStr = "$maxTokens tok",
                    value = maxTokens.toFloat(),
                    range = 64f..2048f,
                    steps = 30,
                    accentColor = CyberLime,
                    onValueChange = { maxTokens = it.roundToInt() }
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    val newParams = initialParams.copy(
                        temperature = temperature,
                        topP = topP,
                        repetitionPenalty = repetitionPenalty,
                        maxTokens = maxTokens
                    )
                    val finalLayers = if (selectedBackend == BackendType.CPU_NEON) 0 else gpuLayers
                    onSave(newParams, selectedBackend, threads, finalLayers)
                    onDismiss()
                },
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = CyberLime,
                    contentColor = Color(0xFF0D1110)
                )
            ) {
                Text("Apply & Save", fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = Color(0xFF94A39D))
            }
        }
    )
}

@Composable
private fun SettingSlider(
    label: String,
    valueStr: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    accentColor: Color,
    onValueChange: (Float) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = label, fontSize = 12.sp, color = Color(0xFF94A39D))
            Text(
                text = valueStr,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                color = accentColor
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            steps = steps,
            colors = SliderDefaults.colors(
                thumbColor = accentColor,
                activeTrackColor = accentColor,
                inactiveTrackColor = ObsidianCardHighlight
            )
        )
    }
}
