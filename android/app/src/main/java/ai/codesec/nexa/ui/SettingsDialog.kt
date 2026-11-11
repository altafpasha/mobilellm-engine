package ai.codesec.nexa.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.SamplingParams
import kotlin.math.roundToInt

@Composable
fun SettingsDialog(
    initialParams: SamplingParams,
    initialBackend: BackendType,
    initialThreads: Int,
    onSave: (SamplingParams, BackendType, Int) -> Unit,
    onDismiss: () -> Unit
) {
    var temperature by remember { mutableFloatStateOf(initialParams.temperature) }
    var topP by remember { mutableFloatStateOf(initialParams.topP) }
    var repetitionPenalty by remember { mutableFloatStateOf(initialParams.repetitionPenalty) }
    var maxTokens by remember { mutableIntStateOf(initialParams.maxTokens) }
    var selectedBackend by remember { mutableStateOf(initialBackend) }
    var threads by remember { mutableIntStateOf(initialThreads) }

    val maxThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("Inference Settings", style = MaterialTheme.typography.titleLarge)
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                Text("Compute Backend", style = MaterialTheme.typography.titleSmall)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = selectedBackend == BackendType.CPU_NEON,
                        onClick = { selectedBackend = BackendType.CPU_NEON },
                        label = { Text("CPU (ARM NEON)") }
                    )
                    FilterChip(
                        selected = selectedBackend == BackendType.VULKAN,
                        onClick = { selectedBackend = BackendType.VULKAN },
                        label = { Text("GPU (Vulkan)") }
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))
                Divider()
                Spacer(modifier = Modifier.height(12.dp))

                SettingSlider(
                    label = "Threads: $threads",
                    value = threads.toFloat(),
                    range = 1f..maxThreads.toFloat(),
                    steps = maxThreads - 1,
                    onValueChange = { threads = it.roundToInt() }
                )

                SettingSlider(
                    label = "Temperature: ${"%.2f".format(temperature)}",
                    value = temperature,
                    range = 0.0f..2.0f,
                    onValueChange = { temperature = it }
                )

                SettingSlider(
                    label = "Top-P: ${"%.2f".format(topP)}",
                    value = topP,
                    range = 0.05f..1.0f,
                    onValueChange = { topP = it }
                )

                SettingSlider(
                    label = "Repetition Penalty: ${"%.2f".format(repetitionPenalty)}",
                    value = repetitionPenalty,
                    range = 1.0f..2.0f,
                    onValueChange = { repetitionPenalty = it }
                )

                SettingSlider(
                    label = "Max Output Tokens: $maxTokens",
                    value = maxTokens.toFloat(),
                    range = 64f..2048f,
                    steps = 30,
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
                    onSave(newParams, selectedBackend, threads)
                    onDismiss()
                }
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            OutlinedButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
private fun SettingSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    onValueChange: (Float) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Text(text = label, style = MaterialTheme.typography.bodyMedium)
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            steps = steps
        )
    }
}

