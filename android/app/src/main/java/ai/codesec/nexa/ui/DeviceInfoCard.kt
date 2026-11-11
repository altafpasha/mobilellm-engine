package ai.codesec.nexa.ui

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun DeviceInfoDialog(
    engineVersion: String,
    hardwareInfo: String,
    loadedModelName: String?,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    val memInfo = ActivityManager.MemoryInfo().also { actManager?.getMemoryInfo(it) }

    val totalRamMb = (memInfo.totalMem) / (1024 * 1024)
    val availRamMb = (memInfo.availMem) / (1024 * 1024)
    val lowMem = memInfo.lowMemory

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("Engine & Hardware Diagnostics", style = MaterialTheme.typography.titleLarge)
        },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                InfoRow("MobileLLM Core", engineVersion)
                InfoRow("Offline Status", "100% On-Device (No Network)")
                InfoRow("Active Model", loadedModelName ?: "None Loaded")

                Spacer(modifier = Modifier.height(8.dp))
                Divider()
                Spacer(modifier = Modifier.height(8.dp))

                Text("System Memory", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                InfoRow("Available RAM", "$availRamMb MB / $totalRamMb MB")
                InfoRow("Low Memory State", if (lowMem) "CRITICAL" else "NORMAL")

                Spacer(modifier = Modifier.height(8.dp))
                Divider()
                Spacer(modifier = Modifier.height(8.dp))

                Text("Hardware Features", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleSmall)
                Text(
                    text = hardwareInfo,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(text = value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
    }
}

