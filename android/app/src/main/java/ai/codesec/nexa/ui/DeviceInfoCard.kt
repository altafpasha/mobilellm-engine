package ai.codesec.nexa.ui

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CrimsonError
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard

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
        containerColor = ObsidianCard,
        shape = RoundedCornerShape(24.dp),
        title = {
            Text(
                "Hardware & Engine Diagnostics",
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
                color = Color.White
            )
        },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                InfoRow("MobileLLM Core", engineVersion, CyberLime)
                InfoRow("Security", "100% On-Device / Zero Network", MintSuccess)
                InfoRow("Active Model", loadedModelName ?: "None Loaded", ColdBlue)

                Spacer(modifier = Modifier.height(10.dp))
                Divider(color = ObsidianBorder)
                Spacer(modifier = Modifier.height(10.dp))

                Text(
                    "System Physical Memory",
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    color = CyberLime
                )
                Spacer(modifier = Modifier.height(4.dp))
                InfoRow("Available RAM", "$availRamMb MB / $totalRamMb MB")
                InfoRow(
                    "Low Memory State",
                    if (lowMem) "CRITICAL" else "NORMAL",
                    if (lowMem) CrimsonError else MintSuccess
                )

                Spacer(modifier = Modifier.height(10.dp))
                Divider(color = ObsidianBorder)
                Spacer(modifier = Modifier.height(10.dp))

                Text(
                    "Hardware Features & Instruction Extensions",
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    color = CyberLime
                )
                Text(
                    text = hardwareInfo,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    color = Color(0xFF94A39D),
                    modifier = Modifier.padding(top = 6.dp)
                )
            }
        },
        confirmButton = {
            Button(
                onClick = onDismiss,
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = CyberLime,
                    contentColor = Color(0xFF0D1110)
                )
            ) {
                Text("Dismiss", fontWeight = FontWeight.Bold)
            }
        }
    )
}

@Composable
private fun InfoRow(label: String, value: String, valueColor: Color = Color(0xFFF1F5F3)) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, fontSize = 12.sp, color = Color(0xFF94A39D))
        Text(
            text = value,
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}
