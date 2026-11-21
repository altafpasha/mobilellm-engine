package ai.codesec.nexa.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.nexa.ui.theme.BrandCoral
import ai.codesec.nexa.ui.theme.ColdBlue
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.MintSuccess
import ai.codesec.nexa.ui.theme.StatsBlue
import ai.codesec.nexa.ui.theme.StudioBorder
import ai.codesec.nexa.ui.theme.StudioCard
import ai.codesec.nexa.ui.theme.StudioCardHighlight
import ai.codesec.nexa.viewmodel.ChatUiState

/**
 * Modern Live Inference Statistics Dashboard directly above chat message input.
 * Inspired by the on-device AI Chat developer playground reference design.
 */
@Composable
fun InferenceStatsDashboard(
    uiState: ChatUiState,
    visible: Boolean,
    onOpenInfo: () -> Unit,
    modifier: Modifier = Modifier
) {
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn() + expandVertically(),
        exit = fadeOut() + shrinkVertically(),
        modifier = modifier
    ) {
        val metrics = uiState.lastMetrics
        val tps = metrics?.tokensPerSecond ?: 0.0
        val ttft = metrics?.timeToFirstTokenMs ?: 0.0
        val genTokens = metrics?.completionTokens ?: 0
        val promptTokens = metrics?.promptTokens ?: 0

        val speedText = if (tps > 0.0) "%.1f tok/s".format(tps) else "-- tok/s"
        val initText = if (ttft > 0.0) "Init: %.1fs".format(ttft / 1000.0) else "Init: --"

        val tokensText = if (genTokens > 0) "$genTokens gen" else "-- gen"
        val ramText = if (uiState.nativeRssMb > 0) "%.0f MB".format(uiState.nativeRssMb) else "-- MB"

        val peakRss = if (uiState.nativePeakRssMb > 0) uiState.nativePeakRssMb else (metrics?.peakRssMb ?: 0.0)
        val peakSubtitle = if (peakRss > 0) "Peak: %.0f MB".format(peakRss) else null

        val contextCap = if (uiState.contextCapacity > 0) uiState.contextCapacity else 4000
        val contextUsed = uiState.usedContextTokens
        val contextRatio = (contextUsed.toFloat() / contextCap.toFloat()).coerceIn(0f, 1f)

        Card(
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.cardColors(containerColor = StudioCard),
            border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 4.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp)
            ) {
                // Top row: Speed, Tokens, RAM + Info button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.Top,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    // Metric 1: Speed
                    Column(modifier = Modifier.weight(1.1f)) {
                        Text(
                            text = "Speed",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF9CA3AF)
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = speedText,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            color = Color.White
                        )
                        Text(
                            text = initText,
                            fontSize = 10.sp,
                            color = Color(0xFF6B7280)
                        )
                    }

                    // Metric 2: Tokens
                    Column(modifier = Modifier.weight(1.1f)) {
                        Text(
                            text = "Tokens",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF9CA3AF)
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = tokensText,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            color = Color.White
                        )
                        if (promptTokens > 0) {
                            Text(
                                text = "$promptTokens prompt",
                                fontSize = 10.sp,
                                color = Color(0xFF6B7280)
                            )
                        }
                    }

                    // Metric 3: RAM
                    Column(modifier = Modifier.weight(1.1f)) {
                        Text(
                            text = "RAM",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = Color(0xFF9CA3AF)
                        )
                        Spacer(modifier = Modifier.height(2.dp))
                        Text(
                            text = ramText,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            color = Color.White
                        )
                        if (peakSubtitle != null) {
                            Text(
                                text = peakSubtitle,
                                fontSize = 10.sp,
                                color = Color(0xFF6B7280)
                            )
                        }
                    }

                    // Info button
                    Box(
                        modifier = Modifier
                            .size(24.dp)
                            .clip(CircleShape)
                            .clickable(onClick = onOpenInfo),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Outlined.Info,
                            contentDescription = "Inference Info",
                            tint = Color(0xFF9CA3AF),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Bottom row: Context progress bar with 75% compression tick
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Context",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color(0xFF9CA3AF)
                    )
                    Text(
                        text = if (contextUsed > 0) "$contextUsed / $contextCap" else "-- / $contextCap",
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFF9CA3AF)
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Custom context bar with 75% tick marker
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(StudioCardHighlight)
                ) {
                    // Filled portion
                    val fillColor = when {
                        contextRatio >= 0.85f -> BrandCoral
                        contextRatio >= 0.70f -> Color(0xFFF59E0B)
                        else -> StatsBlue
                    }
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(contextRatio.coerceIn(0f, 1f))
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(fillColor)
                    )

                    // 75% compression tick marker
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.75f)
                            .height(6.dp),
                        contentAlignment = Alignment.CenterEnd
                    ) {
                        Box(
                            modifier = Modifier
                                .width(2.dp)
                                .height(6.dp)
                                .background(Color(0xFF6B7280))
                        )
                    }
                }

                Spacer(modifier = Modifier.height(2.dp))

                // Subtitle under context bar
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    val backendDesc = when (uiState.preferredBackend) {
                        BackendType.VULKAN -> "Backend: Qualcomm Adreno Vulkan (Exp)"
                        BackendType.OPENCL -> "Backend: Qualcomm Adreno OpenCL 3.0"
                        else -> "Backend: ARM64 NEON Vector Engine"
                    }
                    val backendColor = when (uiState.preferredBackend) {
                        BackendType.VULKAN -> Color(0xFFF59E0B)
                        BackendType.OPENCL -> CyberLime
                        else -> Color(0xFF6B7280)
                    }
                    Text(
                        text = backendDesc,
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace,
                        color = backendColor
                    )
                    Text(
                        text = "75% compress",
                        fontSize = 9.sp,
                        color = Color(0xFF6B7280)
                    )
                }
            }
        }
    }
}
