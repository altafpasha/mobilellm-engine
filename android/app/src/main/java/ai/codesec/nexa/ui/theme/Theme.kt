package ai.codesec.nexa.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// Obsidian Cyber-Lime Studio Palette
val ObsidianBackground = Color(0xFF0D1110)
val ObsidianSurface = Color(0xFF131816)
val ObsidianCard = Color(0xFF1A211E)
val ObsidianCardHighlight = Color(0xFF222C28)
val ObsidianBorder = Color(0xFF28342F)

val CyberLime = Color(0xFFA3E635)
val CyberLimeDark = Color(0xFF65A30D)
val CyberLimeGlow = Color(0x33A3E635)
val CyberLimeContainer = Color(0xFF1E281F)

val ColdBlue = Color(0xFF38BDF8)
val ColdBlueContainer = Color(0xFF132832)
val MintSuccess = Color(0xFF4ADE80)
val CrimsonError = Color(0xFFF87171)

private val DarkColorScheme = darkColorScheme(
    primary = CyberLime,
    onPrimary = Color(0xFF0D1110),
    primaryContainer = CyberLimeContainer,
    onPrimaryContainer = CyberLime,
    secondary = ColdBlue,
    onSecondary = Color(0xFF0D1110),
    secondaryContainer = ColdBlueContainer,
    onSecondaryContainer = ColdBlue,
    tertiary = MintSuccess,
    background = ObsidianBackground,
    onBackground = Color(0xFFF1F5F3),
    surface = ObsidianSurface,
    onSurface = Color(0xFFF1F5F3),
    surfaceVariant = ObsidianCard,
    onSurfaceVariant = Color(0xFF94A39D),
    outline = ObsidianBorder,
    outlineVariant = Color(0xFF1E2623),
    error = CrimsonError
)

// Default to dark theme for developer inference studio aesthetic
private val LightColorScheme = DarkColorScheme

@Composable
fun NexaTheme(
    darkTheme: Boolean = true, // Force dark-first studio theme
    content: @Composable () -> Unit
) {
    val colorScheme = DarkColorScheme
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            window.navigationBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
            WindowCompat.getInsetsController(window, view).isAppearanceLightNavigationBars = false
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
