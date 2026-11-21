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
val ObsidianBackground = Color(0xFF0E0F10)
val ObsidianSurface = Color(0xFF16181A)
val ObsidianCard = Color(0xFF1E2023)
val ObsidianCardHighlight = Color(0xFF26292D)
val ObsidianBorder = Color(0xFF2C3035)

// Modern AI Chat Studio Palette (matching reference screenshot)
val BrandCoral = Color(0xFFE05345)
val BrandCoralDark = Color(0xFFB83A2E)
val BrandCoralGlow = Color(0x33E05345)
val BrandCoralContainer = Color(0xFF2B1816)

val StudioBackground = Color(0xFF0E0F10)
val StudioSurface = Color(0xFF16181A)
val StudioCard = Color(0xFF1E2023)
val StudioCardHighlight = Color(0xFF26292D)
val StudioBorder = Color(0xFF2C3035)

val StatsBlue = Color(0xFF0284C7)
val StatsBlueGlow = Color(0x330284C7)
val StatsBlueContainer = Color(0xFF0E2536)

val CyberLime = Color(0xFFA3E635)
val CyberLimeDark = Color(0xFF65A30D)
val CyberLimeGlow = Color(0x33A3E635)
val CyberLimeContainer = Color(0xFF1E281F)

val ColdBlue = Color(0xFF38BDF8)
val ColdBlueContainer = Color(0xFF132832)
val MintSuccess = Color(0xFF4ADE80)
val CrimsonError = Color(0xFFF87171)

private val DarkColorScheme = darkColorScheme(
    primary = BrandCoral,
    onPrimary = Color.White,
    primaryContainer = BrandCoralContainer,
    onPrimaryContainer = BrandCoral,
    secondary = StatsBlue,
    onSecondary = Color.White,
    secondaryContainer = StatsBlueContainer,
    onSecondaryContainer = StatsBlue,
    tertiary = MintSuccess,
    background = StudioBackground,
    onBackground = Color(0xFFF3F4F6),
    surface = StudioSurface,
    onSurface = Color(0xFFF3F4F6),
    surfaceVariant = StudioCard,
    onSurfaceVariant = Color(0xFF9CA3AF),
    outline = StudioBorder,
    outlineVariant = Color(0xFF222528),
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
