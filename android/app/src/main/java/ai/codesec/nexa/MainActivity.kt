package ai.codesec.nexa

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.codesec.nexa.ui.ChatScreen
import ai.codesec.nexa.ui.ModelManagerScreen
import ai.codesec.nexa.ui.PerformanceDashboardScreen
import ai.codesec.nexa.ui.theme.CyberLime
import ai.codesec.nexa.ui.theme.NexaTheme
import ai.codesec.nexa.ui.theme.ObsidianBackground
import ai.codesec.nexa.ui.theme.ObsidianBorder
import ai.codesec.nexa.ui.theme.ObsidianCard
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.NexaScreen

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            NexaTheme {
                val uiState by viewModel.uiState.collectAsState()

                BackHandler(enabled = uiState.currentScreen != NexaScreen.CHAT) {
                    viewModel.setScreen(NexaScreen.CHAT)
                }

                Scaffold(
                    containerColor = ObsidianBackground,
                    bottomBar = {
                        if (uiState.currentScreen != NexaScreen.CHAT) {
                            ModelLabBottomNav(
                                currentScreen = uiState.currentScreen,
                                onSelectScreen = { viewModel.setScreen(it) }
                            )
                        }
                    }
                ) { innerPadding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(bottom = if (uiState.currentScreen != NexaScreen.CHAT) innerPadding.calculateBottomPadding() else 0.dp)
                    ) {
                        AnimatedContent(
                            targetState = uiState.currentScreen,
                            transitionSpec = { fadeIn() togetherWith fadeOut() },
                            label = "ScreenTransition"
                        ) { screen ->
                            when (screen) {
                                NexaScreen.CHAT -> ChatScreen(viewModel = viewModel)
                                NexaScreen.MODELS -> ModelManagerScreen(
                                    viewModel = viewModel,
                                    onBack = { viewModel.setScreen(NexaScreen.CHAT) }
                                )
                                NexaScreen.DASHBOARD -> PerformanceDashboardScreen(
                                    viewModel = viewModel,
                                    onBack = { viewModel.setScreen(NexaScreen.CHAT) }
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ModelLabBottomNav(
    currentScreen: NexaScreen,
    onSelectScreen: (NexaScreen) -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding(),
        color = ObsidianBackground,
        tonalElevation = 8.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 6.dp)
                .clip(RoundedCornerShape(24.dp))
                .background(ObsidianCard)
                .border(1.dp, ObsidianBorder, RoundedCornerShape(24.dp))
                .padding(horizontal = 12.dp, vertical = 4.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround,
                verticalAlignment = Alignment.CenterVertically
            ) {
                NavItem(
                    label = "Lab",
                    icon = Icons.Default.Bolt,
                    selected = currentScreen == NexaScreen.CHAT,
                    onClick = { onSelectScreen(NexaScreen.CHAT) }
                )
                NavItem(
                    label = "Models",
                    icon = Icons.Default.Folder,
                    selected = currentScreen == NexaScreen.MODELS,
                    onClick = { onSelectScreen(NexaScreen.MODELS) }
                )
                NavItem(
                    label = "Benchmarks",
                    icon = Icons.Default.Speed,
                    selected = currentScreen == NexaScreen.DASHBOARD,
                    onClick = { onSelectScreen(NexaScreen.DASHBOARD) }
                )
            }
        }
    }
}

@Composable
private fun NavItem(
    label: String,
    icon: ImageVector,
    selected: Boolean,
    onClick: () -> Unit
) {
    val activeColor = CyberLime
    val inactiveColor = Color(0xFF869690)

    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .background(if (selected) Color(0xFF1E281F) else Color.Transparent)
            .padding(horizontal = 16.dp, vertical = 6.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = if (selected) activeColor else inactiveColor,
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = label,
                fontSize = 12.sp,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                color = if (selected) activeColor else inactiveColor
            )
        }
    }
}
