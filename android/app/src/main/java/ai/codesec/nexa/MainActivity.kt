package ai.codesec.nexa

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import ai.codesec.nexa.ui.ChatScreen
import ai.codesec.nexa.ui.ModelManagerScreen
import ai.codesec.nexa.ui.PerformanceDashboardScreen
import ai.codesec.nexa.ui.theme.NexaTheme
import ai.codesec.nexa.viewmodel.ChatViewModel
import ai.codesec.nexa.viewmodel.NexaScreen

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            NexaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val uiState by viewModel.uiState.collectAsState()

                    BackHandler(enabled = uiState.currentScreen != NexaScreen.CHAT) {
                        viewModel.setScreen(NexaScreen.CHAT)
                    }

                    when (uiState.currentScreen) {
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
