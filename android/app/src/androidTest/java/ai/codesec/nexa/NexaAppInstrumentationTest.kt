package ai.codesec.nexa

import android.content.pm.PackageManager
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.codesec.mobilellm.MobileLLMClient
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.mobilellm.models.SessionConfig
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class NexaAppInstrumentationTest {

    private val TAG = "NexaInstrumentationTest"

    @Test
    fun verifyNoInternetPermissionGranted() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val packageInfo = appContext.packageManager.getPackageInfo(
            appContext.packageName,
            PackageManager.GET_PERMISSIONS
        )

        val requestedPermissions = packageInfo.requestedPermissions ?: emptyArray()
        assertFalse(
            "Security violation: NEXA must NEVER request android.permission.INTERNET",
            requestedPermissions.contains("android.permission.INTERNET")
        )
    }

    @Test
    fun useAppContext() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("ai.codesec.nexa.debug", appContext.packageName)
    }

    @Test
    fun testNativeInferenceWithDeviceModel() = runBlocking {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val client = MobileLLMClient.getInstance(appContext)
        client.initialize()

        val version = client.getVersion()
        val hw = client.detectHardware()
        Log.i(TAG, "Native Engine Version: $version")
        Log.i(TAG, "Hardware Detection: $hw")
        assertTrue(version.isNotBlank())

        val modelFile = File(appContext.filesDir, "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
        assertTrue("Model file must exist at ${modelFile.absolutePath}", modelFile.exists())
        Log.i(TAG, "Found model file: ${modelFile.name} (${modelFile.length() / (1024 * 1024)} MB)")

        val modelConfig = ModelConfig(
            modelPath = modelFile.absolutePath,
            preferredBackend = BackendType.AUTO,
            contextLength = 1024,
            nThreads = 4
        )
        val sessionConfig = SessionConfig(
            maxContextTokens = 1024,
            enablePrefixCaching = true
        )

        val session = client.loadModel(modelConfig, sessionConfig)
        assertNotNull("Session must be successfully created", session)
        Log.i(TAG, "Model successfully loaded into native memory!")

        val prompt = "Tell me a fun fact about mobile computers."
        val params = SamplingParams(
            temperature = 0.7f,
            maxTokens = 64
        )

        var reportedTtftMs = 0.0
        var reportedTps = 0.0
        var reportedRssMb = 0.0

        val chunks = session.generate(prompt, params) { metrics ->
            reportedTtftMs = metrics.timeToFirstTokenMs
            reportedTps = metrics.tokensPerSecond
            reportedRssMb = metrics.peakRssMb
            Log.i(TAG, "=== INFERENCE TELEMETRY ===")
            Log.i(TAG, "TTFT: ${metrics.timeToFirstTokenMs} ms")
            Log.i(TAG, "Tokens/sec: ${metrics.tokensPerSecond} tps")
            Log.i(TAG, "Prompt tokens: ${metrics.promptTokens}, Completion tokens: ${metrics.completionTokens}")
            Log.i(TAG, "Peak RSS: ${metrics.peakRssMb} MB")
        }.toList()

        val generatedText = chunks.joinToString("") { it.text }
        Log.i(TAG, "=== GENERATED TEXT ===")
        Log.i(TAG, generatedText)

        assertTrue("Generated response must contain tokens", chunks.isNotEmpty())
        assertTrue("Generated text must not be empty", generatedText.isNotBlank())
        session.close()
    }
}

