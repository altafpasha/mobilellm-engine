package ai.codesec.nexa

import android.app.ActivityManager
import android.content.Context
import android.os.PowerManager
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
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.roundToInt
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class VulkanBenchmarkInstrumentationTest {

    companion object {
        private const val TAG = "VulkanBenchmark"
        private const val EXPECTED_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
        private const val PROMPT = "<|im_start|>user\nExplain quantum computing in simple terms<|im_end|>\n<|im_start|>assistant\n"
        private const val MAX_OUTPUT_TOKENS = 64
        private const val CONTEXT_LENGTH = 2048
        private const val DEFAULT_THREADS = 4
        private const val NUM_REPETITIONS = 3
    }

    private fun getModelFile(context: Context): File {
        val extFiles = context.getExternalFilesDir(null)
        val extModel = File(extFiles, "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if (extModel.exists() && extModel.length() > 500_000_000L) {
            return extModel
        }
        val altModel = File("/sdcard/Android/data/ai.codesec.nexa/files/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if (altModel.exists() && altModel.length() > 500_000_000L) {
            return altModel
        }
        val appModel = File(File(context.filesDir, "models"), "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if (appModel.exists() && appModel.length() > 500_000_000L) {
            return appModel
        }
        val downloadModel = File("/sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if (downloadModel.exists() && downloadModel.length() > 500_000_000L) {
            return downloadModel
        }
        throw IllegalStateException("Model file qwen2.5-1.5b-instruct-q4_k_m.gguf not found!")
    }

    private fun readProcessMemoryMb(): Pair<Double, Double> {
        var vmRssMb = 0.0
        var vmHwmMb = 0.0
        try {
            val statusFile = File("/proc/self/status")
            if (statusFile.exists()) {
                statusFile.forEachLine { line ->
                    if (line.startsWith("VmRSS:")) {
                        val parts = line.split("\\s+".toRegex())
                        if (parts.size >= 2) vmRssMb = (parts[1].toDoubleOrNull() ?: 0.0) / 1024.0
                    } else if (line.startsWith("VmHWM:")) {
                        val parts = line.split("\\s+".toRegex())
                        if (parts.size >= 2) vmHwmMb = (parts[1].toDoubleOrNull() ?: 0.0) / 1024.0
                    }
                }
            }
        } catch (_: Exception) {}
        return Pair(vmRssMb, vmHwmMb)
    }

    private fun getAvailableRamMb(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return 0L
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        return memInfo.availMem / (1024 * 1024)
    }

    private fun getThermalStatus(context: Context): Int {
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return -1
        return pm.currentThermalStatus
    }

    private fun computeStats(values: List<Double>): JSONObject {
        val obj = JSONObject()
        if (values.isEmpty()) return obj
        val sorted = values.sorted()
        val n = values.size
        val sum = values.sum()
        val mean = sum / n
        val min = sorted.first()
        val max = sorted.last()

        val variance = if (n > 1) {
            values.map { (it - mean) * (it - mean) }.sum() / (n - 1)
        } else 0.0
        val stddev = sqrt(variance)

        val median = if (n % 2 == 0) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }

        val p90Index = ((n - 1) * 0.90).roundToInt().coerceIn(0, n - 1)
        val p90 = sorted[p90Index]

        obj.put("mean", (mean * 100).roundToInt() / 100.0)
        obj.put("median", (median * 100).roundToInt() / 100.0)
        obj.put("p90", (p90 * 100).roundToInt() / 100.0)
        obj.put("min", (min * 100).roundToInt() / 100.0)
        obj.put("max", (max * 100).roundToInt() / 100.0)
        obj.put("stddev", (stddev * 100).roundToInt() / 100.0)
        return obj
    }

    @Test
    fun testVerifyModelSha256AndMetadata() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            Log.i(TAG, "Testing model at: ${modelFile.absolutePath} (size: ${modelFile.length()} bytes)")
            val meta = client.inspectModel(modelFile, computeHash = true)
            Log.i(TAG, "Model metadata: $meta")
            assertEquals("qwen2", meta.architecture)
            assertEquals("Q4_K_M", meta.quantization)
            assertEquals(EXPECTED_SHA256, meta.sha256)
            assertTrue(meta.isValid)
        }
    }

    @Test
    fun testCpuExecutionVerification() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val cfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = DEFAULT_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val sessCfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            Log.i(TAG, "Loading model with CPU_NEON backend (nGpuLayers=0)...")
            val session = client.loadModel(cfg, sessCfg)
            assertNotNull("CPU session must not be null", session)

            val params = SamplingParams(temperature = 0.0f, maxTokens = 16)
            var reportedTps = 0.0
            var promptToks = 0
            var compToks = 0

            val chunks = session.generate(PROMPT, params) { m ->
                reportedTps = m.tokensPerSecond
                promptToks = m.promptTokens
                compToks = m.completionTokens
                Log.i(TAG, "CPU verification metrics: TTFT=${m.timeToFirstTokenMs}ms, TPS=${m.tokensPerSecond}, PeakRSS=${m.peakRssMb}MB")
            }.toList()

            val generated = chunks.joinToString("") { it.text }
            Log.i(TAG, "CPU generated: $generated")
            assertTrue("Must generate tokens on CPU", chunks.isNotEmpty())
            assertTrue("Generated text must not be empty", generated.isNotBlank())
            assertEquals(16, compToks)
            session.close()
        }
    }

    @Test
    fun testVulkanExecutionVerification() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val sessCfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)
            val params = SamplingParams(temperature = 0.0f, maxTokens = 32)

            val layerCounts = listOf(0, 14, 28)
            val testPrompts = listOf(
                "Math" to "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
                "Quantum" to "<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n"
            )

            for (layers in layerCounts) {
                val backend = if (layers == 0) BackendType.CPU_NEON else BackendType.VULKAN
                Log.i(TAG, "\n========================================================")
                Log.i(TAG, "TESTING LAYER OFF-LOAD: $layers layers (Backend: $backend)")
                Log.i(TAG, "========================================================")

                val cfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = backend,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = DEFAULT_THREADS,
                    useMmap = false,
                    nGpuLayers = layers
                )

                val session = client.loadModel(cfg, sessCfg)
                assertNotNull("Session must load for layers=$layers", session)

                for ((label, p) in testPrompts) {
                    val chunks = session.generate(p, params).toList()
                    val genText = chunks.joinToString("") { it.text }
                    val tokenList = chunks.joinToString(", ") { "${it.tokenId}:'${it.text.replace("\n", "\\n")}'" }
                    Log.i(TAG, "[$label | L=$layers] Text: $genText")
                    Log.i(TAG, "[$label | L=$layers] Tokens: [$tokenList]")
                }

                session.close()
            }
        }
    }

    @Test
    fun testDifferentialTensorValidation() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val sessCfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)
            val params = SamplingParams(temperature = 0.0f, maxTokens = 1)

            val prompt = "<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n"

            val triggerFile = File(context.filesDir, "mobilellm_tensor_dump")
            val traceFile = File(context.filesDir, "tensor_trace.txt")
            if (traceFile.exists()) traceFile.delete()
            triggerFile.createNewFile()
            Log.i(TAG, "Trigger file created: ${triggerFile.absolutePath}")

            try {
                Log.i(TAG, "================ RUNNING DIFFERENTIAL BASELINE: CPU (L=0) ================")
                val cpuCfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = BackendType.CPU_NEON,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = DEFAULT_THREADS,
                    useMmap = false,
                    nGpuLayers = 0
                )
                val cpuSession = client.loadModel(cpuCfg, sessCfg)
                val cpuChunks = cpuSession.generate(prompt, params).toList()
                val cpuToken = cpuChunks.firstOrNull()?.text ?: ""
                Log.i(TAG, "CPU Step 0 Token: '$cpuToken' (${cpuChunks.firstOrNull()?.tokenId})")
                cpuSession.close()

                Log.i(TAG, "================ RUNNING DIFFERENTIAL EVALUATION: VULKAN (L=28) ================")
                val vkCfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = BackendType.VULKAN,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = DEFAULT_THREADS,
                    useMmap = false,
                    nGpuLayers = 28
                )
                val vkSession = client.loadModel(vkCfg, sessCfg)
                val vkChunks = vkSession.generate(prompt, params).toList()
                val vkToken = vkChunks.firstOrNull()?.text ?: ""
                Log.i(TAG, "Vulkan L=28 Step 0 Token: '$vkToken' (${vkChunks.firstOrNull()?.tokenId})")
                vkSession.close()

                Log.i(TAG, "================ RUNNING DIFFERENTIAL EVALUATION: VULKAN (L=1) ================")
                val vk1Cfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = BackendType.VULKAN,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = DEFAULT_THREADS,
                    useMmap = false,
                    nGpuLayers = 1
                )
                val vk1Session = client.loadModel(vk1Cfg, sessCfg)
                val vk1Chunks = vk1Session.generate(prompt, params).toList()
                val vk1Token = vk1Chunks.firstOrNull()?.text ?: ""
                Log.i(TAG, "Vulkan L=1 Step 0 Token: '$vk1Token' (${vk1Chunks.firstOrNull()?.tokenId})")
                vk1Session.close()

            } finally {
                triggerFile.delete()
                Log.i(TAG, "Trigger file cleaned up")
            }
        }
    }

    @Test
    fun testQwen25Vector2Evaluation() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val cfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = DEFAULT_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val sessCfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            Log.i(TAG, "Loading Qwen2.5 1.5B for Vector 2 evaluation on CPU_NEON...")
            val session = client.loadModel(cfg, sessCfg)
            assertNotNull("Session must not be null", session)

            val prompt = "<|im_start|>user\nWrite a Python function to sort a list of numbers in ascending order.<|im_end|>\n<|im_start|>assistant\n"
            val params = SamplingParams(
                temperature = 0.0f,
                maxTokens = 193
            )

            // Warmup pass
            Log.i(TAG, "Warmup pass...")
            session.generate("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", SamplingParams(temperature = 0.0f, maxTokens = 4)).toList()

            Thread.sleep(1000)

            Log.i(TAG, "Executing Vector 2 generation...")
            var reportedTtft = 0.0
            var reportedPrefillTps = 0.0
            var reportedDecodeTps = 0.0
            var reportedTotalMs = 0.0
            var promptTokens = 0
            var completionTokens = 0

            val chunks = session.generate(prompt, params) { m ->
                reportedTtft = m.timeToFirstTokenMs
                reportedPrefillTps = m.promptTokensPerSecond
                reportedDecodeTps = m.tokensPerSecond
                reportedTotalMs = m.totalDurationMs
                promptTokens = m.promptTokens
                completionTokens = m.completionTokens
            }.toList()

            val (rss, hwm) = readProcessMemoryMb()
            val availRam = getAvailableRamMb(context)
            val thermal = getThermalStatus(context)
            val generatedText = chunks.joinToString("") { it.text }

            Log.i(TAG, "================ VECTOR 2 MOBILELLM EVALUATION ================")
            Log.i(TAG, "Prompt: $prompt")
            Log.i(TAG, "Prompt Tokens: $promptTokens")
            Log.i(TAG, "Completion Tokens: $completionTokens")
            Log.i(TAG, "Prefill Speed: $reportedPrefillTps tok/s")
            Log.i(TAG, "Decode Speed: $reportedDecodeTps tok/s")
            Log.i(TAG, "TTFT: $reportedTtft ms")
            Log.i(TAG, "Total Duration: $reportedTotalMs ms")
            Log.i(TAG, "Process RSS: $rss MB, Peak HWM: $hwm MB, Avail RAM: $availRam MB")
            Log.i(TAG, "Thermal Status: $thermal")
            Log.i(TAG, "Generated Code:\n$generatedText")
            Log.i(TAG, "==============================================================")

            assertTrue("Must generate tokens", chunks.isNotEmpty())
            assertTrue("Generated text must not be empty", generatedText.isNotBlank())
            assertTrue("Generated text should contain function definition", generatedText.contains("def "))
            session.close()
        }
    }

    @Test
    fun testBenchmarkCpuVsVulkanLayerSweep() {
        runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val client = MobileLLMClient.getInstance(context)
        client.initialize()

        val modelFile = getModelFile(context)
        val modelPath = modelFile.absolutePath

        data class BenchmarkTarget(
            val name: String,
            val backend: BackendType,
            val gpuLayers: Int,
            val threads: Int = DEFAULT_THREADS
        )

        val configurations = listOf(
            BenchmarkTarget("CPU_NEON_T4", BackendType.CPU_NEON, 0),
            BenchmarkTarget("VULKAN_L4", BackendType.VULKAN, 4),
            BenchmarkTarget("VULKAN_L8", BackendType.VULKAN, 8),
            BenchmarkTarget("VULKAN_L12", BackendType.VULKAN, 12),
            BenchmarkTarget("VULKAN_L16", BackendType.VULKAN, 16),
            BenchmarkTarget("VULKAN_L20", BackendType.VULKAN, 20),
            BenchmarkTarget("VULKAN_L24", BackendType.VULKAN, 24),
            BenchmarkTarget("VULKAN_L28_FULL", BackendType.VULKAN, 28)
        )

        val rootResult = JSONObject()
        rootResult.put("device", "OnePlus 12R (CPH2691)")
        rootResult.put("soc", "Qualcomm Snapdragon 8 Gen 2 / SM8650")
        rootResult.put("gpu", "Qualcomm Adreno (TM) 750 (Vulkan 1.3)")
        rootResult.put("model", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        rootResult.put("sha256", EXPECTED_SHA256)
        rootResult.put("prompt", PROMPT)
        rootResult.put("max_tokens", MAX_OUTPUT_TOKENS)
        rootResult.put("context_length", CONTEXT_LENGTH)
        rootResult.put("repetitions", NUM_REPETITIONS)

        val configResultsArray = JSONArray()

        for (target in configurations) {
            Log.i(TAG, "\n========================================================")
            Log.i(TAG, "BENCHMARKING: ${target.name} (Backend=${target.backend}, Layers=${target.gpuLayers}, Threads=${target.threads})")
            Log.i(TAG, "========================================================")

            val mcfg = ModelConfig(
                modelPath = modelPath,
                preferredBackend = target.backend,
                contextLength = CONTEXT_LENGTH,
                nThreads = target.threads,
                useMmap = false,
                nGpuLayers = target.gpuLayers
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load for ${target.name}", session)

            val sampling = SamplingParams(
                temperature = 0.0f, // Greedy deterministic
                maxTokens = MAX_OUTPUT_TOKENS
            )

            // 1. Warm-up Iteration
            Log.i(TAG, "Executing 1 warm-up iteration for ${target.name}...")
            var warmupTtft = 0.0
            var warmupPrefillTps = 0.0
            var warmupDecodeTps = 0.0
            val warmupChunks = session.generate(PROMPT, sampling) { m ->
                warmupTtft = m.timeToFirstTokenMs
                warmupPrefillTps = m.promptTokensPerSecond
                warmupDecodeTps = m.tokensPerSecond
            }.toList()
            Log.i(TAG, "Warmup completed (${warmupChunks.size} chunks, TTFT=${warmupTtft}ms, Decode=${warmupDecodeTps} tok/s)")

            // 2. Measured Repetitions
            val decodeTpsList = mutableListOf<Double>()
            val prefillTpsList = mutableListOf<Double>()
            val ttftList = mutableListOf<Double>()
            val totalLatencyList = mutableListOf<Double>()
            val vmRssList = mutableListOf<Double>()
            val vmHwmList = mutableListOf<Double>()
            val availRamList = mutableListOf<Long>()
            val thermalList = mutableListOf<Int>()
            var verifiedTokens = 0
            var lastGenText = ""

            for (rep in 1..NUM_REPETITIONS) {
                // Brief cool-down pause between repetitions to prevent thermal buildup
                Thread.sleep(800)

                var repTtft = 0.0
                var repPrefillTps = 0.0
                var repDecodeTps = 0.0
                var repTotalMs = 0.0
                var repCompletionTokens = 0

                val chunks = session.generate(PROMPT, sampling) { m ->
                    repTtft = m.timeToFirstTokenMs
                    repPrefillTps = m.promptTokensPerSecond
                    repDecodeTps = m.tokensPerSecond
                    repTotalMs = m.totalDurationMs
                    repCompletionTokens = m.completionTokens
                }.toList()

                val genText = chunks.joinToString("") { it.text }
                lastGenText = genText
                if (rep == 1) {
                    Log.i(TAG, "[${target.name}] Output Sample: $genText")
                }

                val (rss, hwm) = readProcessMemoryMb()
                val availRam = getAvailableRamMb(context)
                val thermal = getThermalStatus(context)

                decodeTpsList.add(repDecodeTps)
                prefillTpsList.add(repPrefillTps)
                ttftList.add(repTtft)
                totalLatencyList.add(repTotalMs)
                vmRssList.add(rss)
                vmHwmList.add(hwm)
                availRamList.add(availRam)
                thermalList.add(thermal)
                verifiedTokens = repCompletionTokens

                Log.i(TAG, "[${target.name}] Rep $rep/$NUM_REPETITIONS: TTFT=${repTtft}ms, Prefill=${repPrefillTps} tps, Decode=${repDecodeTps} tps, Latency=${repTotalMs}ms, RSS=${rss}MB, AvailRAM=${availRam}MB, Thermal=$thermal, Tokens=$repCompletionTokens")
            }

            session.close()

            val targetJson = JSONObject()
            targetJson.put("name", target.name)
            targetJson.put("backend", target.backend.name)
            targetJson.put("gpu_layers", target.gpuLayers)
            targetJson.put("threads", target.threads)
            targetJson.put("output_tokens", verifiedTokens)
            targetJson.put("generated_text", lastGenText)
            targetJson.put("decode_tokens_per_sec", computeStats(decodeTpsList))
            targetJson.put("prefill_tokens_per_sec", computeStats(prefillTpsList))
            targetJson.put("ttft_ms", computeStats(ttftList))
            targetJson.put("total_latency_ms", computeStats(totalLatencyList))
            targetJson.put("vm_rss_mb", computeStats(vmRssList))
            targetJson.put("vm_hwm_mb", computeStats(vmHwmList))
            targetJson.put("thermal_status_max", thermalList.maxOrNull() ?: 0)

            val rawReps = JSONArray()
            for (i in 0 until NUM_REPETITIONS) {
                val repObj = JSONObject()
                repObj.put("iteration", i + 1)
                repObj.put("ttft_ms", ttftList[i])
                repObj.put("prefill_tok_per_sec", prefillTpsList[i])
                repObj.put("decode_tok_per_sec", decodeTpsList[i])
                repObj.put("total_duration_ms", totalLatencyList[i])
                repObj.put("vm_rss_mb", vmRssList[i])
                repObj.put("avail_ram_mb", availRamList[i])
                repObj.put("thermal_status", thermalList[i])
                rawReps.put(repObj)
            }
            targetJson.put("raw_iterations", rawReps)

            configResultsArray.put(targetJson)
            rootResult.put("results", configResultsArray)

            // Save intermediate results after each configuration completes
            try {
                val outDir = context.getExternalFilesDir(null) ?: context.filesDir
                val outFile = File(outDir, "agent_10_vulkan_benchmark.json")
                outFile.writeText(rootResult.toString(2))
                val logFile = File(outDir, "benchmark_progress.log")
                logFile.appendText("COMPLETED: ${target.name} - Decode: ${computeStats(decodeTpsList).optDouble("median")} tok/s, Prefill: ${computeStats(prefillTpsList).optDouble("median")} tok/s, TTFT: ${computeStats(ttftList).optDouble("median")} ms, RSS: ${computeStats(vmRssList).optDouble("median")} MB\n")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to write intermediate benchmark result: ${e.message}")
            }
        }

        rootResult.put("results", configResultsArray)
        val jsonStr = rootResult.toString(2)

        // Save to file on device
        val outDir = context.getExternalFilesDir(null) ?: context.filesDir
        val outFile = File(outDir, "agent_10_vulkan_benchmark.json")
        outFile.writeText(jsonStr)
        Log.i(TAG, "Saved benchmark JSON to: ${outFile.absolutePath}")

        // Log one-line marker for easy adb logcat scraping
        Log.i(TAG, "BENCHMARK_MATRIX_JSON_START")
        Log.i(TAG, rootResult.toString())
        Log.i(TAG, "BENCHMARK_MATRIX_JSON_END")
        }
    }
}
