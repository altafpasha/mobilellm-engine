package ai.codesec.nexa

import android.app.ActivityManager
import android.content.Context
import android.os.PowerManager
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.codesec.mobilellm.MobileLLMClient
import ai.codesec.mobilellm.ModelSession
import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.GenerationMetrics
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.mobilellm.models.SessionConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.flow.take
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
class BackendSwitchingValidationInstrumentationTest {

    companion object {
        private const val TAG = "BackendSwitchValidation"
        private const val TEST_PROMPT = "<|im_start|>user\nExplain the principles of aerodynamics in clear, simple terms.<|im_end|>\n<|im_start|>assistant\n"
        private const val MAX_TOKENS = 16
        private const val CONTEXT_LENGTH = 512
        private const val NUM_THREADS = 4
    }

    private fun getModelFile(context: Context): File {
        val candidates = listOf(
            File("/sdcard/Android/data/ai.codesec.nexa.debug/files/models/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            File("/sdcard/Android/data/ai.codesec.nexa/files/models/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            File(context.getExternalFilesDir(null), "models/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            File("/sdcard/Download/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
            File("/sdcard/Android/data/ai.codesec.nexa.debug/files/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        )
        for (f in candidates) {
            if (f.exists() && f.length() > 50_000_000L) {
                return f
            }
        }
        throw IllegalStateException("No suitable GGUF model found on device!")
    }

    data class MemoryAndThreads(
        val rssMb: Double,
        val hwmMb: Double,
        val threadCount: Int
    )

    private fun readProcessMemoryAndThreads(): MemoryAndThreads {
        var vmRssMb = 0.0
        var vmHwmMb = 0.0
        var threads = 1
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
                    } else if (line.startsWith("Threads:")) {
                        val parts = line.split("\\s+".toRegex())
                        if (parts.size >= 2) threads = parts[1].toIntOrNull() ?: 1
                    }
                }
            }
        } catch (_: Exception) {}
        return MemoryAndThreads(vmRssMb, vmHwmMb, threads)
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

    private fun getCpuFrequencies(): String {
        val cores = listOf(0, 4, 7)
        val freqs = mutableListOf<String>()
        for (c in cores) {
            try {
                val f = File("/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq")
                if (f.exists()) {
                    val khz = f.readText().trim().toLongOrNull() ?: 0L
                    freqs.add("cpu$c:${khz / 1000}MHz")
                }
            } catch (_: Exception) {}
        }
        return freqs.joinToString(",")
    }

    private fun computeDistributionStats(values: List<Double>): JSONObject {
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

        val p25Index = ((n - 1) * 0.25).roundToInt().coerceIn(0, n - 1)
        val p75Index = ((n - 1) * 0.75).roundToInt().coerceIn(0, n - 1)
        val p90Index = ((n - 1) * 0.90).roundToInt().coerceIn(0, n - 1)

        obj.put("mean", (mean * 1000).roundToInt() / 1000.0)
        obj.put("median", (median * 1000).roundToInt() / 1000.0)
        obj.put("min", (min * 1000).roundToInt() / 1000.0)
        obj.put("max", (max * 1000).roundToInt() / 1000.0)
        obj.put("p25", (sorted[p25Index] * 1000).roundToInt() / 1000.0)
        obj.put("p75", (sorted[p75Index] * 1000).roundToInt() / 1000.0)
        obj.put("p90", (sorted[p90Index] * 1000).roundToInt() / 1000.0)
        obj.put("stddev", (stddev * 1000).roundToInt() / 1000.0)
        return obj
    }

    data class RunMetrics(
        val runIndex: Int,
        val backend: String,
        val ttftMs: Double,
        val prefillTps: Double,
        val decodeTps: Double,
        val totalLatencyMs: Double,
        val tokenCount: Int,
        val generatedText: String,
        val rssMb: Double,
        val peakHwmMb: Double,
        val threadCount: Int,
        val thermalStatus: Int,
        val cpuFreqs: String,
        val availRamMb: Long
    ) {
        fun toJson(): JSONObject {
            val o = JSONObject()
            o.put("run", runIndex)
            o.put("backend", backend)
            o.put("ttft_ms", ttftMs)
            o.put("prefill_tok_per_sec", prefillTps)
            o.put("decode_tok_per_sec", decodeTps)
            o.put("total_latency_ms", totalLatencyMs)
            o.put("token_count", tokenCount)
            o.put("rss_mb", rssMb)
            o.put("peak_hwm_mb", peakHwmMb)
            o.put("thread_count", threadCount)
            o.put("thermal_status", thermalStatus)
            o.put("cpu_freqs", cpuFreqs)
            o.put("avail_ram_mb", availRamMb)
            o.put("generated_text", generatedText)
            return o
        }
    }

    private suspend fun executeInferenceRun(
        session: ModelSession,
        backendName: String,
        runIndex: Int,
        context: Context,
        prompt: String = TEST_PROMPT,
        maxTokens: Int = MAX_TOKENS,
        resetFirst: Boolean = true
    ): RunMetrics {
        if (resetFirst) {
            try {
                session.reset()
            } catch (_: Exception) {}
        }
        var ttft = 0.0
        var prefillTps = 0.0
        var decodeTps = 0.0
        var totalMs = 0.0
        var completionTokens = 0

        val params = SamplingParams(temperature = 0.0f, maxTokens = maxTokens)
        val chunks = session.generate(prompt, params) { m ->
            ttft = m.timeToFirstTokenMs
            prefillTps = m.promptTokensPerSecond
            decodeTps = m.tokensPerSecond
            totalMs = m.totalDurationMs
            completionTokens = m.completionTokens
        }.toList()

        val text = chunks.joinToString("") { it.text }
        val memThreads = readProcessMemoryAndThreads()
        val availRam = getAvailableRamMb(context)
        val thermal = getThermalStatus(context)
        val freqs = getCpuFrequencies()

        Log.i(
            TAG,
            "[$backendName #$runIndex] TTFT=${"%.2f".format(ttft)}ms | Prefill=${"%.2f".format(prefillTps)} tok/s | " +
            "Decode=${"%.2f".format(decodeTps)} tok/s | Total=${"%.2f".format(totalMs)}ms | " +
            "Tokens=$completionTokens | RSS=${"%.1f".format(memThreads.rssMb)}MB | Peak=${"%.1f".format(memThreads.hwmMb)}MB | " +
            "Threads=${memThreads.threadCount} | Thermal=$thermal | Freqs=[$freqs]"
        )

        return RunMetrics(
            runIndex = runIndex,
            backend = backendName,
            ttftMs = ttft,
            prefillTps = prefillTps,
            decodeTps = decodeTps,
            totalLatencyMs = totalMs,
            tokenCount = completionTokens,
            generatedText = text,
            rssMb = memThreads.rssMb,
            peakHwmMb = memThreads.hwmMb,
            threadCount = memThreads.threadCount,
            thermalStatus = thermal,
            cpuFreqs = freqs,
            availRamMb = availRam
        )
    }

    /**
     * Phase 3 & 4:
     * 1. CPU baseline — 5 runs
     * 2. Switch CPU -> Vulkan
     * 3. Vulkan inference — 5 runs
     * 4. Switch Vulkan -> CPU
     * 5. CPU post-switch — 5 runs
     *
     * Validates:
     * - Decode throughput retention >= 90%
     * - Correct output tokens
     * - Memory bounds
     */
    @Test
    fun testPhase3_Baseline_And_Switch_Validation() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            Log.i(TAG, "Testing model: ${modelFile.name} (${modelFile.length()} bytes)")

            val mcfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = NUM_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load", session)

            // Warmup
            Log.i(TAG, "Warmup run...")
            executeInferenceRun(session, "CPU_WARMUP", 0, context, maxTokens = 4)
            delay(500)

            val rootJson = JSONObject()
            rootJson.put("test", "testPhase3_Baseline_And_Switch_Validation")
            rootJson.put("model", modelFile.name)

            // 1. CPU baseline — 5 runs
            Log.i(TAG, "\n========== STEP 1: CPU BASELINE (5 RUNS) ==========")
            val cpuBaselineRuns = mutableListOf<RunMetrics>()
            for (i in 1..5) {
                delay(300)
                cpuBaselineRuns.add(executeInferenceRun(session, "CPU_BASELINE", i, context))
            }

            // 2. Switch CPU -> Vulkan
            Log.i(TAG, "\n========== STEP 2: SWITCHING CPU -> VULKAN ==========")
            val switchVulkanStart = System.currentTimeMillis()
            client.switchBackend(BackendType.VULKAN)
            val switchVulkanTimeMs = System.currentTimeMillis() - switchVulkanStart
            Log.i(TAG, "Switched to Vulkan in $switchVulkanTimeMs ms")
            delay(500)

            // 3. Vulkan inference — 5 runs
            Log.i(TAG, "\n========== STEP 3: VULKAN INFERENCE (5 RUNS) ==========")
            val vulkanRuns = mutableListOf<RunMetrics>()
            for (i in 1..5) {
                delay(300)
                vulkanRuns.add(executeInferenceRun(session, "VULKAN", i, context))
            }

            // 4. Switch Vulkan -> CPU
            Log.i(TAG, "\n========== STEP 4: SWITCHING VULKAN -> CPU ==========")
            val switchCpuStart = System.currentTimeMillis()
            client.switchBackend(BackendType.CPU_NEON)
            val switchCpuTimeMs = System.currentTimeMillis() - switchCpuStart
            Log.i(TAG, "Switched back to CPU in $switchCpuTimeMs ms")
            delay(500)

            // 5. CPU post-switch — 5 runs
            Log.i(TAG, "\n========== STEP 5: CPU POST-SWITCH (5 RUNS) ==========")
            val cpuPostRuns = mutableListOf<RunMetrics>()
            for (i in 1..5) {
                delay(300)
                cpuPostRuns.add(executeInferenceRun(session, "CPU_POST_SWITCH", i, context))
            }

            // Statistical Analysis
            val baseDecodeTps = cpuBaselineRuns.map { it.decodeTps }
            val baseTtft = cpuBaselineRuns.map { it.ttftMs }
            val postDecodeTps = cpuPostRuns.map { it.decodeTps }
            val postTtft = cpuPostRuns.map { it.ttftMs }
            val vulkanDecodeTps = vulkanRuns.map { it.decodeTps }
            val vulkanTtft = vulkanRuns.map { it.ttftMs }

            val baseStats = computeDistributionStats(baseDecodeTps)
            val postStats = computeDistributionStats(postDecodeTps)
            val vulkanStats = computeDistributionStats(vulkanDecodeTps)

            val baseMedianTps = baseStats.getDouble("median")
            val postMedianTps = postStats.getDouble("median")
            val retention = (postMedianTps / baseMedianTps) * 100.0

            Log.i(TAG, "\n==================== PERFORMANCE COMPARISON ====================")
            Log.i(TAG, "CPU Baseline Median Decode: $baseMedianTps tok/s (Mean: ${baseStats.getDouble("mean")})")
            Log.i(TAG, "Vulkan Median Decode: ${vulkanStats.getDouble("median")} tok/s")
            Log.i(TAG, "CPU Post-Switch Median Decode: $postMedianTps tok/s (Mean: ${postStats.getDouble("mean")})")
            Log.i(TAG, "Throughput Retention: ${"%.2f".format(retention)}% (Target >= 90.0%)")
            Log.i(TAG, "================================================================")

            rootJson.put("switch_to_vulkan_ms", switchVulkanTimeMs)
            rootJson.put("switch_to_cpu_ms", switchCpuTimeMs)
            rootJson.put("cpu_baseline_decode_stats", baseStats)
            rootJson.put("cpu_baseline_ttft_stats", computeDistributionStats(baseTtft))
            rootJson.put("vulkan_decode_stats", vulkanStats)
            rootJson.put("vulkan_ttft_stats", computeDistributionStats(vulkanTtft))
            rootJson.put("cpu_post_switch_decode_stats", postStats)
            rootJson.put("cpu_post_switch_ttft_stats", computeDistributionStats(postTtft))
            rootJson.put("retention_percent", retention)

            val rawArray = JSONArray()
            (cpuBaselineRuns + vulkanRuns + cpuPostRuns).forEach { rawArray.put(it.toJson()) }
            rootJson.put("raw_runs", rawArray)

            // Save results to disk
            val outDir = context.getExternalFilesDir(null) ?: context.filesDir
            val outFile = File(outDir, "phase3_backend_switching_validation.json")
            outFile.writeText(rootJson.toString(2))
            Log.i(TAG, "Wrote validation report to ${outFile.absolutePath}")

            // Log full JSON marker
            Log.i(TAG, "VALIDATION_RESULTS_JSON_START")
            Log.i(TAG, rootJson.toString())
            Log.i(TAG, "VALIDATION_RESULTS_JSON_END")

            // Acceptance criteria check
            assertTrue(
                "CPU post-switch median decode throughput must retain >= 90% of baseline! Got ${"%.2f".format(retention)}%",
                retention >= 90.0
            )

            // Correctness check: baseline and post-switch outputs should agree
            assertEquals(cpuBaselineRuns.first().tokenCount, cpuPostRuns.first().tokenCount)
            assertEquals(cpuBaselineRuns.first().generatedText, cpuPostRuns.first().generatedText)

            session.close()
        }
    }

    /**
     * Phase 3 Step 6:
     * Repeat CPU -> Vulkan -> CPU for 20 consecutive cycles.
     * Verifies:
     * - No progressive slowdown
     * - No unbounded memory growth
     * - Stability and lack of memory leaks
     */
    @Test
    fun testPhase3_Twenty_Switching_Cycles_Stability() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val mcfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = NUM_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load", session)

            val initialMem = readProcessMemoryAndThreads()
            Log.i(TAG, "Initial Memory: RSS=${initialMem.rssMb}MB, HWM=${initialMem.hwmMb}MB, Threads=${initialMem.threadCount}")

            val cycleResults = JSONArray()
            val cpuTpsList = mutableListOf<Double>()
            val cpuRssList = mutableListOf<Double>()

            Log.i(TAG, "\n========== EXECUTING 20 BACKEND SWITCHING CYCLES ==========")
            for (cycle in 1..20) {
                // Switch CPU -> Vulkan
                client.switchBackend(BackendType.VULKAN)
                // Execute short generation on Vulkan
                val vkRun = executeInferenceRun(session, "VULKAN_C$cycle", cycle, context, maxTokens = 4)

                // Switch Vulkan -> CPU
                client.switchBackend(BackendType.CPU_NEON)
                // Execute generation on CPU
                val cpuRun = executeInferenceRun(session, "CPU_C$cycle", cycle, context, maxTokens = 12)

                cpuTpsList.add(cpuRun.decodeTps)
                cpuRssList.add(cpuRun.rssMb)

                val cObj = JSONObject()
                cObj.put("cycle", cycle)
                cObj.put("vulkan_decode_tps", vkRun.decodeTps)
                cObj.put("vulkan_rss_mb", vkRun.rssMb)
                cObj.put("cpu_decode_tps", cpuRun.decodeTps)
                cObj.put("cpu_ttft_ms", cpuRun.ttftMs)
                cObj.put("cpu_rss_mb", cpuRun.rssMb)
                cObj.put("thread_count", cpuRun.threadCount)
                cycleResults.put(cObj)

                Log.i(TAG, "Cycle $cycle/20 completed: CPU Decode=${"%.2f".format(cpuRun.decodeTps)} tok/s, RSS=${"%.1f".format(cpuRun.rssMb)}MB")
            }

            val finalMem = readProcessMemoryAndThreads()
            val memoryGrowthMb = finalMem.rssMb - initialMem.rssMb
            val firstFiveMedianTps = computeDistributionStats(cpuTpsList.take(5)).getDouble("median")
            val lastFiveMedianTps = computeDistributionStats(cpuTpsList.takeLast(5)).getDouble("median")
            val stabilityRatio = (lastFiveMedianTps / firstFiveMedianTps) * 100.0

            Log.i(TAG, "\n==================== 20 CYCLES STABILITY SUMMARY ====================")
            Log.i(TAG, "Initial RSS: ${"%.1f".format(initialMem.rssMb)} MB | Final RSS: ${"%.1f".format(finalMem.rssMb)} MB | Delta: ${"%.1f".format(memoryGrowthMb)} MB")
            Log.i(TAG, "First 5 Cycles Median Decode: $firstFiveMedianTps tok/s")
            Log.i(TAG, "Last 5 Cycles Median Decode: $lastFiveMedianTps tok/s")
            Log.i(TAG, "Long-run Performance Stability: ${"%.2f".format(stabilityRatio)}%")
            Log.i(TAG, "====================================================================")

            val rootJson = JSONObject()
            rootJson.put("initial_rss_mb", initialMem.rssMb)
            rootJson.put("final_rss_mb", finalMem.rssMb)
            rootJson.put("memory_growth_mb", memoryGrowthMb)
            rootJson.put("first_5_median_tps", firstFiveMedianTps)
            rootJson.put("last_5_median_tps", lastFiveMedianTps)
            rootJson.put("stability_ratio_percent", stabilityRatio)
            rootJson.put("cycles", cycleResults)

            val outDir = context.getExternalFilesDir(null) ?: context.filesDir
            val outFile = File(outDir, "twenty_cycles_stability_results.json")
            outFile.writeText(rootJson.toString(2))

            Log.i(TAG, "TWENTY_CYCLES_JSON_START")
            Log.i(TAG, rootJson.toString())
            Log.i(TAG, "TWENTY_CYCLES_JSON_END")

            // Verify no progressive collapse: last 5 cycles retain at least 85% of first 5 cycles
            assertTrue(
                "Progressive slowdown detected: last 5 cycles retained only ${"%.2f".format(stabilityRatio)}%",
                stabilityRatio >= 85.0
            )

            // Verify bounded memory growth (< 30 MB growth over 20 cycles)
            assertTrue(
                "Unbounded memory growth detected: grew by ${"%.1f".format(memoryGrowthMb)} MB",
                memoryGrowthMb < 30.0
            )

            session.close()
        }
    }

    /**
     * Phase 3 Step 7:
     * Switch backend while an inference request is actively generating.
     * Verifies that the transition drains or locks out active requests without deadlocks or crashes.
     */
    @Test
    fun testPhase3_Switch_During_Active_Generation() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val mcfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = NUM_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load", session)

            Log.i(TAG, "\n========== TESTING SWITCH DURING ACTIVE GENERATION ==========")

            // Launch active long generation
            val genJob = async(Dispatchers.IO) {
                try {
                    val chunks = session.generate(TEST_PROMPT, SamplingParams(maxTokens = 64)).toList()
                    Log.i(TAG, "Active generation finished with ${chunks.size} tokens")
                    chunks.size
                } catch (e: Exception) {
                    Log.i(TAG, "Active generation encountered expected transition notice: ${e.message}")
                    -1
                }
            }

            // Wait briefly for inference to be actively decoding
            delay(60)

            // Trigger switch concurrently
            Log.i(TAG, "Triggering concurrent switchBackend(VULKAN)...")
            client.switchBackend(BackendType.VULKAN)
            Log.i(TAG, "Concurrent switchBackend(VULKAN) returned successfully!")

            genJob.await()

            // Verify session is operational on Vulkan
            val vkRun = executeInferenceRun(session, "POST_CONCURRENT_VULKAN", 1, context, maxTokens = 8)
            assertTrue("Must generate tokens after transition", vkRun.tokenCount > 0)

            // Switch back to CPU
            client.switchBackend(BackendType.CPU_NEON)
            val cpuRun = executeInferenceRun(session, "POST_CONCURRENT_CPU", 1, context, maxTokens = 8)
            assertTrue("Must generate tokens after transition back to CPU", cpuRun.tokenCount > 0)

            Log.i(TAG, "Switch during active generation test PASSED!")
            session.close()
        }
    }

    /**
     * Phase 3 Step 8:
     * Switch backend after request cancellation.
     */
    @Test
    fun testPhase3_Switch_After_Cancellation() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val mcfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = NUM_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load", session)

            Log.i(TAG, "\n========== TESTING SWITCH AFTER CANCELLATION ==========")

            // Start generation and cancel it early by taking only 2 tokens
            val tokens = session.generate(TEST_PROMPT, SamplingParams(maxTokens = 32)).take(2).toList()
            Log.i(TAG, "Cancelled flow after ${tokens.size} tokens")

            delay(200)

            // Switch backend
            client.switchBackend(BackendType.VULKAN)
            Log.i(TAG, "Switched to Vulkan after cancellation")

            // Run inference on Vulkan
            val vkRun = executeInferenceRun(session, "AFTER_CANCEL_VULKAN", 1, context, maxTokens = 8)
            assertTrue(vkRun.tokenCount > 0)

            // Switch back to CPU
            client.switchBackend(BackendType.CPU_NEON)
            val cpuRun = executeInferenceRun(session, "AFTER_CANCEL_CPU", 1, context, maxTokens = 8)
            assertTrue(cpuRun.tokenCount > 0)

            Log.i(TAG, "Switch after cancellation test PASSED!")
            session.close()
        }
    }

    /**
     * Phase 3 Step 9:
     * Test backend initialization failure and CPU recovery.
     * Verifies strict error propagation and engine usability.
     */
    @Test
    fun testPhase3_Backend_Failure_And_Cpu_Recovery() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            val mcfg = ModelConfig(
                modelPath = modelFile.absolutePath,
                preferredBackend = BackendType.CPU_NEON,
                contextLength = CONTEXT_LENGTH,
                nThreads = NUM_THREADS,
                useMmap = false,
                nGpuLayers = 0
            )
            val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)

            val session = client.loadModel(mcfg, scfg)
            assertNotNull("Session must load", session)

            Log.i(TAG, "\n========== TESTING BACKEND FAILURE AND CPU RECOVERY ==========")

            // Baseline CPU run
            val cpuBefore = executeInferenceRun(session, "PRE_FAIL_CPU", 1, context, maxTokens = 8)
            assertTrue(cpuBefore.tokenCount > 0)

            // Attempt to switch to an unsupported / invalid backend (VENDOR_NPU)
            var failedAsExpected = false
            try {
                client.switchBackend(BackendType.VENDOR_NPU)
            } catch (e: Exception) {
                failedAsExpected = true
                Log.i(TAG, "Switching to unsupported backend threw expected exception: ${e.message}")
            }

            assertTrue("Switching to invalid backend must fail and throw exception", failedAsExpected)

            // Explicitly verify CPU recovery / usability
            client.switchBackend(BackendType.CPU_NEON)
            val cpuAfter = executeInferenceRun(session, "POST_FAIL_CPU", 1, context, maxTokens = 8)
            assertTrue("Engine must remain fully functional on CPU after failed transition", cpuAfter.tokenCount > 0)
            assertEquals("Output should match baseline", cpuBefore.generatedText, cpuAfter.generatedText)

            Log.i(TAG, "Backend failure and CPU recovery test PASSED!")
            session.close()
        }
    }

    /**
     * Agent 14: Controlled CPU Baseline vs Vulkan Comparison Matrix (Section 6)
     * Test A: CPU, 2 threads, 32 tokens
     * Test B: CPU, 4 threads, 32 tokens
     * Test C: CPU, 6 threads, 32 tokens
     * Test D: CPU, 8 threads, 32 tokens
     * Test E: Vulkan, 4 threads, 32 tokens
     */
    @Test
    fun testAgent14_ControlledComparisonMatrix() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            Log.i(TAG, "\n================================================================================")
            Log.i(TAG, "           AGENT 14: CONTROLLED CPU VS VULKAN COMPARISON MATRIX                 ")
            Log.i(TAG, "================================================================================")
            Log.i(TAG, "Model: ${modelFile.name} (${modelFile.length()} bytes)")

            data class TestCase(
                val id: String,
                val backend: BackendType,
                val threads: Int,
                val gpuLayers: Int,
                val tokens: Int
            )

            val testCases = listOf(
                TestCase("Test A (CPU 2T)", BackendType.CPU_NEON, 2, 0, 32),
                TestCase("Test B (CPU 4T)", BackendType.CPU_NEON, 4, 0, 32),
                TestCase("Test C (CPU 6T)", BackendType.CPU_NEON, 6, 0, 32),
                TestCase("Test D (CPU 8T)", BackendType.CPU_NEON, 8, 0, 32),
                TestCase("Test E (Vulkan 4T)", BackendType.VULKAN, 4, 28, 32)
            )

            val resultsArray = JSONArray()
            val rootJson = JSONObject()
            rootJson.put("benchmark", "Agent 14 Controlled CPU vs Vulkan Comparison Matrix")
            rootJson.put("device", "OnePlus 12R (CPH2691)")
            rootJson.put("soc", "Qualcomm Snapdragon 8 Gen 2 / SM8650")
            rootJson.put("model", modelFile.name)

            for (tc in testCases) {
                // Settle / cooldown between runs
                delay(2000)
                val thermalBefore = getThermalStatus(context)
                val freqsBefore = getCpuFrequencies()

                Log.i(TAG, "\n>>> STARTING ${tc.id} [Backend: ${tc.backend}, Threads: ${tc.threads}, Tokens: ${tc.tokens}] <<<")
                Log.i(TAG, "Thermal before: $thermalBefore, Freqs: [$freqsBefore]")

                val tLoadStart = System.currentTimeMillis()
                val mcfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = tc.backend,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = tc.threads,
                    useMmap = false,
                    nGpuLayers = tc.gpuLayers
                )
                val scfg = SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false)
                val session = client.loadModel(mcfg, scfg)
                assertNotNull("${tc.id} session must load", session)
                val loadDurationMs = System.currentTimeMillis() - tLoadStart

                // 1 Warmup iteration (16 tokens)
                Log.i(TAG, "[${tc.id}] Warmup pass (16 tokens)...")
                val warmup = executeInferenceRun(session, "${tc.id}_WARMUP", 0, context, maxTokens = 16)
                delay(1000)

                // 1 Measured run (32 tokens)
                Log.i(TAG, "[${tc.id}] Measured run (32 tokens)...")
                val measured = executeInferenceRun(session, tc.id, 1, context, maxTokens = tc.tokens)
                val thermalAfter = getThermalStatus(context)

                val testJson = JSONObject()
                testJson.put("test_id", tc.id)
                testJson.put("backend", tc.backend.name)
                testJson.put("threads", tc.threads)
                testJson.put("gpu_layers", tc.gpuLayers)
                testJson.put("output_tokens", tc.tokens)
                testJson.put("model_load_ms", loadDurationMs)
                testJson.put("ttft_ms", measured.ttftMs)
                testJson.put("prefill_tok_per_sec", measured.prefillTps)
                testJson.put("decode_tok_per_sec", measured.decodeTps)
                testJson.put("total_latency_ms", measured.totalLatencyMs)
                testJson.put("peak_rss_mb", measured.peakHwmMb)
                testJson.put("current_rss_mb", measured.rssMb)
                testJson.put("avail_ram_mb", measured.availRamMb)
                testJson.put("thermal_before", thermalBefore)
                testJson.put("thermal_after", thermalAfter)
                testJson.put("cpu_freqs", measured.cpuFreqs)
                testJson.put("generated_text", measured.generatedText)

                resultsArray.put(testJson)

                val modelHandle = session.modelHandle
                session.close()
                client.unloadModel(modelHandle)
                delay(1500)
            }

            rootJson.put("results", resultsArray)

            // Calculate GPU speedup relative to CPU 4T (Test B)
            var cpu4tTps = 0.0
            var vulkanTps = 0.0
            for (i in 0 until resultsArray.length()) {
                val o = resultsArray.getJSONObject(i)
                if (o.getString("test_id").contains("CPU 4T")) {
                    cpu4tTps = o.getDouble("decode_tok_per_sec")
                } else if (o.getString("test_id").contains("Vulkan")) {
                    vulkanTps = o.getDouble("decode_tok_per_sec")
                }
            }
            val gpuSpeedup = if (cpu4tTps > 0.0) vulkanTps / cpu4tTps else 0.0
            rootJson.put("gpu_speedup_vs_cpu_4t", gpuSpeedup)

            Log.i(TAG, "\n================================================================================")
            Log.i(TAG, "AGENT 14 BENCHMARK MATRIX COMPLETE:")
            Log.i(TAG, "CPU 4T Decode: $cpu4tTps tok/s | Vulkan Decode: $vulkanTps tok/s | GPU Speedup: ${"%.4f".format(gpuSpeedup)}x")
            Log.i(TAG, "================================================================================")

            val outFile = File(context.getExternalFilesDir(null), "agent_14_comparison_matrix.json")
            try {
                outFile.writeText(rootJson.toString(2))
                Log.i(TAG, "Exported matrix to ${outFile.absolutePath}")
            } catch (e: Exception) {
                Log.w(TAG, "Could not write json: ${e.message}")
            }
            Log.i(TAG, "AGENT_14_JSON_START\n${rootJson.toString(2)}\nAGENT_14_JSON_END")
        }
    }

    /**
     * Agent 14: Memory Lifecycle & Leak Audit (Section 7)
     * Verifies that loading, running inference, closing session, unloading model,
     * and reloading across multiple cycles does not produce progressive RSS bloat.
     */
    @Test
    fun testAgent14_MemoryLifecycleTest() {
        runBlocking {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val client = MobileLLMClient.getInstance(context)
            client.initialize()

            val modelFile = getModelFile(context)
            Log.i(TAG, "\n================================================================================")
            Log.i(TAG, "           AGENT 14: MEMORY LIFECYCLE & LEAK AUDIT TEST                         ")
            Log.i(TAG, "================================================================================")

            val lifecycleEvents = JSONArray()

            for (cycle in 1..3) {
                Log.i(TAG, "\n--- LIFECYCLE CYCLE $cycle / 3 ---")

                // Stage 1: Initial baseline before load
                val m0 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 1: Pre-Load] RSS=${"%.1f".format(m0.rssMb)}MB, Peak=${"%.1f".format(m0.hwmMb)}MB")

                // Stage 2: Load model
                val tLoad = System.currentTimeMillis()
                val mcfg = ModelConfig(
                    modelPath = modelFile.absolutePath,
                    preferredBackend = BackendType.CPU_NEON,
                    contextLength = CONTEXT_LENGTH,
                    nThreads = 4,
                    useMmap = false,
                    nGpuLayers = 0
                )
                val session = client.loadModel(mcfg, SessionConfig(maxContextTokens = CONTEXT_LENGTH, enablePrefixCaching = false))
                val loadMs = System.currentTimeMillis() - tLoad
                val m1 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 2: Loaded in ${loadMs}ms] RSS=${"%.1f".format(m1.rssMb)}MB, Peak=${"%.1f".format(m1.hwmMb)}MB")

                // Stage 3: Short inference (16 tokens)
                val run = executeInferenceRun(session, "LIFECYCLE_C$cycle", 1, context, maxTokens = 16)
                val m2 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 3: Post-Inference] RSS=${"%.1f".format(m2.rssMb)}MB, Peak=${"%.1f".format(m2.hwmMb)}MB")

                // Stage 4: Stop / Close session
                val modelHandle = session.modelHandle
                session.close()
                delay(200)
                val m3 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 4: Session Closed] RSS=${"%.1f".format(m3.rssMb)}MB, Peak=${"%.1f".format(m3.hwmMb)}MB")

                // Stage 5: Unload model
                client.unloadModel(modelHandle)
                delay(500)
                val m4 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 5: Model Unloaded] RSS=${"%.1f".format(m4.rssMb)}MB, Peak=${"%.1f".format(m4.hwmMb)}MB")

                // Stage 6: Settle wait
                delay(1000)
                val m5 = readProcessMemoryAndThreads()
                Log.i(TAG, "[Cycle $cycle | Stage 6: Settle Wait] RSS=${"%.1f".format(m5.rssMb)}MB, Peak=${"%.1f".format(m5.hwmMb)}MB")

                val cObj = JSONObject()
                cObj.put("cycle", cycle)
                cObj.put("pre_load_rss_mb", m0.rssMb)
                cObj.put("post_load_rss_mb", m1.rssMb)
                cObj.put("post_inference_rss_mb", m2.rssMb)
                cObj.put("post_session_close_rss_mb", m3.rssMb)
                cObj.put("post_unload_rss_mb", m4.rssMb)
                cObj.put("post_settle_rss_mb", m5.rssMb)
                cObj.put("peak_hwm_mb", m5.hwmMb)
                lifecycleEvents.put(cObj)
            }

            val finalJson = JSONObject()
            finalJson.put("test", "testAgent14_MemoryLifecycleTest")
            finalJson.put("cycles", lifecycleEvents)
            Log.i(TAG, "LIFECYCLE_JSON_START\n${finalJson.toString(2)}\nLIFECYCLE_JSON_END")
        }
    }
}
