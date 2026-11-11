package ai.codesec.mobilellm

import ai.codesec.mobilellm.models.BackendType
import ai.codesec.mobilellm.models.GenerationMetrics
import ai.codesec.mobilellm.models.ModelConfig
import ai.codesec.mobilellm.models.SamplingParams
import ai.codesec.mobilellm.models.SessionConfig
import ai.codesec.mobilellm.models.TokenChunk
import com.google.common.truth.Truth.assertThat
import org.junit.Assert.assertThrows
import org.junit.Test

class SdkUnitTest {

    @Test
    fun backendType_correctCodesMatchingCApi() {
        assertThat(BackendType.AUTO.code).isEqualTo(0)
        assertThat(BackendType.CPU_NEON.code).isEqualTo(1)
        assertThat(BackendType.VULKAN.code).isEqualTo(2)
        assertThat(BackendType.VENDOR_NPU.code).isEqualTo(3)
    }

    @Test
    fun samplingParams_defaultValuesAreOptimalForMobile() {
        val params = SamplingParams()
        assertThat(params.temperature).isEqualTo(0.7f)
        assertThat(params.topP).isEqualTo(0.9f)
        assertThat(params.topK).isEqualTo(40)
        assertThat(params.minP).isEqualTo(0.05f)
        assertThat(params.repetitionPenalty).isEqualTo(1.1f)
        assertThat(params.maxTokens).isEqualTo(512)
    }

    @Test
    fun samplingParams_validationRejectsInvalidRanges() {
        assertThrows(IllegalArgumentException::class.java) {
            SamplingParams(temperature = -0.1f)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SamplingParams(topP = 1.5f)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SamplingParams(maxTokens = 0)
        }
    }

    @Test
    fun modelConfig_validationRejectsBlankPath() {
        assertThrows(IllegalArgumentException::class.java) {
            ModelConfig(modelPath = "")
        }
        assertThrows(IllegalArgumentException::class.java) {
            ModelConfig(modelPath = "test.gguf", nThreads = 0)
        }
    }

    @Test
    fun sessionConfig_validationRejectsInvalidContext() {
        assertThrows(IllegalArgumentException::class.java) {
            SessionConfig(maxContextTokens = 0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SessionConfig(maxContextTokens = -10)
        }
        val valid = SessionConfig(maxContextTokens = 2048, enablePrefixCaching = true)
        assertThat(valid.maxContextTokens).isEqualTo(2048)
        assertThat(valid.enablePrefixCaching).isTrue()
    }

    @Test
    fun tokenChunk_propertiesPreserved() {
        val chunk = TokenChunk(
            text = "Hello",
            tokenId = 1500,
            isSpecial = false,
            logprob = -0.25f
        )
        assertThat(chunk.text).isEqualTo("Hello")
        assertThat(chunk.tokenId).isEqualTo(1500)
        assertThat(chunk.isSpecial).isFalse()
        assertThat(chunk.logprob).isEqualTo(-0.25f)
    }

    @Test
    fun generationMetrics_calculationsPreserved() {
        val metrics = GenerationMetrics(
            promptTokens = 32,
            completionTokens = 128,
            timeToFirstTokenMs = 95.4,
            tokensPerSecond = 24.8,
            peakRssMb = 480.0
        )
        assertThat(metrics.promptTokens).isEqualTo(32)
        assertThat(metrics.completionTokens).isEqualTo(128)
        assertThat(metrics.timeToFirstTokenMs).isEqualTo(95.4)
        assertThat(metrics.tokensPerSecond).isEqualTo(24.8)
        assertThat(metrics.peakRssMb).isEqualTo(480.0)
    }

    @Test
    fun exception_hasStandardErrorCodes() {
        val ex = MobileLLMException(MobileLLMException.CODE_OUT_OF_MEMORY, "Memory exhausted")
        assertThat(ex.errorCode).isEqualTo(MobileLLMException.CODE_OUT_OF_MEMORY)
        assertThat(ex.message).contains("Memory exhausted")
    }
}

