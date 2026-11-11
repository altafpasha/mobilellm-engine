package ai.codesec.mobilellm.models

/**
 * Hardware compute backend targets for inference execution.
 */
enum class BackendType(val code: Int) {
    /**
     * Automatic runtime hardware detection and capability gating.
     * Prefers Vulkan GPU compute when supported; falls back safely to ARM NEON CPU.
     */
    AUTO(0),

    /**
     * ARM64 CPU compute optimized with ARM NEON SIMD intrinsics.
     * The foundational golden reference backend.
     */
    CPU_NEON(1),

    /**
     * Vulkan compute backend utilizing mobile GPU compute shaders.
     */
    VULKAN(2),

    /**
     * Experimental vendor NPU integration (Qualcomm QNN / Android NNAPI).
     */
    VENDOR_NPU(3);

    companion object {
        fun fromCode(code: Int): BackendType =
            entries.find { it.code == code } ?: AUTO
    }
}
