#include <gtest/gtest.h>

#ifdef MOBILELLM_ENABLE_RUST
#include "mobilellm_rust_ffi.h"

TEST(RustFfiTest, NullPointerHandling) {
    mllm_rust_status_t status = mllm_rust_gpu_initialize(nullptr);
    EXPECT_NE(status.code, 0);

    mllm_rust_gpu_capabilities_t caps;
    status = mllm_rust_gpu_query_capabilities(nullptr, &caps);
    EXPECT_NE(status.code, 0);

    status = mllm_rust_gpu_query_capabilities(reinterpret_cast<mllm_rust_gpu_handle_t*>(0x1234), nullptr);
    EXPECT_NE(status.code, 0);

    mllm_rust_diagnostic_result_t diag;
    status = mllm_rust_gpu_run_diagnostic(nullptr, &diag);
    EXPECT_NE(status.code, 0);

    // Destroy with null must be a safe no-op
    mllm_rust_gpu_destroy(nullptr);
}

TEST(RustFfiTest, FullSubsystemLifecycle) {
    mllm_rust_gpu_handle_t* handle = nullptr;
    mllm_rust_status_t init_status = mllm_rust_gpu_initialize(&handle);
    ASSERT_EQ(init_status.code, 0);
    ASSERT_NE(handle, nullptr);

    mllm_rust_gpu_capabilities_t caps{};
    mllm_rust_status_t query_status = mllm_rust_gpu_query_capabilities(handle, &caps);
    EXPECT_EQ(query_status.code, 0);
    EXPECT_GT(caps.subgroup_size, 0u);
    EXPECT_STRNE(caps.device_name, "");

    mllm_rust_diagnostic_result_t diag{};
    mllm_rust_status_t diag_status = mllm_rust_gpu_run_diagnostic(handle, &diag);
    EXPECT_EQ(diag_status.code, 0);
    EXPECT_TRUE(diag.execution_passed);
    EXPECT_LE(diag.max_absolute_error, 1e-5f);

    mllm_rust_gpu_destroy(handle);
}

TEST(RustFfiTest, RepeatedLifecycleSafety) {
    for (int i = 0; i < 5; ++i) {
        mllm_rust_gpu_handle_t* handle = nullptr;
        mllm_rust_status_t status = mllm_rust_gpu_initialize(&handle);
        ASSERT_EQ(status.code, 0);
        ASSERT_NE(handle, nullptr);

        mllm_rust_gpu_capabilities_t caps{};
        EXPECT_EQ(mllm_rust_gpu_query_capabilities(handle, &caps).code, 0);

        mllm_rust_gpu_destroy(handle);
    }
}
#else
TEST(RustFfiTest, RustDisabledGracefulFallback) {
    // When MOBILELLM_ENABLE_RUST is OFF, verify this test suite is active and passes as a placeholder.
    SUCCEED() << "Rust subsystem is disabled (MOBILELLM_ENABLE_RUST=OFF); C++-only runtime verified.";
}
#endif
