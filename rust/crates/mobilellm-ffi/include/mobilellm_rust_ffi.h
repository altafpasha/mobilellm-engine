/**
 * @file mobilellm_rust_ffi.h
 * @brief Stable C-ABI boundary for MobileLLM Rust subsystem.
 *
 * All functions are thread-safe unless otherwise documented.
 * All functions catch internal panics and return explicit status codes.
 * No C++ exceptions or Rust panics cross this ABI boundary.
 */

#ifndef MOBILELLM_RUST_FFI_H
#define MOBILELLM_RUST_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle representing the Rust GPU and diagnostic subsystem context.
 */
typedef struct mllm_rust_gpu_handle mllm_rust_gpu_handle_t;

/**
 * @brief Standard return status structure for Rust FFI calls.
 *
 * code == 0 indicates success. Non-zero indicates failure.
 */
typedef struct {
    int32_t code;
    char message[256];
} mllm_rust_status_t;

/**
 * @brief Fixed-size C representation of GPU capabilities.
 */
typedef struct {
    uint32_t vendor_id;
    uint32_t device_id;
    char device_name[256];
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t subgroup_size;
    uint64_t device_local_memory_bytes;
    bool has_fp16;
    bool has_int8;
    bool has_subgroups;
    bool has_subgroup_arithmetic;
} mllm_rust_gpu_capabilities_t;

/**
 * @brief Diagnostic execution outcome.
 */
typedef struct {
    bool execution_passed;
    float execution_time_ms;
    float max_absolute_error;
    char details[256];
} mllm_rust_diagnostic_result_t;

/**
 * @brief Allocate and initialize the Rust GPU subsystem.
 *
 * @param handle_out Pointer to receive the allocated opaque handle.
 * @return Status code and description.
 */
mllm_rust_status_t mllm_rust_gpu_initialize(
    mllm_rust_gpu_handle_t** handle_out
);

/**
 * @brief Query GPU hardware specifications and capabilities.
 *
 * @param handle Valid handle returned by mllm_rust_gpu_initialize.
 * @param caps_out Pointer to receive capabilities struct.
 * @return Status code and description.
 */
mllm_rust_status_t mllm_rust_gpu_query_capabilities(
    const mllm_rust_gpu_handle_t* handle,
    mllm_rust_gpu_capabilities_t* caps_out
);

/**
 * @brief Run an isolated deterministic micro-compute diagnostic pass.
 *
 * @param handle Valid handle returned by mllm_rust_gpu_initialize.
 * @param diag_out Pointer to receive diagnostic metrics.
 * @return Status code and description.
 */
mllm_rust_status_t mllm_rust_gpu_run_diagnostic(
    const mllm_rust_gpu_handle_t* handle,
    mllm_rust_diagnostic_result_t* diag_out
);

/**
 * @brief Release and destroy the Rust GPU context.
 *
 * Safe to call with NULL.
 *
 * @param handle Handle to destroy.
 */
void mllm_rust_gpu_destroy(
    mllm_rust_gpu_handle_t* handle
);

#ifdef __cplusplus
}
#endif

#endif /* MOBILELLM_RUST_FFI_H */
