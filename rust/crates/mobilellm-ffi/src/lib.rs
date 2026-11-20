//! MobileLLM Rust FFI Boundary
//!
//! Provides an internal C ABI boundary for MobileLLM.
//! All calls use opaque handles, explicit allocation/destruction, strict null
//! pointer validation, and panic-catching guards.

use mobilellm_core::{ErrorCode, GpuDeviceInfo};
use mobilellm_diagnostics::DiagnosticsRunner;
use mobilellm_gpu::GpuContext;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Opaque handle representing the Rust GPU and diagnostic subsystem.
pub struct mllm_rust_gpu_handle {
    ctx: GpuContext,
}

#[repr(C)]
pub struct mllm_rust_status_t {
    pub code: i32,
    pub message: [c_char; 256],
}

#[repr(C)]
pub struct mllm_rust_gpu_capabilities_t {
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_name: [c_char; 256],
    pub api_version: u32,
    pub driver_version: u32,
    pub subgroup_size: u32,
    pub device_local_memory_bytes: u64,
    pub has_fp16: bool,
    pub has_int8: bool,
    pub has_subgroups: bool,
    pub has_subgroup_arithmetic: bool,
}

#[repr(C)]
pub struct mllm_rust_diagnostic_result_t {
    pub execution_passed: bool,
    pub execution_time_ms: f32,
    pub max_absolute_error: f32,
    pub details: [c_char; 256],
}

fn write_c_string(dest: &mut [c_char; 256], src: &str) {
    let bytes = src.as_bytes();
    let copy_len = bytes.len().min(255);
    for i in 0..copy_len {
        dest[i] = bytes[i] as c_char;
    }
    dest[copy_len] = 0;
}

fn make_status(code: ErrorCode, msg: &str) -> mllm_rust_status_t {
    let mut status = mllm_rust_status_t {
        code: code as i32,
        message: [0; 256],
    };
    write_c_string(&mut status.message, msg);
    status
}

/// Allocate and initialize the Rust GPU context.
#[no_mangle]
pub unsafe extern "C" fn mllm_rust_gpu_initialize(
    handle_out: *mut *mut mllm_rust_gpu_handle,
) -> mllm_rust_status_t {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle_out.is_null() {
            return make_status(
                ErrorCode::InvalidArgument,
                "handle_out pointer must not be null",
            );
        }

        match GpuContext::new() {
            Ok(ctx) => {
                let boxed = Box::new(mllm_rust_gpu_handle { ctx });
                *handle_out = Box::into_raw(boxed);
                make_status(ErrorCode::Success, "Rust GPU subsystem initialized successfully")
            }
            Err(e) => make_status(e.code, &e.message),
        }
    }));

    match result {
        Ok(status) => status,
        Err(_) => make_status(
            ErrorCode::InternalPanic,
            "Internal panic caught during mllm_rust_gpu_initialize",
        ),
    }
}

/// Query hardware capabilities and device specifications.
#[no_mangle]
pub unsafe extern "C" fn mllm_rust_gpu_query_capabilities(
    handle: *const mllm_rust_gpu_handle,
    caps_out: *mut mllm_rust_gpu_capabilities_t,
) -> mllm_rust_status_t {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return make_status(ErrorCode::InvalidArgument, "Handle must not be null");
        }
        if caps_out.is_null() {
            return make_status(
                ErrorCode::InvalidArgument,
                "caps_out destination must not be null",
            );
        }

        let rust_handle = &*handle;
        match rust_handle.ctx.query_device_info() {
            Ok(info) => {
                let mut caps = mllm_rust_gpu_capabilities_t {
                    vendor_id: info.vendor_id,
                    device_id: info.device_id,
                    device_name: [0; 256],
                    api_version: info.api_version,
                    driver_version: info.driver_version,
                    subgroup_size: info.subgroup_size,
                    device_local_memory_bytes: info.device_local_memory_bytes,
                    has_fp16: info.capabilities.has_fp16,
                    has_int8: info.capabilities.has_int8,
                    has_subgroups: info.capabilities.has_subgroups,
                    has_subgroup_arithmetic: info.capabilities.has_subgroup_arithmetic,
                };
                write_c_string(&mut caps.device_name, &info.device_name);
                *caps_out = caps;
                make_status(ErrorCode::Success, "Capabilities queried successfully")
            }
            Err(e) => make_status(e.code, &e.message),
        }
    }));

    match result {
        Ok(status) => status,
        Err(_) => make_status(
            ErrorCode::InternalPanic,
            "Internal panic caught during mllm_rust_gpu_query_capabilities",
        ),
    }
}

/// Run an isolated deterministic micro-compute diagnostic pass.
#[no_mangle]
pub unsafe extern "C" fn mllm_rust_gpu_run_diagnostic(
    handle: *const mllm_rust_gpu_handle,
    diag_out: *mut mllm_rust_diagnostic_result_t,
) -> mllm_rust_status_t {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return make_status(ErrorCode::InvalidArgument, "Handle must not be null");
        }
        if diag_out.is_null() {
            return make_status(
                ErrorCode::InvalidArgument,
                "diag_out destination must not be null",
            );
        }

        let rust_handle = &*handle;
        let test_input = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let mut test_output = [0.0f32; 4];

        match rust_handle
            .ctx
            .run_deterministic_compute(&test_input, &mut test_output)
        {
            Ok(metrics) => {
                let mut diag = mllm_rust_diagnostic_result_t {
                    execution_passed: metrics.verified_correct,
                    execution_time_ms: metrics.duration_ms as f32,
                    max_absolute_error: metrics.max_absolute_error,
                    details: [0; 256],
                };
                write_c_string(
                    &mut diag.details,
                    &format!(
                        "Micro-compute passed: error={:.1e}, duration={:.3}ms",
                        metrics.max_absolute_error, metrics.duration_ms
                    ),
                );
                *diag_out = diag;
                make_status(ErrorCode::Success, "Diagnostic evaluated successfully")
            }
            Err(e) => make_status(e.code, &e.message),
        }
    }));

    match result {
        Ok(status) => status,
        Err(_) => make_status(
            ErrorCode::InternalPanic,
            "Internal panic caught during mllm_rust_gpu_run_diagnostic",
        ),
    }
}

/// Destroy and release a Rust GPU context handle.
#[no_mangle]
pub unsafe extern "C" fn mllm_rust_gpu_destroy(handle: *mut mllm_rust_gpu_handle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(handle));
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_ffi_null_pointer_rejection() {
        unsafe {
            let status = mllm_rust_gpu_initialize(ptr::null_mut());
            assert_eq!(status.code, ErrorCode::InvalidArgument as i32);

            let mut caps = std::mem::zeroed();
            let status_caps = mllm_rust_gpu_query_capabilities(ptr::null(), &mut caps);
            assert_eq!(status_caps.code, ErrorCode::InvalidArgument as i32);

            let status_caps2 = mllm_rust_gpu_query_capabilities(ptr::null(), ptr::null_mut());
            assert_eq!(status_caps2.code, ErrorCode::InvalidArgument as i32);

            let mut diag = std::mem::zeroed();
            let status_diag = mllm_rust_gpu_run_diagnostic(ptr::null(), &mut diag);
            assert_eq!(status_diag.code, ErrorCode::InvalidArgument as i32);

            // Null destroy must be a safe no-op
            mllm_rust_gpu_destroy(ptr::null_mut());
        }
    }

    #[test]
    fn test_ffi_full_lifecycle() {
        unsafe {
            let mut handle: *mut mllm_rust_gpu_handle = ptr::null_mut();
            let init_status = mllm_rust_gpu_initialize(&mut handle);
            assert_eq!(init_status.code, 0);
            assert!(!handle.is_null());

            let mut caps = std::mem::zeroed();
            let query_status = mllm_rust_gpu_query_capabilities(handle, &mut caps);
            assert_eq!(query_status.code, 0);
            assert!(caps.subgroup_size > 0);

            let mut diag = std::mem::zeroed();
            let diag_status = mllm_rust_gpu_run_diagnostic(handle, &mut diag);
            assert_eq!(diag_status.code, 0);
            assert!(diag.execution_passed);

            mllm_rust_gpu_destroy(handle);
        }
    }
}
