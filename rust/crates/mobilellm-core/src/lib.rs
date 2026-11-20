//! MobileLLM Core Rust Subsystem
//!
//! Provides fundamental error types, hardware capability models, and
//! diagnostic structures. Does NOT duplicate C++ scheduler, tokenizer,
//! model loader, or sampling algorithms.

use std::fmt;

/// Standard status codes aligned across C++ and Rust boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    Success = 0,
    InvalidArgument = 1,
    VulkanUnavailable = 2,
    InitializationFailed = 3,
    AllocationFailed = 4,
    ExecutionFailed = 5,
    DeviceLost = 6,
    UnsupportedFeature = 7,
    InternalPanic = 99,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::Success => "Success",
            ErrorCode::InvalidArgument => "Invalid argument or null pointer",
            ErrorCode::VulkanUnavailable => "Vulkan runtime loader or driver unavailable",
            ErrorCode::InitializationFailed => "Subsystem initialization failed",
            ErrorCode::AllocationFailed => "GPU buffer or memory allocation failed",
            ErrorCode::ExecutionFailed => "Compute pipeline execution failure",
            ErrorCode::DeviceLost => "Vulkan physical or logical device lost",
            ErrorCode::UnsupportedFeature => "Requested feature unsupported on this device",
            ErrorCode::InternalPanic => "Internal panic caught at ABI boundary",
        }
    }
}

/// Structured error type for MobileLLM Rust components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, msg)
    }

    pub fn vulkan_unavailable(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::VulkanUnavailable, msg)
    }

    pub fn init_failed(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::InitializationFailed, msg)
    }

    pub fn alloc_failed(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::AllocationFailed, msg)
    }

    pub fn exec_failed(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::ExecutionFailed, msg)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Hardware specifications of an enumerated GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_name: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub subgroup_size: u32,
    pub device_local_memory_bytes: u64,
    pub capabilities: ExecutionCapabilities,
}

impl Default for GpuDeviceInfo {
    fn default() -> Self {
        Self {
            vendor_id: 0,
            device_id: 0,
            device_name: "Unknown / Fallback Device".to_string(),
            api_version: 0,
            driver_version: 0,
            subgroup_size: 32,
            device_local_memory_bytes: 0,
            capabilities: ExecutionCapabilities::default(),
        }
    }
}

/// Compute and shader execution capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionCapabilities {
    pub has_fp16: bool,
    pub has_int8: bool,
    pub has_subgroups: bool,
    pub has_subgroup_arithmetic: bool,
    pub has_storage_buffer_16bit: bool,
}

/// Metrics gathered from a compute dispatch execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputeMetrics {
    pub duration_ms: f64,
    pub workgroup_count: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub max_absolute_error: f32,
    pub verified_correct: bool,
}

/// Diagnostic outcome report.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticReport {
    pub device_info: GpuDeviceInfo,
    pub execution_passed: bool,
    pub compute_metrics: Option<ComputeMetrics>,
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_strings() {
        assert_eq!(ErrorCode::Success.as_str(), "Success");
        assert_eq!(ErrorCode::InvalidArgument.as_str(), "Invalid argument or null pointer");
        assert_eq!(ErrorCode::VulkanUnavailable.as_str(), "Vulkan runtime loader or driver unavailable");
    }

    #[test]
    fn test_error_formatting() {
        let err = Error::invalid_arg("null pointer received");
        assert_eq!(err.code, ErrorCode::InvalidArgument);
        let s = format!("{}", err);
        assert!(s.contains("InvalidArgument"));
        assert!(s.contains("null pointer received"));
    }

    #[test]
    fn test_gpu_device_info_defaults() {
        let info = GpuDeviceInfo::default();
        assert_eq!(info.vendor_id, 0);
        assert_eq!(info.subgroup_size, 32);
        assert!(!info.capabilities.has_fp16);
    }
}
