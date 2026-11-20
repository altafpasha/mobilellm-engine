//! MobileLLM Diagnostics Subsystem
//!
//! Orchestrates end-to-end hardware capability reporting and numerical
//! validation of the GPU compute path.

use mobilellm_core::{ComputeMetrics, DiagnosticReport, Error, GpuDeviceInfo};
use mobilellm_gpu::GpuContext;

pub struct DiagnosticsRunner {
    gpu_context: GpuContext,
}

impl DiagnosticsRunner {
    pub fn new() -> Result<Self, Error> {
        let gpu_context = GpuContext::new()?;
        Ok(Self { gpu_context })
    }

    pub fn from_context(gpu_context: GpuContext) -> Self {
        Self { gpu_context }
    }

    /// Execute system diagnostics pass.
    pub fn execute(&self) -> Result<DiagnosticReport, Error> {
        let device_info = self.gpu_context.query_device_info()?;

        // Generate synthetic test vector representing layer normalization outputs
        let test_len = 1024;
        let mut input = Vec::with_capacity(test_len);
        for i in 0..test_len {
            input.push((i as f32) * 0.001 - 0.5);
        }
        let mut output = vec![0.0f32; test_len];

        let compute_metrics = self
            .gpu_context
            .run_deterministic_compute(&input, &mut output)?;

        let execution_passed = compute_metrics.verified_correct;

        let summary = format!(
            "Device: {} | Vendor: 0x{:04x} | Subgroup Size: {} | Duration: {:.3} ms | Max Error: {:.2e}",
            device_info.device_name,
            device_info.vendor_id,
            device_info.subgroup_size,
            compute_metrics.duration_ms,
            compute_metrics.max_absolute_error
        );

        Ok(DiagnosticReport {
            device_info,
            execution_passed,
            compute_metrics: Some(compute_metrics),
            summary,
        })
    }
}

/// Convenience function to execute an immediate diagnostic evaluation.
pub fn run_system_diagnostics() -> Result<DiagnosticReport, Error> {
    let runner = DiagnosticsRunner::new()?;
    runner.execute()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_execution() {
        let report = run_system_diagnostics().expect("Diagnostics run must succeed");
        assert!(report.execution_passed);
        assert!(report.compute_metrics.is_some());
        let metrics = report.compute_metrics.unwrap();
        assert!(metrics.max_absolute_error <= 1e-6);
        assert!(!report.summary.is_empty());
    }
}
