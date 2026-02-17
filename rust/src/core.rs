//! Hydrophone Deconvolution - Multi-language Implementation
//!
//! Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
//! DOI: 10.5281/zenodo.10079801
//! Original License: CC BY 4.0
//!
//! This implementation: 2024
//! License: CC BY 4.0

use num_complex::Complex64;
use rustfft::FftPlanner;

pub fn deconvolve_without_uncertainty(
    measured_signal: &[f64],
    frequency_response: &[Complex64],
    _sampling_rate: f64,
) -> Vec<f64> {
    let n = measured_signal.len();
    let mut signal: Vec<Complex64> = measured_signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut signal);

    let epsilon = Complex64::new(1e-12, 0.0);
    let mut deconvolved: Vec<Complex64> = signal
        .iter()
        .zip(frequency_response.iter())
        .map(|(s, f)| s / (f + epsilon))
        .collect();

    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut deconvolved);

    let scale = 1.0 / (n as f64);
    deconvolved.iter().map(|c| c.re * scale).collect()
}

pub fn deconvolve_with_uncertainty(
    measured_signal: &[f64],
    signal_uncertainty: &[f64],
    frequency_response: &[Complex64],
    response_uncertainty: &[f64],
    sampling_rate: f64,
    num_monte_carlo: usize,
) -> (Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    let n = measured_signal.len();
    let mut rng = rand::thread_rng();
    let mut mc_results = vec![vec![0.0f64; n]; num_monte_carlo];

    for i in 0..num_monte_carlo {
        let signal_pert: Vec<f64> = measured_signal
            .iter()
            .zip(signal_uncertainty.iter())
            .map(|(&s, &u)| s + u * rng.sample::<f64, _>(StandardNormal))
            .collect();

        let freq_resp_pert: Vec<Complex64> = frequency_response
            .iter()
            .zip(response_uncertainty.iter())
            .map(|(&f, &u)| {
                let noise = u * rng.sample::<f64, _>(StandardNormal);
                f + Complex64::new(noise, noise)
            })
            .collect();

        mc_results[i] = deconvolve_without_uncertainty(&signal_pert, &freq_resp_pert, sampling_rate);
    }

    let mut mean = vec![0.0f64; n];
    let mut std = vec![0.0f64; n];
    for j in 0..n {
        let sum: f64 = mc_results.iter().map(|r| r[j]).sum();
        mean[j] = sum / num_monte_carlo as f64;
        let var: f64 = mc_results.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>()
            / num_monte_carlo as f64;
        std[j] = var.sqrt();
    }

    (mean, std)
}

#[derive(Debug, Clone)]
pub struct PulseParameters {
    pub pc_value: f64,
    pub pc_uncertainty: f64,
    pub pc_index: usize,
    pub pc_time: f64,
    pub pr_value: f64,
    pub pr_uncertainty: f64,
    pub pr_index: usize,
    pub pr_time: f64,
    pub ppsi_value: f64,
    pub ppsi_uncertainty: f64,
}

pub fn pulse_parameters(
    time: &[f64],
    pressure: &[f64],
    u_pressure: PulseUncertainty,
) -> PulseParameters {
    let n = pressure.len();

    // Build covariance matrix (stored as flat n×n)
    let u_p: Vec<f64> = match u_pressure {
        PulseUncertainty::Scalar(u) => {
            let mut m = vec![0.0; n * n];
            let u_sq = u * u;
            for i in 0..n {
                m[i * n + i] = u_sq;
            }
            m
        }
        PulseUncertainty::Vector(ref v) => {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                m[i * n + i] = v[i] * v[i];
            }
            m
        }
        PulseUncertainty::Matrix(ref m) => m.clone(),
    };

    let dt = (time[n - 1] - time[0]) / (n - 1) as f64;

    // pc: compressional peak
    let mut pc_index = 0usize;
    let mut pc_value = pressure[0];
    for i in 1..n {
        if pressure[i] > pc_value {
            pc_value = pressure[i];
            pc_index = i;
        }
    }
    let pc_uncertainty = u_p[pc_index * n + pc_index].sqrt();
    let pc_time = time[pc_index];

    // pr: rarefactional peak
    let mut pr_index = 0usize;
    let mut pr_min = pressure[0];
    for i in 1..n {
        if pressure[i] < pr_min {
            pr_min = pressure[i];
            pr_index = i;
        }
    }
    let pr_value = -pr_min;
    let pr_uncertainty = u_p[pr_index * n + pr_index].sqrt();
    let pr_time = time[pr_index];

    // ppsi
    let ppsi_value: f64 = pressure.iter().map(|p| p * p).sum::<f64>() * dt;
    let c: Vec<f64> = pressure.iter().map(|p| 2.0 * p.abs() * dt).collect();
    let mut ppsi_var = 0.0;
    for i in 0..n {
        for j in 0..n {
            ppsi_var += c[i] * u_p[i * n + j] * c[j];
        }
    }
    let ppsi_uncertainty = ppsi_var.sqrt();

    PulseParameters {
        pc_value, pc_uncertainty, pc_index, pc_time,
        pr_value, pr_uncertainty, pr_index, pr_time,
        ppsi_value, ppsi_uncertainty,
    }
}

pub enum PulseUncertainty {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<f64>),
}
