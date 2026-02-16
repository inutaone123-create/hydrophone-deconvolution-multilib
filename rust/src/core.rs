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
