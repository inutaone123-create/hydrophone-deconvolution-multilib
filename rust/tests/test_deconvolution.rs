//! Hydrophone Deconvolution - Multi-language Implementation
//!
//! Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
//! DOI: 10.5281/zenodo.10079801
//! License: CC BY 4.0

use hydrophone_deconvolution::deconvolve_without_uncertainty;
use num_complex::Complex64;
use rustfft::FftPlanner;

#[test]
fn test_output_length() {
    let n = 1024;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let freq_resp: Vec<Complex64> = (0..n).map(|_| Complex64::new(1.0, 0.0)).collect();
    let result = deconvolve_without_uncertainty(&signal, &freq_resp, 1e7);
    assert_eq!(result.len(), n);
}

#[test]
fn test_known_signal_recovery() {
    let n = 256;
    let original: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();

    // Create frequency response via FFT (well-conditioned) + offset
    let impulse: Vec<f64> = (0..n).map(|i| if i == 0 { 2.0 } else { 0.1 * (i as f64 * 0.01).sin() }).collect();
    let mut freq_resp: Vec<Complex64> = impulse.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut freq_resp);

    // Convolve: FFT(original) * freq_resp, IFFT, scale
    let mut original_fft: Vec<Complex64> = original.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut original_fft);

    let mut convolved: Vec<Complex64> = original_fft.iter().zip(freq_resp.iter())
        .map(|(s, f)| s * f).collect();

    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut convolved);

    let scale = 1.0 / n as f64;
    let measured: Vec<f64> = convolved.iter().map(|c| c.re * scale).collect();

    let recovered = deconvolve_without_uncertainty(&measured, &freq_resp, 1e7);

    let max_err = recovered.iter().zip(original.iter())
        .map(|(r, o)| (r - o).abs())
        .fold(0.0f64, f64::max);

    assert!(max_err < 1e-10, "Max error: {}", max_err);
}
