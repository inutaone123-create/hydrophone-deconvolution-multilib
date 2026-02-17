//! Hydrophone Deconvolution - Multi-language Implementation
//!
//! Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
//! DOI: 10.5281/zenodo.10079801
//! License: CC BY 4.0

use hydrophone_deconvolution::{deconvolve_without_uncertainty, pulse_parameters, PulseUncertainty};
use num_complex::Complex64;
use std::fs;
use std::io::Write;

fn load_csv(path: &str) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect()
}

fn main() {
    let data_dir = "/workspace/test-data";
    let results_dir = "/workspace/validation/results";
    fs::create_dir_all(results_dir).unwrap();

    let measured = load_csv(&format!("{}/measured_signal.csv", data_dir));
    let freq_real = load_csv(&format!("{}/freq_response_real.csv", data_dir));
    let freq_imag = load_csv(&format!("{}/freq_response_imag.csv", data_dir));

    let freq_response: Vec<Complex64> = freq_real
        .iter()
        .zip(freq_imag.iter())
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect();

    let result = deconvolve_without_uncertainty(&measured, &freq_response, 1e7);

    let mut file = fs::File::create(format!("{}/rust_result.csv", results_dir)).unwrap();
    for val in &result {
        writeln!(file, "{:.18e}", val).unwrap();
    }

    println!("Rust result exported: {} samples", result.len());

    // Pulse parameters
    let sig_unc = load_csv(&format!("{}/signal_uncertainty.csv", data_dir));
    let n = result.len();
    let sampling_rate = 1e7;
    let time_vec: Vec<f64> = (0..n).map(|i| i as f64 / sampling_rate).collect();

    let pp = pulse_parameters(&time_vec, &result, PulseUncertainty::Vector(sig_unc));

    let mut pp_file = fs::File::create(format!("{}/rust_pulse_params.csv", results_dir)).unwrap();
    writeln!(pp_file, "{:.18e},{:.18e},{:.18e},{:.18e},{:.18e},{:.18e}",
        pp.pc_value, pp.pc_uncertainty, pp.pr_value, pp.pr_uncertainty,
        pp.ppsi_value, pp.ppsi_uncertainty).unwrap();

    println!("Rust pulse parameters exported");
}
