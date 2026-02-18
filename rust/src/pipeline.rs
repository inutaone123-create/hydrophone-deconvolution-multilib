// Hydrophone Deconvolution - Multi-language Implementation
//
// Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
// DOI: 10.5281/zenodo.10079801
// Original License: CC BY 4.0
//
// This implementation: 2024
// License: CC BY 4.0

//! GUM-based deconvolution pipeline with full uncertainty propagation.
//!
//! Implements GUM_DFT, GUM_iDFT, DFT_deconv, DFT_multiply, Bode equation,
//! regularization filters, and the complete processing pipeline.
//!
//! All spectra use PyDynamic format: [Re₀..Re_{M-1}, Im₀..Im_{M-1}]
//! Covariance matrices are stored as flat Vec<f64> with row-major layout.

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::fs;
use std::io::{self, BufRead};

use crate::core::{pulse_parameters, PulseUncertainty, PulseParameters};

// ── Matrix helpers (flat row-major) ──

#[inline]
fn mat_set(m: &mut [f64], _rows: usize, cols: usize, r: usize, c: usize, v: f64) {
    m[r * cols + c] = v;
}

// ── Frequency scale ──

pub fn calcfreqscale(time: &[f64]) -> Vec<f64> {
    let n = time.len();
    let m = n / 2 + 1;
    let t_min = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fmax = (n - 1) as f64 / ((t_max - t_min) * n as f64 / (n - 1) as f64);
    let mut freq = vec![0.0; 2 * m];
    for i in 0..m {
        let fi = fmax * i as f64 / (n - 1) as f64;
        freq[i] = fi;
        freq[i + m] = fi;
    }
    freq
}

// ── DFT Result ──

pub struct DftResult {
    pub f: Vec<f64>,
    pub uf: Vec<f64>, // flat [2M × 2M] row-major
}

// ── GUM DFT ──

pub fn gum_dft(x: &[f64], ux: &[f64]) -> DftResult {
    let n = x.len();
    let m = n / 2 + 1;

    // FFT for best estimate
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut signal: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft.process(&mut signal);

    let mut f = vec![0.0; 2 * m];
    for k in 0..m {
        f[k] = signal[k].re;
        f[k + m] = signal[k].im;
    }

    // Check uniform variance
    let ux0 = ux[0];
    let uniform = ux.iter().all(|&v| (v - ux0).abs() <= 1e-30 * ux0.abs().max(1e-300));

    let dim = 2 * m;
    let mut uf = vec![0.0; dim * dim];

    if uniform {
        // Diagonal UF for uniform variance
        for k in 0..m {
            let mut sum_cc = 0.0;
            let mut sum_ss = 0.0;
            for n_idx in 0..n {
                let phase = 2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                let c = phase.cos();
                let s = -phase.sin();
                sum_cc += c * c;
                sum_ss += s * s;
            }
            mat_set(&mut uf, dim, dim, k, k, ux0 * sum_cc);
            mat_set(&mut uf, dim, dim, k + m, k + m, ux0 * sum_ss);
        }
    } else {
        // General case: precompute cos/sin, compute UF = C*diag(Ux)*C'
        let mut cos_vals = vec![0.0; m * n];
        let mut sin_vals = vec![0.0; m * n];
        for k in 0..m {
            for n_idx in 0..n {
                let phase = 2.0 * PI * k as f64 * n_idx as f64 / n as f64;
                cos_vals[k * n + n_idx] = phase.cos();
                sin_vals[k * n + n_idx] = -phase.sin();
            }
        }

        for k1 in 0..m {
            for k2 in k1..m {
                let mut cc = 0.0;
                let mut cs = 0.0;
                let mut ss = 0.0;
                for n_idx in 0..n {
                    let c1 = cos_vals[k1 * n + n_idx];
                    let s1 = sin_vals[k1 * n + n_idx];
                    let c2 = cos_vals[k2 * n + n_idx];
                    let s2 = sin_vals[k2 * n + n_idx];
                    let u = ux[n_idx];
                    cc += c1 * u * c2;
                    cs += c1 * u * s2;
                    ss += s1 * u * s2;
                }
                mat_set(&mut uf, dim, dim, k1, k2, cc);
                mat_set(&mut uf, dim, dim, k2, k1, cc);
                mat_set(&mut uf, dim, dim, k1, k2 + m, cs);
                mat_set(&mut uf, dim, dim, k2 + m, k1, cs);
                mat_set(&mut uf, dim, dim, k1 + m, k2 + m, ss);
                mat_set(&mut uf, dim, dim, k2 + m, k1 + m, ss);

                // SC block
                if k1 != k2 {
                    let mut sc = 0.0;
                    for n_idx in 0..n {
                        sc += sin_vals[k1 * n + n_idx] * ux[n_idx] * cos_vals[k2 * n + n_idx];
                    }
                    mat_set(&mut uf, dim, dim, k1 + m, k2, sc);
                    mat_set(&mut uf, dim, dim, k2, k1 + m, sc);
                } else {
                    mat_set(&mut uf, dim, dim, k1 + m, k2, cs);
                    mat_set(&mut uf, dim, dim, k2, k1 + m, cs);
                }
            }
            // SC for k2 < k1
            for k2 in 0..k1 {
                let mut sc = 0.0;
                for n_idx in 0..n {
                    sc += sin_vals[k1 * n + n_idx] * ux[n_idx] * cos_vals[k2 * n + n_idx];
                }
                mat_set(&mut uf, dim, dim, k1 + m, k2, sc);
                mat_set(&mut uf, dim, dim, k2, k1 + m, sc);
            }
        }
    }

    DftResult { f, uf }
}

// ── GUM iDFT ──

pub fn gum_idft(f_spec: &[f64], uf: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let total = f_spec.len();
    let m = total / 2;
    let n = 2 * (m - 1);
    let dim = 2 * m;

    // iFFT for best estimate
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut spec = vec![Complex64::new(0.0, 0.0); n];
    spec[0] = Complex64::new(f_spec[0], f_spec[m]);
    for k in 1..m {
        spec[k] = Complex64::new(f_spec[k], f_spec[k + m]);
        spec[n - k] = Complex64::new(f_spec[k], -f_spec[k + m]);
    }
    ifft.process(&mut spec);

    let scale = 1.0 / n as f64;
    let x: Vec<f64> = spec.iter().map(|c| c.re * scale).collect();

    // Multipliers for rfft convention
    let mut mult = vec![2.0; m];
    mult[0] = 1.0;
    if n % 2 == 0 { mult[m - 1] = 1.0; }

    // Check if UF is diagonal
    let is_diag = {
        let mut diag = true;
        'outer: for i in 0..dim {
            for j in 0..dim {
                if i != j && uf[i * dim + j].abs() > 1e-30 * uf[i * dim + i].abs().max(1e-300) {
                    diag = false;
                    break 'outer;
                }
            }
        }
        diag
    };

    let inv_n2 = 1.0 / (n as f64 * n as f64);
    let mut u_x = vec![0.0; n * n];

    if is_diag {
        for n1 in 0..n {
            for n2 in n1..n {
                let mut sum = 0.0;
                for k in 0..m {
                    let p1 = 2.0 * PI * n1 as f64 * k as f64 / n as f64;
                    let p2 = 2.0 * PI * n2 as f64 * k as f64 / n as f64;
                    let c1 = p1.cos() * mult[k];
                    let s1 = -p1.sin() * mult[k];
                    let c2 = p2.cos() * mult[k];
                    let s2 = -p2.sin() * mult[k];
                    sum += c1 * uf[k * dim + k] * c2 + s1 * uf[(k + m) * dim + (k + m)] * s2;
                }
                let val = sum * inv_n2;
                u_x[n1 * n + n2] = val;
                u_x[n2 * n + n1] = val;
            }
        }
    } else {
        // General path: Ux = C*UF*C' / N²
        // C = [Cc | Cs] where Cc[n,k]=cos*mult, Cs[n,k]=-sin*mult
        // Compute Ac = Cc*RR + Cs*IR, As = Cc*RI + Cs*II
        // Then Ux = Ac*Cc' + As*Cs'

        let mut ac = vec![0.0; n * m]; // [N × M]
        let mut a_s = vec![0.0; n * m]; // [N × M]

        for n1 in 0..n {
            for k2 in 0..m {
                let mut sum_c = 0.0;
                let mut sum_s = 0.0;
                for k1 in 0..m {
                    let phase = 2.0 * PI * n1 as f64 * k1 as f64 / n as f64;
                    let cc = phase.cos() * mult[k1];
                    let cs = -phase.sin() * mult[k1];
                    sum_c += cc * uf[k1 * dim + k2] + cs * uf[(k1 + m) * dim + k2];
                    sum_s += cc * uf[k1 * dim + (k2 + m)] + cs * uf[(k1 + m) * dim + (k2 + m)];
                }
                ac[n1 * m + k2] = sum_c;
                a_s[n1 * m + k2] = sum_s;
            }
        }

        for n1 in 0..n {
            for n2 in n1..n {
                let mut sum = 0.0;
                for k in 0..m {
                    let phase = 2.0 * PI * n2 as f64 * k as f64 / n as f64;
                    let cc2 = phase.cos() * mult[k];
                    let cs2 = -phase.sin() * mult[k];
                    sum += ac[n1 * m + k] * cc2 + a_s[n1 * m + k] * cs2;
                }
                let val = sum * inv_n2;
                u_x[n1 * n + n2] = val;
                u_x[n2 * n + n1] = val;
            }
        }
    }

    (x, u_x)
}

// ── DFT deconvolution ──

pub fn dft_deconv(h: &[f64], y: &[f64], uh: &[f64], uy: &[f64]) -> DftResult {
    let m = h.len() / 2;
    let dim = 2 * m;
    let mut x_out = vec![0.0; dim];
    let mut ux = vec![0.0; dim * dim];

    for k in 0..m {
        let hr = h[k];
        let hi = h[k + m];
        let yr = y[k];
        let yi = y[k + m];
        let d = hr * hr + hi * hi;
        let d = if d == 0.0 { 1e-30 } else { d };

        let xr = (yr * hr + yi * hi) / d;
        let xi = (yi * hr - yr * hi) / d;
        x_out[k] = xr;
        x_out[k + m] = xi;

        // Jacobians
        let jy00 = hr / d;
        let jy01 = hi / d;
        let jy10 = -hi / d;
        let jy11 = hr / d;
        let jh00 = (yr - 2.0 * hr * xr) / d;
        let jh01 = (yi - 2.0 * hi * xr) / d;
        let jh10 = (yi - 2.0 * hr * xi) / d;
        let jh11 = (-yr - 2.0 * hi * xi) / d;

        let uy00 = uy[k * dim + k];
        let uy01 = uy[k * dim + (k + m)];
        let uy10 = uy[(k + m) * dim + k];
        let uy11 = uy[(k + m) * dim + (k + m)];
        let uh00 = uh[k * dim + k];
        let uh01 = uh[k * dim + (k + m)];
        let uh10 = uh[(k + m) * dim + k];
        let uh11 = uh[(k + m) * dim + (k + m)];

        // UX_k = JY*UY_k*JY' + JH*UH_k*JH'
        // tmp = U*J' where J'[b,c] = J[c,b]
        let ty00 = uy00 * jy00 + uy01 * jy01;
        let ty01 = uy00 * jy10 + uy01 * jy11;
        let ty10 = uy10 * jy00 + uy11 * jy01;
        let ty11 = uy10 * jy10 + uy11 * jy11;
        let th00 = uh00 * jh00 + uh01 * jh01;
        let th01 = uh00 * jh10 + uh01 * jh11;
        let th10 = uh10 * jh00 + uh11 * jh01;
        let th11 = uh10 * jh10 + uh11 * jh11;

        let ux00 = jy00 * ty00 + jy01 * ty10 + jh00 * th00 + jh01 * th10;
        let ux01 = jy00 * ty01 + jy01 * ty11 + jh00 * th01 + jh01 * th11;
        let ux10 = jy10 * ty00 + jy11 * ty10 + jh10 * th00 + jh11 * th10;
        let ux11 = jy10 * ty01 + jy11 * ty11 + jh10 * th01 + jh11 * th11;

        mat_set(&mut ux, dim, dim, k, k, ux00);
        mat_set(&mut ux, dim, dim, k, k + m, ux01);
        mat_set(&mut ux, dim, dim, k + m, k, ux10);
        mat_set(&mut ux, dim, dim, k + m, k + m, ux11);
    }

    DftResult { f: x_out, uf: ux }
}

// ── DFT multiply ──

pub fn dft_multiply(y: &[f64], filt: &[f64], uy: &[f64]) -> DftResult {
    let m = y.len() / 2;
    let dim = 2 * m;
    let mut z = vec![0.0; dim];
    let mut uz = vec![0.0; dim * dim];

    for k in 0..m {
        let yr = y[k];
        let yi = y[k + m];
        let fr = filt[k];
        let fi = filt[k + m];
        z[k] = yr * fr - yi * fi;
        z[k + m] = yr * fi + yi * fr;

        // JY = [[Fr, -Fi],[Fi, Fr]], UZ = JY * UY * JY'
        let uy00 = uy[k * dim + k];
        let uy01 = uy[k * dim + (k + m)];
        let uy10 = uy[(k + m) * dim + k];
        let uy11 = uy[(k + m) * dim + (k + m)];

        // tmp = UY * JY'
        let t00 = uy00 * fr - uy01 * fi;
        let t01 = uy00 * fi + uy01 * fr;
        let t10 = uy10 * fr - uy11 * fi;
        let t11 = uy10 * fi + uy11 * fr;

        let uz00 = fr * t00 + (-fi) * t10;
        let uz01 = fr * t01 + (-fi) * t11;
        let uz10 = fi * t00 + fr * t10;
        let uz11 = fi * t01 + fr * t11;

        mat_set(&mut uz, dim, dim, k, k, uz00);
        mat_set(&mut uz, dim, dim, k, k + m, uz01);
        mat_set(&mut uz, dim, dim, k + m, k, uz10);
        mat_set(&mut uz, dim, dim, k + m, k + m, uz11);
    }

    DftResult { f: z, uf: uz }
}

// ── Bode equation ──

pub fn bode_equation(freq: &[f64], amp: &[f64], varamp: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let df = freq[1] - freq[0];
    let mut phase = vec![0.0; n];
    let mut varphase = vec![0.0; n];

    let logamp: Vec<f64> = amp.iter().map(|a| a.ln()).collect();

    for i in 0..n {
        let coeff = 2.0 * freq[i] / PI * df;
        let mut sum_p = 0.0;
        let mut sum_v = 0.0;
        for j in 0..n {
            let denom = if j == i { 1.0 } else { freq[j] * freq[j] - freq[i] * freq[i] };
            sum_p += (logamp[j] - logamp[i]) / denom;
            let denomu = amp[j] * denom;
            let numu = if j == i { 0.0 } else { 1.0 };
            sum_v += (numu / denomu) * (numu / denomu) * varamp[j];
        }
        phase[i] = coeff * sum_p;
        varphase[i] = coeff * coeff * sum_v;
    }

    (phase, varphase)
}

// ── Amp+Phase to DFT ──

pub fn amp_phase_to_dft(amp: &[f64], phase: &[f64], uap: &[f64]) -> DftResult {
    let m = amp.len();
    let dim = 2 * m;
    let mut x = vec![0.0; dim];
    let mut ux = vec![0.0; dim * dim];

    for k in 0..m {
        let cp = phase[k].cos();
        let sp = phase[k].sin();
        x[k] = amp[k] * cp;
        x[k + m] = amp[k] * sp;

        let va = uap[k];
        let vp = uap[k + m];
        mat_set(&mut ux, dim, dim, k, k, cp * cp * va + amp[k] * amp[k] * sp * sp * vp);
        let off = cp * sp * va - amp[k] * amp[k] * sp * cp * vp;
        mat_set(&mut ux, dim, dim, k, k + m, off);
        mat_set(&mut ux, dim, dim, k + m, k, off);
        mat_set(&mut ux, dim, dim, k + m, k + m, sp * sp * va + amp[k] * amp[k] * cp * cp * vp);
    }

    DftResult { f: x, uf: ux }
}

// ── Regularization filter ──

pub fn regularization_filter(freq: &[f64], fc: f64, filter_type: &str) -> Vec<f64> {
    let m = freq.len();
    let mut hc: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m];

    for i in 0..m {
        let jf = Complex64::new(0.0, freq[i] / fc);
        hc[i] = match filter_type {
            "LowPass" => {
                let s = Complex64::new(1.0, 0.0) + Complex64::new(0.0, freq[i] / (fc * 1.555));
                Complex64::new(1.0, 0.0) / (s * s)
            }
            "CriticalDamping" => {
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + 1.28719 * jf + 0.41421 * jf * jf)
            }
            "Bessel" => {
                Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + 1.3617 * jf
                    - Complex64::new(0.6180 * freq[i] * freq[i] / (fc * fc), 0.0))
            }
            "None" => Complex64::new(1.0, 0.0),
            _ => panic!("Unknown filter type: {}", filter_type),
        };
    }

    if filter_type != "None" {
        // Phase correction
        let sqrt_half = (0.5_f64).sqrt();
        let mut ind_3db = 0;
        let mut min_diff = f64::MAX;
        for i in 0..m {
            let diff = (hc[i].norm() - sqrt_half).abs();
            if diff < min_diff {
                min_diff = diff;
                ind_3db = i;
            }
        }
        if ind_3db > 1 {
            let mut sum_f_ang = 0.0;
            let mut sum_ff = 0.0;
            for i in 0..ind_3db {
                sum_f_ang += freq[i] * hc[i].arg();
                sum_ff += freq[i] * freq[i];
            }
            let w = sum_f_ang / sum_ff;
            for i in 0..m {
                hc[i] *= Complex64::new(0.0, -w * freq[i]).exp();
            }
        }
    }

    let mut filt = vec![0.0; 2 * m];
    for i in 0..m {
        filt[i] = hc[i].re;
        filt[i + m] = hc[i].im;
    }
    filt
}

// ── I/O ──

pub struct SignalData {
    pub time: Vec<f64>,
    pub voltage: Vec<f64>,
    pub dt: f64,
}

pub struct CalibrationData {
    pub frequency: Vec<f64>,
    pub real_part: Vec<f64>,
    pub imag_part: Vec<f64>,
    pub var_real: Vec<f64>,
    pub var_imag: Vec<f64>,
    pub kovar: Vec<f64>,
}

pub fn read_dat_signal(filepath: &str) -> io::Result<SignalData> {
    let content = fs::read_to_string(filepath)?;
    let mut values: Vec<f64> = Vec::new();
    for line in content.lines() {
        if let Ok(v) = line.trim().parse::<f64>() {
            values.push(v);
        }
    }

    let n_samples = values[0] as usize;
    let dt = values[1];
    let mut voltage: Vec<f64> = values[4..4 + n_samples].to_vec();
    let mean: f64 = voltage.iter().sum::<f64>() / n_samples as f64;
    for v in voltage.iter_mut() {
        *v -= mean;
    }

    let time: Vec<f64> = (0..n_samples).map(|i| i as f64 * dt).collect();
    Ok(SignalData { time, voltage, dt })
}

pub fn read_calibration_csv(filepath: &str) -> io::Result<CalibrationData> {
    let file = fs::File::open(filepath)?;
    let reader = io::BufReader::new(file);
    let mut freq = Vec::new();
    let mut real_p = Vec::new();
    let mut imag_p = Vec::new();
    let mut var_r = Vec::new();
    let mut var_i = Vec::new();
    let mut kov = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 { continue; }
        freq.push(parts[0].trim().parse::<f64>().unwrap() * 1e6);
        real_p.push(parts[1].trim().parse::<f64>().unwrap());
        imag_p.push(parts[2].trim().parse::<f64>().unwrap());
        var_r.push(parts[3].trim().parse::<f64>().unwrap());
        var_i.push(parts[4].trim().parse::<f64>().unwrap());
        kov.push(parts[5].trim().parse::<f64>().unwrap());
    }

    Ok(CalibrationData {
        frequency: freq,
        real_part: real_p,
        imag_part: imag_p,
        var_real: var_r,
        var_imag: var_i,
        kovar: kov,
    })
}

fn interp1(xp: &[f64], fp: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            if xi <= xp[0] { return fp[0]; }
            if xi >= *xp.last().unwrap() { return *fp.last().unwrap(); }
            let idx = xp.partition_point(|&v| v < xi);
            if idx == 0 { return fp[0]; }
            let t = (xi - xp[idx - 1]) / (xp[idx] - xp[idx - 1]);
            fp[idx - 1] + t * (fp[idx] - fp[idx - 1])
        })
        .collect()
}

fn find_nearest(arr: &[f64], val: f64) -> usize {
    arr.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((**a - val).abs()).partial_cmp(&((**b - val).abs())).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// ── Pipeline result ──

pub struct PipelineResult {
    pub time: Vec<f64>,
    pub scaled: Vec<f64>,
    pub deconvolved: Vec<f64>,
    pub regularized: Vec<f64>,
    pub uncertainty: Vec<f64>,
    pub pulse_params: PulseParameters,
    pub n_samples: usize,
}

// ── Full pipeline ──

pub fn full_pipeline(
    measurement_file: &str,
    noise_file: &str,
    cal_file: &str,
    usebode: bool,
    filter_type: &str,
    fc: f64,
) -> io::Result<PipelineResult> {
    let fmin = 1e6;
    let fmax_cal = 60e6;

    let sig = read_dat_signal(measurement_file)?;
    let n = sig.voltage.len();

    // Noise uncertainty
    let _noise_raw = read_dat_signal(noise_file)?;
    // Re-read raw noise to compute std (before DC removal)
    let noise_content = fs::read_to_string(noise_file)?;
    let mut nvals: Vec<f64> = Vec::new();
    for line in noise_content.lines() {
        if let Ok(v) = line.trim().parse::<f64>() {
            nvals.push(v);
        }
    }
    let n_noise = nvals[0] as usize;
    let n_mean: f64 = nvals[4..4 + n_noise].iter().sum::<f64>() / n_noise as f64;
    let n_var: f64 = nvals[4..4 + n_noise].iter().map(|v| (v - n_mean).powi(2)).sum::<f64>();
    let stdev = (n_var / n_noise as f64).sqrt();

    let ux_vec: Vec<f64> = vec![stdev * stdev; n];

    let frequency = calcfreqscale(&sig.time);
    let m = n / 2 + 1;
    let half_freq: Vec<f64> = frequency[..m].to_vec();

    // GUM_DFT
    let mut dft_result = gum_dft(&sig.voltage, &ux_vec);
    let dim = 2 * m;
    for v in dft_result.f.iter_mut() { *v *= 2.0 / n as f64; }
    let scale_uf = 4.0 / (n as f64 * n as f64);
    for v in dft_result.uf.iter_mut() { *v *= scale_uf; }

    // Calibration
    let cal = read_calibration_csv(cal_file)?;
    let imin = find_nearest(&cal.frequency, fmin);
    let imax = find_nearest(&cal.frequency, fmax_cal);
    let (h, uh) = if usebode {
        let mut amp = vec![0.0; cal.frequency.len()];
        let mut varamp = vec![0.0; cal.frequency.len()];
        for i in 0..amp.len() {
            amp[i] = (cal.real_part[i] * cal.real_part[i] + cal.imag_part[i] * cal.imag_part[i]).sqrt();
            varamp[i] = (cal.real_part[i] / amp[i]).powi(2) * cal.var_real[i]
                + (cal.imag_part[i] / amp[i]).powi(2) * cal.var_imag[i]
                + 2.0 * (cal.real_part[i] / amp[i]) * (cal.imag_part[i] / amp[i]) * cal.kovar[i];
        }

        let freq_trim: Vec<f64> = cal.frequency[imin..=imax].to_vec();
        let amp_trim: Vec<f64> = amp[imin..=imax].to_vec();
        let varamp_trim: Vec<f64> = varamp[imin..=imax].to_vec();

        let ampip = interp1(&freq_trim, &amp_trim, &half_freq);
        let varampip = interp1(&freq_trim, &varamp_trim, &half_freq);

        let (phaseip, varphaseip) = bode_equation(&half_freq, &ampip, &varampip);

        let mut uap = vec![0.0; 2 * m];
        uap[..m].copy_from_slice(&varampip);
        uap[m..].copy_from_slice(&varphaseip);

        let ap_result = amp_phase_to_dft(&ampip, &phaseip, &uap);
        (ap_result.f, ap_result.uf)
    } else {
        let freq_trim: Vec<f64> = cal.frequency[imin..=imax].to_vec();
        let real_trim: Vec<f64> = cal.real_part[imin..=imax].to_vec();
        let imag_trim: Vec<f64> = cal.imag_part[imin..=imax].to_vec();
        let vr_trim: Vec<f64> = cal.var_real[imin..=imax].to_vec();
        let vi_trim: Vec<f64> = cal.var_imag[imin..=imax].to_vec();
        let kov_trim: Vec<f64> = cal.kovar[imin..=imax].to_vec();

        let real_ip = interp1(&freq_trim, &real_trim, &half_freq);
        let imag_ip = interp1(&freq_trim, &imag_trim, &half_freq);
        let vr_ip = interp1(&freq_trim, &vr_trim, &half_freq);
        let vi_ip = interp1(&freq_trim, &vi_trim, &half_freq);
        let kov_ip = interp1(&freq_trim, &kov_trim, &half_freq);

        let mut x = vec![0.0; dim];
        x[..m].copy_from_slice(&real_ip);
        x[m..].copy_from_slice(&imag_ip);

        let mut ux = vec![0.0; dim * dim];
        for i in 0..m {
            mat_set(&mut ux, dim, dim, i, i, vr_ip[i]);
            mat_set(&mut ux, dim, dim, i, i + m, kov_ip[i]);
            mat_set(&mut ux, dim, dim, i + m, i, kov_ip[i]);
            mat_set(&mut ux, dim, dim, i + m, i + m, vi_ip[i]);
        }
        (x, ux)
    };

    // DFT_deconv
    let deconv_result = dft_deconv(&h, &dft_result.f, &uh, &dft_result.uf);

    // Filter
    let filt = regularization_filter(&half_freq, fc, filter_type);

    // DFT_multiply
    let regul_result = dft_multiply(&deconv_result.f, &filt, &deconv_result.uf);

    // GUM_iDFT
    let (sigp_raw, usigp_raw) = gum_idft(&regul_result.f, &regul_result.uf);
    let n_sig = sigp_raw.len();
    let sigp: Vec<f64> = sigp_raw.iter().map(|v| v * n_sig as f64 / 2.0).collect();
    let mut usigp = vec![0.0; n_sig * n_sig];
    let scale_u = (n_sig as f64 * n_sig as f64) / 4.0;
    for i in 0..usigp_raw.len() { usigp[i] = usigp_raw[i] * scale_u; }

    // Unfiltered deconvolution
    let (deconv_raw, _) = gum_idft(&deconv_result.f, &deconv_result.uf);
    let deconv_time: Vec<f64> = deconv_raw.iter().map(|v| v * deconv_raw.len() as f64 / 2.0).collect();

    // Scaled signal
    let mut hyd_amp = vec![0.0; m];
    let mut sig_amp = vec![0.0; m];
    for i in 0..m {
        hyd_amp[i] = (h[i] * h[i] + h[i + m] * h[i + m]).sqrt();
        sig_amp[i] = (dft_result.f[i] * dft_result.f[i] + dft_result.f[i + m] * dft_result.f[i + m]).sqrt();
    }
    let iffun = sig_amp.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let hydempffun = hyd_amp[iffun];
    let scaled: Vec<f64> = sig.voltage.iter().map(|v| v / hydempffun).collect();

    // Uncertainty
    let delta_sigp: Vec<f64> = (0..n_sig).map(|i| usigp[i * n_sig + i].sqrt()).collect();

    // Pulse parameters
    let pp = pulse_parameters(&sig.time, &sigp, PulseUncertainty::Matrix(usigp));

    Ok(PipelineResult {
        time: sig.time,
        scaled,
        deconvolved: deconv_time,
        regularized: sigp,
        uncertainty: delta_sigp,
        pulse_params: pp,
        n_samples: n,
    })
}
