#!/usr/bin/env python3
"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

This implementation: 2024
License: CC BY 4.0
https://creativecommons.org/licenses/by/4.0/

Generate reference results using PyDynamic for all 128 patterns
(16 data patterns × 4 filters × 2 Bode options).
"""

import os
import sys
import numpy as np
import PyDynamic
from PyDynamic.uncertainty.interpolate import interp1d_unc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "original-data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ORIGINAL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "original-results")

# ── 16 data patterns ──
PATTERNS = [
    # i, measurement_file, noise_file, cal_file, hydrophone_name, measurement_type, usebode_default
    (1, "MeasuredSignals/M-Mode 3 MHz/M3_MH44.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "M-Mode3MHz", True),
    (2, "MeasuredSignals/M-Mode 3 MHz/M3_MH46.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "M-Mode3MHz", True),
    (3, "MeasuredSignals/M-Mode 3 MHz/M3_ON1704.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "M-Mode3MHz", False),
    (4, "MeasuredSignals/M-Mode 3 MHz/M3_PA1434.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "M-Mode3MHz", True),
    (5, "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "Pulse-Doppler-Mode3MHz", True),
    (6, "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "Pulse-Doppler-Mode3MHz", True),
    (7, "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "Pulse-Doppler-Mode3MHz", False),
    (8, "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "Pulse-Doppler-Mode3MHz", True),
    (9, "MeasuredSignals/M-Mode 6 MHz/M6_MH44.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "M-Mode6MHz", True),
    (10, "MeasuredSignals/M-Mode 6 MHz/M6_MH46.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "M-Mode6MHz", True),
    (11, "MeasuredSignals/M-Mode 6 MHz/M6_ON1704.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "M-Mode6MHz", False),
    (12, "MeasuredSignals/M-Mode 6 MHz/M6_PA1434.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "M-Mode6MHz", True),
    (13, "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "Pulse-Doppler-Mode7MHz", True),
    (14, "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "Pulse-Doppler-Mode7MHz", True),
    (15, "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "Pulse-Doppler-Mode7MHz", False),
    (16, "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "Pulse-Doppler-Mode7MHz", True),
]

# Cutoff frequencies per measurement type
FC_MAP = {
    "M-Mode3MHz": 100e6,
    "Pulse-Doppler-Mode3MHz": 120e6,
    "M-Mode6MHz": 150e6,
    "Pulse-Doppler-Mode7MHz": 200e6,
}

FILTER_TYPES = ["LowPass", "CriticalDamping", "Bessel", "None"]
BODE_OPTIONS = [True, False]

FMIN = 1e6
FMAX = 60e6


def calcfreqscale(timeseries, sign2=1):
    """Calculate frequency scale matching PyDynamic's GUM_DFT output."""
    n = np.size(timeseries)
    sign = -1 if sign2 < 0 else 1
    fmax = 1 / ((np.max(timeseries) - np.min(timeseries)) * n / (n - 1)) * (n - 1)
    f = np.linspace(0, fmax, n)
    if sign2 == 0:
        return f[0:int(n / 2.0 + 1)]
    return np.hstack((f[0:int(n / 2.0 + 1)], sign * f[0:int(n / 2 + 1)]))


def findnearestmatch(liste, value):
    return np.argmin(abs(liste - value))


def read_dat_signal(filepath):
    """Read .DAT signal file: header has n_samples, dt, skip2 lines, then voltage."""
    rawdata = np.loadtxt(filepath)
    n_samples = int(rawdata[0])
    dt = rawdata[1]
    voltage = rawdata[4:]  # skip first 4 values (n_samples, dt, 2 header values)
    voltage = voltage - np.mean(voltage)  # remove DC
    time = np.array(range(0, n_samples)) * dt
    return time, voltage, dt


def read_calibration_csv(filepath):
    """Read hydrophone calibration CSV."""
    data = np.loadtxt(filepath, skiprows=1, delimiter=",")
    return {
        "frequency": data[:, 0] * 1e6,  # MHz -> Hz
        "real": data[:, 1],
        "imag": data[:, 2],
        "varreal": data[:, 3],
        "varimag": data[:, 4],
        "kovar": data[:, 5],
    }


def bodeequation(frequencies, amplitudes, varamplitudes):
    """Reconstruct phase from amplitude using Bode equation."""
    df = frequencies[1] - frequencies[0]
    phase = np.zeros_like(amplitudes)
    varphase = np.zeros_like(amplitudes)
    for i in range(np.size(frequencies)):
        numerator = np.log(amplitudes) - np.log(amplitudes[i])
        denominator = frequencies**2 - frequencies[i]**2
        denominator[i] = 1
        phase[i] = 2.0 * frequencies[i] / np.pi * df * np.sum(numerator / denominator)
        denominatoru = amplitudes * denominator
        numeratoru = np.ones_like(denominatoru)
        numeratoru[i] = 0
        varphase[i] = (2.0 * frequencies[i] / np.pi * df)**2 * np.sum(
            ((numeratoru / denominatoru)**2) * varamplitudes
        )
    return phase, varphase


def make_filter(freq, fc, filter_type):
    """Create regularization filter in PyDynamic format [Re..., Im...]."""
    f = freq  # positive frequencies only (first half)
    if filter_type == "LowPass":
        Hc = 1.0 / (1.0 + 1j * f / (fc * 1.555))**2
    elif filter_type == "CriticalDamping":
        Hc = 1.0 / (1.0 + 1.28719 * 1j * f / fc + 0.41421 * (1j * f / fc)**2)
    elif filter_type == "Bessel":
        Hc = 1.0 / (1.0 + 1.3617 * 1j * f / fc - 0.6180 * f**2 / fc**2)
    elif filter_type == "None":
        Hc = np.ones_like(f, dtype=complex)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Phase correction: remove linear phase component up to -3dB point
    if filter_type != "None":
        ind3dB = np.argmin(np.abs(np.abs(Hc) - np.sqrt(0.5)))
        if ind3dB > 1:
            w = np.linalg.lstsq(f[:ind3dB].reshape(-1, 1), np.angle(Hc[:ind3dB]), rcond=1)[0]
            Hc = Hc * np.exp(-1j * w[0] * f)

    filt = np.hstack([np.real(Hc), np.imag(Hc)])
    return filt


def pulseparameter(time, pressure, u_pressure):
    """Calculate pulse parameters matching original tutorial."""
    assert len(time) == len(pressure)
    if len(np.shape(u_pressure)) == 0:
        u_pressure = np.ones_like(pressure) * u_pressure
    if len(np.shape(u_pressure)) == 1:
        u_pressure = np.diag(u_pressure**2.0)
    dt = (max(time) - min(time)) / (len(time) - 1)
    result = {"dt": dt}
    pc_index = np.argmax(pressure)
    result["pc_index"] = int(pc_index)
    result["pc_value"] = float(pressure[pc_index])
    result["pc_uncertainty"] = float(np.sqrt(u_pressure[pc_index, pc_index]))
    result["pc_time"] = float(time[pc_index])
    pr_index = np.argmin(pressure)
    result["pr_index"] = int(pr_index)
    result["pr_value"] = float(-pressure[pr_index])
    result["pr_uncertainty"] = float(np.sqrt(u_pressure[pr_index, pr_index]))
    result["pr_time"] = float(time[pr_index])
    result["ppsi_value"] = float(np.sum(np.square(pressure)) * dt)
    C = 2 * np.abs(pressure) * dt
    var = np.dot(C, np.dot(u_pressure, np.transpose(C)))
    result["ppsi_uncertainty"] = float(np.sqrt(var))
    return result


def run_single_pattern(pattern_idx, measurement_file, noise_file, cal_file,
                       hyd_name, meas_type, usebode, filter_type, fc):
    """Run a single deconvolution pattern using PyDynamic."""
    # 1. Read measurement signal
    meas_path = os.path.join(DATA_DIR, measurement_file)
    time, voltage, dt = read_dat_signal(meas_path)
    n_samples = len(voltage)

    # 2. Read noise and estimate uncertainty
    noise_path = os.path.join(DATA_DIR, noise_file)
    noise_voltage = np.loadtxt(noise_path)[4:]
    stdev = np.std(noise_voltage)
    uncertainty = np.ones(n_samples) * stdev

    # 3. Calculate frequency scale
    frequency = calcfreqscale(time)

    # 4. GUM_DFT
    spectrum, varspec = PyDynamic.GUM_DFT(voltage, uncertainty**2)
    # Normalize
    spectrum = 2 * spectrum / n_samples
    varspec = 4 * varspec / n_samples**2

    # 5. Read calibration data
    cal_path = os.path.join(DATA_DIR, cal_file)
    hyd_data = read_calibration_csv(cal_path)

    # 6. Frequency range selection
    imin = findnearestmatch(hyd_data["frequency"], FMIN)
    imax = findnearestmatch(hyd_data["frequency"], FMAX)

    # 7. Interpolation and Bode/direct calibration
    if usebode:
        # Amplitude from real+imag
        amp = np.sqrt(hyd_data["real"]**2 + hyd_data["imag"]**2)
        summand_real = (hyd_data["real"] / amp)**2 * hyd_data["varreal"]
        summand_imag = (hyd_data["imag"] / amp)**2 * hyd_data["varimag"]
        summand_corr = 2 * (hyd_data["real"] / amp) * (hyd_data["imag"] / amp) * hyd_data["kovar"]
        varamp = summand_real + summand_imag + summand_corr

        # Trim to frequency range
        freq_trim = hyd_data["frequency"][imin:imax + 1]
        amp_trim = amp[imin:imax + 1]
        varamp_trim = varamp[imin:imax + 1]

        # Interpolate amplitude
        half_freq = frequency[0:int(len(frequency) / 2)]
        ampip = np.interp(half_freq, freq_trim, amp_trim)
        varampip = np.interp(half_freq, freq_trim, varamp_trim)

        # Bode equation
        phaseip, varphaseip = bodeequation(half_freq, ampip, varampip)

        # AmpPhase2DFT
        x, ux = PyDynamic.AmpPhase2DFT(ampip, phaseip, np.hstack([varampip, varphaseip]))
    else:
        # Direct complex calibration - interpolate real and imag separately
        half_freq = frequency[0:int(len(frequency) / 2)]
        _, real_ip, varreal_ip, _ = interp1d_unc(
            half_freq,
            hyd_data["frequency"][imin:imax + 1],
            hyd_data["real"][imin:imax + 1],
            hyd_data["varreal"][imin:imax + 1],
            bounds_error=False, fill_value="extrapolate", fill_unc="extrapolate", returnC=True)
        _, imag_ip, varimag_ip, _ = interp1d_unc(
            half_freq,
            hyd_data["frequency"][imin:imax + 1],
            hyd_data["imag"][imin:imax + 1],
            hyd_data["varimag"][imin:imax + 1],
            bounds_error=False, fill_value="extrapolate", fill_unc="extrapolate", returnC=True)
        kovar_ip = np.interp(half_freq, hyd_data["frequency"][imin:imax + 1],
                             hyd_data["kovar"][imin:imax + 1])

        x = np.append(real_ip, imag_ip)
        a = np.hstack([np.diag(varreal_ip), np.diag(kovar_ip)])
        b = np.hstack([np.diag(kovar_ip), np.diag(varimag_ip)])
        ux = np.vstack([a, b])

    # 8. DFT_deconv
    deconv_p, deconv_Up = PyDynamic.DFT_deconv(x, spectrum, ux, varspec)

    # 9. Regularization filter
    half_freq = frequency[0:int(len(frequency) / 2)]
    filt = make_filter(half_freq, fc, filter_type)

    # 10. DFT_multiply (filter application)
    regul_p, regul_Up = PyDynamic.DFT_multiply(deconv_p, filt, deconv_Up)

    # 11. GUM_iDFT (back to time domain)
    sigp_raw, Usigp = PyDynamic.GUM_iDFT(regul_p, regul_Up)

    # 12. De-normalize
    N_sig = len(sigp_raw)
    sigp = sigp_raw * N_sig / 2
    Usigp_denorm = Usigp * (N_sig**2) / 4

    # Also get unfiltered deconvolution in time domain
    deconv_time_raw, deconv_time_U = PyDynamic.GUM_iDFT(deconv_p, deconv_Up)
    deconv_time_p = deconv_time_raw * len(deconv_time_raw) / 2

    # 13. Scaled signal (voltage / sensitivity at fundamental freq)
    M = int(len(frequency) / 2)
    # Find hydrophone sensitivity at fundamental frequency
    # Use interpolated hyd data
    hyd_amp = np.sqrt(x[:M]**2 + x[M:]**2)
    sig_amp = np.sqrt(spectrum[:M]**2 + spectrum[M:]**2)
    iffun = np.argmax(sig_amp)
    hydempffun = hyd_amp[iffun]
    scaled = voltage / hydempffun

    # 14. Uncertainty with wavelet model (simplified - just GUM uncertainty)
    deltasigp = np.sqrt(np.diag(Usigp_denorm))

    # 15. Pulse parameters
    pp = pulseparameter(time, sigp, Usigp_denorm)

    return {
        "time": time,
        "scaled": scaled,
        "deconvolved": deconv_time_p,
        "regularized": sigp,
        "uncertainty": deltasigp,
        "pulse_params": pp,
        "n_samples": n_samples,
    }


def result_filename(meas_type, hyd_name, usebode, filter_type, fc):
    """Generate result filename matching original naming convention."""
    bode_str = "_Bode" if usebode else ""
    if filter_type == "None":
        fc_str = ""
    else:
        fc_str = f"{fc / 1e6:.0f}MHz"
    return f"{meas_type}_{hyd_name}{bode_str}_{filter_type}{fc_str}.csv"


def save_result(filepath, result):
    """Save result in CSV format (time;scaled;deconvolved;regularized;uncertainty)."""
    header = "time;scaled;deconvolved;regularized;uncertainty(k=1)"
    data = np.column_stack([
        result["time"],
        result["scaled"],
        result["deconvolved"],
        result["regularized"],
        result["uncertainty"],
    ])
    pp = result["pulse_params"]
    footer = (f"pc_value={pp['pc_value']};pc_uncertainty={pp['pc_uncertainty']};"
              f"pc_time={pp['pc_time']};pr_value={pp['pr_value']};"
              f"pr_uncertainty={pp['pr_uncertainty']};pr_time={pp['pr_time']};"
              f"ppsi_value={pp['ppsi_value']};ppsi_uncertainty={pp['ppsi_uncertainty']}")
    np.savetxt(filepath, data, header=header, delimiter=";", footer=footer)


def verify_against_original():
    """Verify PyDynamic results against original 16 result files."""
    print("\n=== Verifying against original Results/ ===")
    pass_count = 0
    fail_count = 0

    for pat in PATTERNS:
        idx, meas_file, noise_file, cal_file, hyd_name, meas_type, usebode_default = pat
        fc = FC_MAP[meas_type]

        # Original results use default Bode setting + LowPass filter
        bode_str = "_Bode" if usebode_default else ""
        orig_name_map = {
            ("M-Mode3MHz", "GAMPTMH44"): f"M-Mode3MHz_GAMPTMH44{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode3MHz", "GAMPTMH46"): f"M-Mode3MHz_GAMPTMH46{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode3MHz", "ONDA1704"): f"M-Mode3MHz_ONDA1704{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode3MHz", "PrecisionAcoustics1434"): f"M-Mode3MHz_PrecisionAcoustics1434{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode3MHz", "GAMPTMH44"): f"Pulse-Doppler-Mode3MHz_GAMPTMH44{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode3MHz", "GAMPTMH46"): f"Pulse-Doppler-Mode3MHz_GAMPTMH46{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode3MHz", "ONDA1704"): f"Pulse-Doppler-Mode3MHz_ONDA1704{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode3MHz", "PrecisionAcoustics1434"): f"Pulse-Doppler-Mode3MHz_PrecisionAcoustics1434{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode6MHz", "GAMPTMH44"): f"M-Mode6MHz_GAMPTMH44{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode6MHz", "GAMPTMH46"): f"M-Mode6MHz_GAMPTMH46{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode6MHz", "ONDA1704"): f"M-Mode6MHz_ONDA1704{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("M-Mode6MHz", "PrecisionAcoustics1434"): f"M-Mode6MHz_PrecisionAcoustics1434{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode7MHz", "GAMPTMH44"): f"Pulse-Doppler-Mode7MHz_GAMPTMH44{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode7MHz", "GAMPTMH46"): f"Pulse-Doppler-Mode7MHz_GAMPTMH46{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode7MHz", "ONDA1704"): f"Pulse-Doppler-Mode7MHz_ONDA1704{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
            ("Pulse-Doppler-Mode7MHz", "PrecisionAcoustics1434"): f"Pulse-Doppler-Mode7MHz_PrecisionAcoustics1434{bode_str}_LowPass{fc/1e6:.0f}MHz.dat",
        }
        orig_filename = orig_name_map.get((meas_type, hyd_name))
        if orig_filename is None:
            continue

        orig_path = os.path.join(ORIGINAL_RESULTS_DIR, orig_filename)
        ref_name = result_filename(meas_type, hyd_name, usebode_default, "LowPass", fc)
        ref_path = os.path.join(RESULTS_DIR, ref_name)

        if not os.path.exists(orig_path) or not os.path.exists(ref_path):
            print(f"  SKIP i={idx}: files not found")
            continue

        # Load original (semicolon-delimited, first line is comment header)
        orig_data = np.loadtxt(orig_path, delimiter=";", comments="#")
        ref_data = np.loadtxt(ref_path, delimiter=";", comments="#")

        # Compare regularized column (index 3) and time column (index 0)
        if orig_data.shape != ref_data.shape:
            print(f"  FAIL i={idx}: shape mismatch {orig_data.shape} vs {ref_data.shape}")
            fail_count += 1
            continue

        # Relative error on regularized signal
        reg_orig = orig_data[:, 3]
        reg_ref = ref_data[:, 3]
        max_val = np.max(np.abs(reg_orig))
        if max_val > 0:
            rel_err = np.max(np.abs(reg_orig - reg_ref)) / max_val
        else:
            rel_err = np.max(np.abs(reg_orig - reg_ref))

        if rel_err < 1e-6:
            print(f"  PASS i={idx:2d} {meas_type}/{hyd_name}: rel_err = {rel_err:.2e}")
            pass_count += 1
        else:
            print(f"  FAIL i={idx:2d} {meas_type}/{hyd_name}: rel_err = {rel_err:.2e}")
            fail_count += 1

    print(f"\nVerification: {pass_count} PASS, {fail_count} FAIL out of {pass_count + fail_count}")
    return fail_count == 0


def run_single_and_save(args_str):
    """Entry point for subprocess execution of a single pattern."""
    import json
    args = json.loads(args_str)
    result = run_single_pattern(
        args["idx"], args["meas_file"], args["noise_file"], args["cal_file"],
        args["hyd_name"], args["meas_type"], args["usebode"], args["filter_type"],
        args["fc"]
    )
    save_result(args["output_path"], result)


def main():
    import gc
    import subprocess
    import json

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(PATTERNS) * len(FILTER_TYPES) * len(BODE_OPTIONS)
    count = 0
    errors = []

    for pat in PATTERNS:
        idx, meas_file, noise_file, cal_file, hyd_name, meas_type, usebode_default = pat
        fc = FC_MAP[meas_type]

        for filter_type in FILTER_TYPES:
            for usebode in BODE_OPTIONS:
                count += 1
                fname = result_filename(meas_type, hyd_name, usebode, filter_type, fc)
                output_path = os.path.join(RESULTS_DIR, fname)

                # Skip if already generated
                if os.path.exists(output_path):
                    print(f"[{count:3d}/{total}] i={idx:2d} {fname} ... SKIP (exists)")
                    continue

                print(f"[{count:3d}/{total}] i={idx:2d} {fname} ...", end=" ", flush=True)

                # Run in subprocess to avoid memory accumulation
                args = json.dumps({
                    "idx": idx, "meas_file": meas_file, "noise_file": noise_file,
                    "cal_file": cal_file, "hyd_name": hyd_name, "meas_type": meas_type,
                    "usebode": usebode, "filter_type": filter_type, "fc": fc,
                    "output_path": output_path,
                })
                cmd = [sys.executable, "-c",
                       f"import sys; sys.path.insert(0,'{SCRIPT_DIR}');"
                       f"from generate_reference import run_single_and_save;"
                       f"run_single_and_save('''{args}''')"]
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if proc.returncode == 0:
                        print("OK")
                    else:
                        print(f"ERROR: {proc.stderr[-200:]}")
                        errors.append((fname, proc.stderr[-200:]))
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    errors.append((fname, "timeout"))

                gc.collect()

    print(f"\n=== Completed: {count - len(errors)}/{total} OK, {len(errors)} errors ===")
    if errors:
        for fname, err in errors:
            print(f"  ERROR: {fname}: {err}")

    # Verify against original results
    verify_against_original()


if __name__ == "__main__":
    main()
