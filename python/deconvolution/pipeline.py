"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

This implementation: 2024
License: CC BY 4.0
https://creativecommons.org/licenses/by/4.0/

Full GUM-based deconvolution pipeline WITHOUT PyDynamic dependency.
Implements GUM_DFT, GUM_iDFT, DFT_deconv, DFT_multiply with uncertainty propagation.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# ── Frequency scale ──

def calcfreqscale(timeseries: np.ndarray, sign2: int = 1) -> np.ndarray:
    """Calculate frequency scale matching PyDynamic's GUM_DFT output format.

    Returns frequencies in PyDynamic layout: [f_cos(0..M-1), f_sin(0..M-1)]
    where M = N/2 + 1.
    """
    n = np.size(timeseries)
    sign = -1 if sign2 < 0 else 1
    fmax = 1 / ((np.max(timeseries) - np.min(timeseries)) * n / (n - 1)) * (n - 1)
    f = np.linspace(0, fmax, n)
    if sign2 == 0:
        return f[0:int(n / 2.0 + 1)]
    return np.hstack((f[0:int(n / 2.0 + 1)], sign * f[0:int(n / 2 + 1)]))


# ── GUM DFT (uncertainty-aware DFT) ──

def gum_dft(x: np.ndarray, Ux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """GUM-compliant DFT with uncertainty propagation.

    Compatible with PyDynamic's GUM_DFT. Uses standard DFT convention:
    F[k] = Σ x[n] exp(-j2πkn/N), stored as [Re₀..Re_{M-1}, Im₀..Im_{M-1}].

    Args:
        x: Time-domain signal [N]
        Ux: Variance vector [N] (diagonal of covariance matrix)

    Returns:
        F: Spectrum in PyDynamic format [Re₀..Re_{M-1}, Im₀..Im_{M-1}] (2M)
        UF: Covariance matrix [2M × 2M]
    """
    N = len(x)
    M = N // 2 + 1

    # Best estimate via rfft (matches PyDynamic)
    F_complex = np.fft.rfft(x)
    F = np.concatenate([np.real(F_complex), np.imag(F_complex)])

    # Sensitivity matrix: CxCos[k,n] = cos(2πkn/N), CxSin[k,n] = -sin(2πkn/N)
    beta = 2 * np.pi * np.arange(N) / N
    k_idx = np.arange(M)
    phase = np.outer(k_idx, beta)  # [M, N]

    CxCos = np.cos(phase)    # [M, N]
    CxSin = -np.sin(phase)   # [M, N]

    # Uncertainty propagation: UF = C · diag(Ux) · Cᵀ
    C = np.vstack([CxCos, CxSin])  # [2M, N]

    if np.ndim(Ux) == 1 or (np.ndim(Ux) == 2 and Ux.shape[0] != Ux.shape[1]):
        # Diagonal case: Ux is variance vector
        Ux_vec = Ux.ravel()
        CU = C * Ux_vec[np.newaxis, :]
        UF = CU @ C.T
    else:
        # Full covariance matrix
        UF = C @ Ux @ C.T

    return F, UF


# ── GUM iDFT (uncertainty-aware inverse DFT) ──

def gum_idft(F: np.ndarray, UF: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """GUM-compliant inverse DFT with uncertainty propagation.

    Compatible with PyDynamic's GUM_iDFT.

    Args:
        F: Spectrum in PyDynamic format [Re₀..Re_{M-1}, Im₀..Im_{M-1}] (2M)
        UF: Covariance matrix [2M × 2M]

    Returns:
        x: Time-domain signal [N]
        Ux: Covariance matrix [N × N]
    """
    total = len(F)
    M = total // 2
    N = 2 * (M - 1)  # assumes even-length signal

    # Best estimate via irfft (matches PyDynamic)
    x = np.fft.irfft(F[:M] + 1j * F[M:], n=N)

    # Sensitivity matrices (without 1/N factor, applied at end)
    beta = 2 * np.pi * np.arange(N) / N
    k_idx = np.arange(M)
    k_beta = np.outer(beta, k_idx)  # [N, M]

    Cc = np.cos(k_beta)   # [N, M]
    Cs = -np.sin(k_beta)  # [N, M]

    # Adjust for rfft convention: multiply interior bins by 2
    Cc[:, 1:] *= 2
    Cs[:, 1:] *= 2
    # Undo factor 2 for Nyquist (even N)
    if N % 2 == 0:
        Cc[:, M - 1] *= 0.5
        Cs[:, M - 1] *= 0.5

    # Uncertainty propagation
    RR = UF[:M, :M]
    RI = UF[:M, M:]
    II = UF[M:, M:]

    Ux = Cc @ RR @ Cc.T
    term2 = Cc @ RI @ Cs.T
    Ux += term2 + term2.T
    Ux += Cs @ II @ Cs.T

    Ux /= N**2

    return x, Ux


# ── DFT deconvolution ──

def dft_deconv(H: np.ndarray, Y: np.ndarray,
               UH: np.ndarray, UY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency-domain deconvolution X = Y / H with uncertainty propagation.

    All inputs in PyDynamic format [Re₀..Re_{M-1}, Im₀..Im_{M-1}].

    Args:
        H: Transfer function (hydrophone) [2M]
        Y: Measured spectrum [2M]
        UH: Covariance matrix of H [2M × 2M]
        UY: Covariance matrix of Y [2M × 2M]

    Returns:
        X: Deconvolved spectrum [2M]
        UX: Covariance matrix [2M × 2M]
    """
    M = len(H) // 2
    Hr = H[:M]
    Hi = H[M:]
    Yr = Y[:M]
    Yi = Y[M:]

    # Complex division: X = Y / H
    # Xr = (Yr*Hr + Yi*Hi) / (Hr² + Hi²)
    # Xi = (Yi*Hr - Yr*Hi) / (Hr² + Hi²)
    denom = Hr**2 + Hi**2
    denom = np.where(denom == 0, 1e-30, denom)

    Xr = (Yr * Hr + Yi * Hi) / denom
    Xi = (Yi * Hr - Yr * Hi) / denom
    X = np.concatenate([Xr, Xi])

    # Jacobian J = ∂X/∂(Y,H) — block structure
    # For each frequency bin k, the Jacobian is 2×4 (output: Xr,Xi; input: Yr,Yi,Hr,Hi)
    # We build the full [2M, 4M] Jacobian and compute UX = J · blkdiag(UY, UH) · Jᵀ

    # ∂Xr/∂Yr = Hr/denom,  ∂Xr/∂Yi = Hi/denom
    # ∂Xr/∂Hr = (Yr*denom - (Yr*Hr+Yi*Hi)*2*Hr) / denom² = (Yr*(Hi²-Hr²) - 2*Yi*Hr*Hi) / denom²
    # Actually, let's use the standard complex division derivatives:
    # ∂Xr/∂Hr = (Yr*(Hr²+Hi²) - (Yr*Hr+Yi*Hi)*2*Hr) / denom² = (Yr*Hi² - Yi*2*Hr*Hi - Yr*Hr²) / denom²
    # Simplify: = (-Yr*(Hr²-Hi²) - 2*Yi*Hr*Hi) / denom²

    # More carefully:
    # Xr = (Yr*Hr + Yi*Hi) / D where D = Hr² + Hi²
    # ∂Xr/∂Yr = Hr/D
    # ∂Xr/∂Yi = Hi/D
    # ∂Xr/∂Hr = (Yr*D - (Yr*Hr+Yi*Hi)*2*Hr) / D² = Yr/D - 2*Hr*Xr/D = (Yr - 2*Hr*Xr)/D
    # ∂Xr/∂Hi = (Yi*D - (Yr*Hr+Yi*Hi)*2*Hi) / D² = Yi/D - 2*Hi*Xr/D = (Yi - 2*Hi*Xr)/D

    # Xi = (Yi*Hr - Yr*Hi) / D
    # ∂Xi/∂Yr = -Hi/D
    # ∂Xi/∂Yi = Hr/D
    # ∂Xi/∂Hr = (Yi*D - (Yi*Hr-Yr*Hi)*2*Hr) / D² = Yi/D - 2*Hr*Xi/D = (Yi - 2*Hr*Xi)/D
    # ∂Xi/∂Hi = (-Yr*D - (Yi*Hr-Yr*Hi)*2*Hi) / D² = -Yr/D - 2*Hi*Xi/D = (-Yr - 2*Hi*Xi)/D

    # Build Jacobian w.r.t. [Y, H] = [Yr, Yi, Hr, Hi] in PyDynamic layout
    # Input vector: [Yr₀..Yr_{M-1}, Yi₀..Yi_{M-1}, Hr₀..Hr_{M-1}, Hi₀..Hi_{M-1}]
    # Output vector: [Xr₀..Xr_{M-1}, Xi₀..Xi_{M-1}]

    # For efficiency, compute per-bin and build block-diagonal Jacobian
    UX = np.zeros((2 * M, 2 * M))

    for k in range(M):
        D = denom[k]
        # 2×2 Jacobian w.r.t. Y (Yr_k, Yi_k)
        JY = np.array([
            [Hr[k] / D, Hi[k] / D],
            [-Hi[k] / D, Hr[k] / D]
        ])
        # 2×2 Jacobian w.r.t. H (Hr_k, Hi_k)
        JH = np.array([
            [(Yr[k] - 2 * Hr[k] * Xr[k]) / D, (Yi[k] - 2 * Hi[k] * Xr[k]) / D],
            [(Yi[k] - 2 * Hr[k] * Xi[k]) / D, (-Yr[k] - 2 * Hi[k] * Xi[k]) / D]
        ])

        # Extract 2×2 covariance blocks for Y and H at bin k
        UY_k = np.array([
            [UY[k, k], UY[k, k + M]],
            [UY[k + M, k], UY[k + M, k + M]]
        ])
        UH_k = np.array([
            [UH[k, k], UH[k, k + M]],
            [UH[k + M, k], UH[k + M, k + M]]
        ])

        # Propagate: UX_k = JY · UY_k · JYᵀ + JH · UH_k · JHᵀ
        UX_k = JY @ UY_k @ JY.T + JH @ UH_k @ JH.T

        UX[k, k] = UX_k[0, 0]
        UX[k, k + M] = UX_k[0, 1]
        UX[k + M, k] = UX_k[1, 0]
        UX[k + M, k + M] = UX_k[1, 1]

    return X, UX


# ── DFT multiply ──

def dft_multiply(Y: np.ndarray, F: np.ndarray,
                 UY: np.ndarray,
                 UF: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency-domain multiplication Z = Y * F with uncertainty propagation.

    Args:
        Y: Input spectrum [2M] (PyDynamic format)
        F: Filter spectrum [2M] (PyDynamic format, deterministic)
        UY: Covariance matrix of Y [2M × 2M]
        UF: Covariance matrix of F (optional, usually None for deterministic filter)

    Returns:
        Z: Product spectrum [2M]
        UZ: Covariance matrix [2M × 2M]
    """
    M = len(Y) // 2
    Yr = Y[:M]
    Yi = Y[M:]
    Fr = F[:M]
    Fi = F[M:]

    # Complex multiplication: Z = Y * F
    Zr = Yr * Fr - Yi * Fi
    Zi = Yr * Fi + Yi * Fr
    Z = np.concatenate([Zr, Zi])

    # Jacobian w.r.t. Y (F is deterministic unless UF given)
    # ∂Zr/∂Yr = Fr,  ∂Zr/∂Yi = -Fi
    # ∂Zi/∂Yr = Fi,  ∂Zi/∂Yi = Fr
    UZ = np.zeros((2 * M, 2 * M))

    for k in range(M):
        JY = np.array([
            [Fr[k], -Fi[k]],
            [Fi[k], Fr[k]]
        ])
        UY_k = np.array([
            [UY[k, k], UY[k, k + M]],
            [UY[k + M, k], UY[k + M, k + M]]
        ])
        UZ_k = JY @ UY_k @ JY.T

        if UF is not None:
            JF = np.array([
                [Yr[k], -Yi[k]],
                [Yi[k], Yr[k]]
            ])
            UF_k = np.array([
                [UF[k, k], UF[k, k + M]],
                [UF[k + M, k], UF[k + M, k + M]]
            ])
            UZ_k += JF @ UF_k @ JF.T

        UZ[k, k] = UZ_k[0, 0]
        UZ[k, k + M] = UZ_k[0, 1]
        UZ[k + M, k] = UZ_k[1, 0]
        UZ[k + M, k + M] = UZ_k[1, 1]

    return Z, UZ


# ── AmpPhase to DFT conversion ──

def amp_phase_to_dft(amp: np.ndarray, phase: np.ndarray,
                     Uap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert amplitude+phase to real+imag (PyDynamic format) with uncertainty.

    Args:
        amp: Amplitude [M]
        phase: Phase [M]
        Uap: Uncertainty vector [2M] = [var_amp₀..var_amp_{M-1}, var_phase₀..var_phase_{M-1}]

    Returns:
        x: [Re₀..Re_{M-1}, Im₀..Im_{M-1}]
        ux: Covariance matrix [2M × 2M]
    """
    M = len(amp)
    re = amp * np.cos(phase)
    im = amp * np.sin(phase)
    x = np.concatenate([re, im])

    var_amp = Uap[:M]
    var_phase = Uap[M:]

    ux = np.zeros((2 * M, 2 * M))
    for k in range(M):
        # Jacobian of (re, im) w.r.t. (amp, phase)
        J = np.array([
            [np.cos(phase[k]), -amp[k] * np.sin(phase[k])],
            [np.sin(phase[k]), amp[k] * np.cos(phase[k])]
        ])
        U_in = np.array([
            [var_amp[k], 0],
            [0, var_phase[k]]
        ])
        U_out = J @ U_in @ J.T
        ux[k, k] = U_out[0, 0]
        ux[k, k + M] = U_out[0, 1]
        ux[k + M, k] = U_out[1, 0]
        ux[k + M, k + M] = U_out[1, 1]

    return x, ux


# ── Bode equation ──

def bode_equation(frequencies: np.ndarray, amplitudes: np.ndarray,
                  varamplitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct phase from amplitude using the Bode integral equation.

    Based on the Kramers-Kronig / Hilbert transform relation between
    log-amplitude and phase for minimum-phase systems.
    Vectorized implementation for performance.
    """
    df = frequencies[1] - frequencies[0]
    n = np.size(frequencies)
    logamp = np.log(amplitudes)

    # Build [n x n] matrix: denom[i,j] = freq[j]^2 - freq[i]^2
    f2 = frequencies ** 2
    denom = f2[np.newaxis, :] - f2[:, np.newaxis]  # [n, n]
    np.fill_diagonal(denom, 1.0)  # avoid division by zero

    # Phase: phase[i] = coeff[i] * sum_j (logamp[j] - logamp[i]) / denom[i,j]
    coeff = 2.0 * frequencies / np.pi * df
    num = logamp[np.newaxis, :] - logamp[:, np.newaxis]  # [n, n]
    phase = coeff * np.sum(num / denom, axis=1)

    # Variance: varphase[i] = coeff[i]^2 * sum_{j!=i} (1/(amp[j]*denom[i,j]))^2 * varamp[j]
    denominatoru = amplitudes[np.newaxis, :] * denom  # [n, n]
    mask = np.ones((n, n))
    np.fill_diagonal(mask, 0.0)
    varphase = coeff ** 2 * np.sum((mask / denominatoru) ** 2 * varamplitudes[np.newaxis, :], axis=1)

    return phase, varphase


# ── Regularization filter ──

def regularization_filter(freq: np.ndarray, fc: float,
                          filter_type: str) -> np.ndarray:
    """Create regularization filter in PyDynamic format [Re..., Im...].

    Args:
        freq: Positive frequencies (first half) [M]
        fc: Cutoff frequency [Hz]
        filter_type: "LowPass", "CriticalDamping", "Bessel", or "None"

    Returns:
        filt: Filter in PyDynamic format [2M]
    """
    f = freq
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

    # Phase correction: remove linear phase up to -3dB point
    if filter_type != "None":
        ind3dB = np.argmin(np.abs(np.abs(Hc) - np.sqrt(0.5)))
        if ind3dB > 1:
            w = np.linalg.lstsq(f[:ind3dB].reshape(-1, 1),
                                np.angle(Hc[:ind3dB]), rcond=1)[0]
            Hc = Hc * np.exp(-1j * w[0] * f)

    return np.hstack([np.real(Hc), np.imag(Hc)])


# ── I/O functions ──

def read_dat_signal(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read .DAT measurement signal file.

    Format: line0=n_samples, line1=dt, line2-3=skip, line4+=voltage

    Returns:
        time, voltage, dt
    """
    rawdata = np.loadtxt(filepath)
    n_samples = int(rawdata[0])
    dt = rawdata[1]
    voltage = rawdata[4:]
    voltage = voltage - np.mean(voltage)
    time = np.arange(n_samples) * dt
    return time, voltage, dt


def read_calibration_csv(filepath: str) -> Dict:
    """Read hydrophone calibration CSV.

    Returns dict with frequency, real, imag, varreal, varimag, kovar.
    """
    data = np.loadtxt(filepath, skiprows=1, delimiter=",")
    return {
        "frequency": data[:, 0] * 1e6,
        "real": data[:, 1],
        "imag": data[:, 2],
        "varreal": data[:, 3],
        "varimag": data[:, 4],
        "kovar": data[:, 5],
    }


# ── Interpolation ──

def interpolate_calibration(target_freq: np.ndarray, hyd_data: Dict,
                            fmin: float, fmax: float) -> Dict:
    """Interpolate calibration data to target frequency grid.

    Args:
        target_freq: Target frequencies (positive half) [M]
        hyd_data: Calibration data dict
        fmin, fmax: Frequency range for calibration data

    Returns:
        Dict with interpolated real, imag, varreal, varimag, kovar
    """
    imin = np.argmin(np.abs(hyd_data["frequency"] - fmin))
    imax = np.argmin(np.abs(hyd_data["frequency"] - fmax))

    src_freq = hyd_data["frequency"][imin:imax + 1]
    result = {}
    for key in ["real", "imag", "varreal", "varimag", "kovar"]:
        result[key] = np.interp(target_freq, src_freq, hyd_data[key][imin:imax + 1])
    return result


# ── Full pipeline ──

def full_pipeline(measurement_file: str, noise_file: str, cal_file: str,
                  usebode: bool, filter_type: str, fc: float,
                  fmin: float = 1e6, fmax: float = 60e6) -> Dict:
    """Run complete deconvolution pipeline.

    Args:
        measurement_file: Path to measurement .DAT file
        noise_file: Path to noise .DAT file
        cal_file: Path to calibration .CSV file
        usebode: Use Bode equation for phase reconstruction
        filter_type: "LowPass", "CriticalDamping", "Bessel", or "None"
        fc: Filter cutoff frequency [Hz]
        fmin, fmax: Calibration frequency range [Hz]

    Returns:
        Dict with time, scaled, deconvolved, regularized, uncertainty, pulse_params
    """
    # 1. Read measurement signal
    time, voltage, dt = read_dat_signal(measurement_file)
    N = len(voltage)

    # 2. Read noise and estimate uncertainty
    noise_voltage = np.loadtxt(noise_file)[4:]
    stdev = np.std(noise_voltage)
    uncertainty_vec = np.ones(N) * stdev

    # 3. Calculate frequency scale
    frequency = calcfreqscale(time)
    M = N // 2 + 1
    half_freq = frequency[:M]

    # 4. GUM_DFT
    spectrum, varspec = gum_dft(voltage, uncertainty_vec**2)
    # Normalize
    spectrum = 2 * spectrum / N
    varspec = 4 * varspec / N**2

    # 5. Read calibration data
    hyd_data = read_calibration_csv(cal_file)

    # 6-7. Calibration: Bode or direct
    if usebode:
        # Compute amplitude from complex calibration
        amp = np.sqrt(hyd_data["real"]**2 + hyd_data["imag"]**2)
        summand_real = (hyd_data["real"] / amp)**2 * hyd_data["varreal"]
        summand_imag = (hyd_data["imag"] / amp)**2 * hyd_data["varimag"]
        summand_corr = 2 * (hyd_data["real"] / amp) * (hyd_data["imag"] / amp) * hyd_data["kovar"]
        varamp = summand_real + summand_imag + summand_corr

        # Trim to frequency range
        imin = np.argmin(np.abs(hyd_data["frequency"] - fmin))
        imax = np.argmin(np.abs(hyd_data["frequency"] - fmax))
        freq_trim = hyd_data["frequency"][imin:imax + 1]
        amp_trim = amp[imin:imax + 1]
        varamp_trim = varamp[imin:imax + 1]

        # Interpolate amplitude
        ampip = np.interp(half_freq, freq_trim, amp_trim)
        varampip = np.interp(half_freq, freq_trim, varamp_trim)

        # Bode equation
        phaseip, varphaseip = bode_equation(half_freq, ampip, varampip)

        # AmpPhase -> DFT
        x, ux = amp_phase_to_dft(ampip, phaseip, np.hstack([varampip, varphaseip]))
    else:
        # Direct complex interpolation
        interp_data = interpolate_calibration(half_freq, hyd_data, fmin, fmax)
        x = np.concatenate([interp_data["real"], interp_data["imag"]])
        a = np.hstack([np.diag(interp_data["varreal"]), np.diag(interp_data["kovar"])])
        b = np.hstack([np.diag(interp_data["kovar"]), np.diag(interp_data["varimag"])])
        ux = np.vstack([a, b])

    # 8. DFT_deconv
    deconv_p, deconv_Up = dft_deconv(x, spectrum, ux, varspec)

    # 9. Regularization filter
    filt = regularization_filter(half_freq, fc, filter_type)

    # 10. DFT_multiply
    regul_p, regul_Up = dft_multiply(deconv_p, filt, deconv_Up)

    # 11. GUM_iDFT for regularized signal
    sigp_raw, Usigp_raw = gum_idft(regul_p, regul_Up)
    N_sig = len(sigp_raw)
    sigp = sigp_raw * N_sig / 2
    Usigp = Usigp_raw * (N_sig**2) / 4

    # Also get unfiltered deconvolved signal
    deconv_raw, deconv_U_raw = gum_idft(deconv_p, deconv_Up)
    deconv_time_p = deconv_raw * len(deconv_raw) / 2

    # Scaled signal
    hyd_amp = np.sqrt(x[:M]**2 + x[M:]**2)
    sig_amp = np.sqrt(spectrum[:M]**2 + spectrum[M:]**2)
    iffun = np.argmax(sig_amp)
    hydempffun = hyd_amp[iffun]
    scaled = voltage / hydempffun

    # Uncertainty (diagonal)
    deltasigp = np.sqrt(np.diag(Usigp))

    # Pulse parameters
    from .core import pulse_parameters
    pp = pulse_parameters(time, sigp, Usigp)

    return {
        "time": time,
        "scaled": scaled,
        "deconvolved": deconv_time_p,
        "regularized": sigp,
        "uncertainty": deltasigp,
        "pulse_params": pp,
        "n_samples": N,
    }
