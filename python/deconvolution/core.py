"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

This implementation: 2024
License: CC BY 4.0
https://creativecommons.org/licenses/by/4.0/
"""

import numpy as np
from typing import Dict, Tuple, Union


def deconvolve_without_uncertainty(
    measured_signal: np.ndarray,
    frequency_response: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Deconvolve hydrophone signal without uncertainty propagation.
    Uses PocketFFT (via NumPy).
    """
    signal_fft = np.fft.fft(measured_signal)
    epsilon = 1e-12
    deconvolved_fft = signal_fft / (frequency_response + epsilon)
    deconvolved = np.fft.ifft(deconvolved_fft).real
    return deconvolved


def deconvolve_with_uncertainty(
    measured_signal: np.ndarray,
    signal_uncertainty: np.ndarray,
    frequency_response: np.ndarray,
    response_uncertainty: np.ndarray,
    sampling_rate: float,
    num_monte_carlo: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deconvolve with uncertainty propagation using Monte Carlo.
    """
    n_samples = len(measured_signal)
    mc_results = np.zeros((num_monte_carlo, n_samples))

    for i in range(num_monte_carlo):
        signal_pert = measured_signal + np.random.normal(
            0, signal_uncertainty, n_samples
        )
        freq_resp_pert = frequency_response + np.random.normal(
            0, response_uncertainty, len(frequency_response)
        ) * (1 + 1j)
        mc_results[i, :] = deconvolve_without_uncertainty(
            signal_pert, freq_resp_pert, sampling_rate
        )

    mean = np.mean(mc_results, axis=0)
    std = np.std(mc_results, axis=0)
    return mean, std


def pulse_parameters(
    time: np.ndarray,
    pressure: np.ndarray,
    u_pressure: Union[float, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate pulse parameters (pc, pr, ppsi) and their uncertainties.

    Based on Weber & Wilkens (2023) analytical method.

    Args:
        time: Time array.
        pressure: Pressure array (deconvolved).
        u_pressure: Uncertainty - scalar, 1D vector, or 2D covariance matrix.

    Returns:
        Dict with pc_value, pc_uncertainty, pc_index, pc_time,
                   pr_value, pr_uncertainty, pr_index, pr_time,
                   ppsi_value, ppsi_uncertainty.
    """
    n = len(pressure)

    # Build covariance matrix U_p
    u_arr = np.asarray(u_pressure, dtype=float)
    if u_arr.ndim == 0:
        # scalar -> diagonal covariance matrix
        U_p = np.diag(np.full(n, u_arr**2))
    elif u_arr.ndim == 1:
        # vector -> diagonal covariance matrix
        U_p = np.diag(u_arr**2)
    else:
        # 2D covariance matrix
        U_p = u_arr

    dt = (time[-1] - time[0]) / (n - 1)

    # pc: compressional peak pressure
    pc_index = int(np.argmax(pressure))
    pc_value = float(pressure[pc_index])
    pc_uncertainty = float(np.sqrt(U_p[pc_index, pc_index]))
    pc_time = float(time[pc_index])

    # pr: rarefactional peak pressure (positive value)
    pr_index = int(np.argmin(pressure))
    pr_value = float(-pressure[pr_index])
    pr_uncertainty = float(np.sqrt(U_p[pr_index, pr_index]))
    pr_time = float(time[pr_index])

    # ppsi: pulse pressure-squared integral
    ppsi_value = float(np.sum(pressure**2) * dt)
    # Sensitivity vector: C = 2 * |p| * dt
    C = 2.0 * np.abs(pressure) * dt
    ppsi_uncertainty = float(np.sqrt(C @ U_p @ C))

    return {
        "pc_value": pc_value,
        "pc_uncertainty": pc_uncertainty,
        "pc_index": pc_index,
        "pc_time": pc_time,
        "pr_value": pr_value,
        "pr_uncertainty": pr_uncertainty,
        "pr_index": pr_index,
        "pr_time": pr_time,
        "ppsi_value": ppsi_value,
        "ppsi_uncertainty": ppsi_uncertainty,
    }
