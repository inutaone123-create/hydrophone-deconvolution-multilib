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
from typing import Tuple


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
