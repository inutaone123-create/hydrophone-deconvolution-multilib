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
from pathlib import Path


def generate_test_data():
    """Generate deterministic test data for cross-language validation."""
    np.random.seed(12345)
    n = 1024
    sampling_rate = 1e7

    t = np.arange(n) / sampling_rate
    measured_signal = np.sin(2 * np.pi * 1e5 * t) + 0.5 * np.sin(2 * np.pi * 3e5 * t)

    np.random.seed(54321)
    freq_response_real = np.random.randn(n) * 0.1 + 1.0
    freq_response_imag = np.random.randn(n) * 0.05
    frequency_response = freq_response_real + 1j * freq_response_imag

    # Signal uncertainty (deterministic, based on measured signal)
    signal_uncertainty = np.abs(measured_signal) * 0.01 + 1e-6

    out_dir = Path(__file__).parent
    np.savetxt(out_dir / "measured_signal.csv", measured_signal, fmt="%.18e")
    np.savetxt(out_dir / "freq_response_real.csv", freq_response_real, fmt="%.18e")
    np.savetxt(out_dir / "freq_response_imag.csv", freq_response_imag, fmt="%.18e")
    np.savetxt(out_dir / "signal_uncertainty.csv", signal_uncertainty, fmt="%.18e")

    print(f"Generated test data: {n} samples, sampling rate {sampling_rate} Hz")
    print(f"Output directory: {out_dir}")
    return measured_signal, frequency_response, sampling_rate


if __name__ == "__main__":
    generate_test_data()
