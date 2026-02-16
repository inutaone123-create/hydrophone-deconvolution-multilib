"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
License: CC BY 4.0
"""

import numpy as np
from pathlib import Path
from deconvolution.core import deconvolve_without_uncertainty


def main():
    data_dir = Path(__file__).parent.parent / "test-data"
    results_dir = Path(__file__).parent.parent / "validation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    measured_signal = np.loadtxt(data_dir / "measured_signal.csv")
    freq_real = np.loadtxt(data_dir / "freq_response_real.csv")
    freq_imag = np.loadtxt(data_dir / "freq_response_imag.csv")
    frequency_response = freq_real + 1j * freq_imag

    result = deconvolve_without_uncertainty(measured_signal, frequency_response, 1e7)
    np.savetxt(results_dir / "python_result.csv", result, fmt="%.18e")
    print(f"Python result exported: {len(result)} samples")


if __name__ == "__main__":
    main()
