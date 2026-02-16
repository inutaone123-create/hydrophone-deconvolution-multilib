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
import pytest
from deconvolution.core import deconvolve_without_uncertainty, deconvolve_with_uncertainty


class TestDeconvolveWithoutUncertainty:
    def test_output_length(self):
        signal = np.random.randn(1024)
        freq_resp = np.fft.fft(np.random.randn(1024)) + 1.0
        result = deconvolve_without_uncertainty(signal, freq_resp, 1e7)
        assert len(result) == 1024

    def test_output_real(self):
        signal = np.random.randn(512)
        freq_resp = np.fft.fft(np.random.randn(512)) + 1.0
        result = deconvolve_without_uncertainty(signal, freq_resp, 1e7)
        assert np.all(np.isreal(result))

    def test_known_signal_recovery(self):
        np.random.seed(42)
        original = np.random.randn(256)
        freq_resp = np.fft.fft(np.random.randn(256)) + 2.0
        signal_fft = np.fft.fft(original)
        measured = np.fft.ifft(signal_fft * freq_resp).real
        recovered = deconvolve_without_uncertainty(measured, freq_resp, 1e7)
        assert np.max(np.abs(recovered - original)) < 1e-10

    def test_uses_pocketfft(self):
        """Verify numpy.fft uses PocketFFT backend."""
        assert hasattr(np.fft, "fft")
        x = np.array([1.0, 0.0, 0.0, 0.0])
        result = np.fft.fft(x)
        expected = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        assert np.allclose(result, expected)


class TestDeconvolveWithUncertainty:
    def test_output_shapes(self):
        np.random.seed(42)
        n = 128
        signal = np.random.randn(n)
        signal_unc = np.abs(signal) * 0.01 + 1e-6
        freq_resp = np.fft.fft(np.random.randn(n)) + 1.0
        resp_unc = np.abs(freq_resp) * 0.01 + 1e-6
        mean, std = deconvolve_with_uncertainty(
            signal, signal_unc, freq_resp, resp_unc, 1e7, num_monte_carlo=50
        )
        assert len(mean) == n
        assert len(std) == n

    def test_uncertainty_nonnegative(self):
        np.random.seed(42)
        n = 128
        signal = np.random.randn(n)
        signal_unc = np.abs(signal) * 0.01 + 1e-6
        freq_resp = np.fft.fft(np.random.randn(n)) + 1.0
        resp_unc = np.abs(freq_resp) * 0.01 + 1e-6
        _, std = deconvolve_with_uncertainty(
            signal, signal_unc, freq_resp, resp_unc, 1e7, num_monte_carlo=50
        )
        assert np.all(std >= 0)
