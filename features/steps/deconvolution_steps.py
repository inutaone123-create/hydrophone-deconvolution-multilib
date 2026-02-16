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
from behave import given, when, then

import sys
sys.path.insert(0, "python")
from deconvolution.core import deconvolve_without_uncertainty, deconvolve_with_uncertainty


@given("a measured signal of length {n:d}")
def step_given_measured_signal(context, n):
    np.random.seed(42)
    context.measured_signal = np.random.randn(n)
    context.n = n


@given("a frequency response of length {n:d}")
def step_given_frequency_response(context, n):
    np.random.seed(123)
    context.frequency_response = np.fft.fft(np.random.randn(n)) + 1.0


@given("a sampling rate of {rate:d} Hz")
def step_given_sampling_rate(context, rate):
    context.sampling_rate = float(rate)


@given("signal uncertainty values")
def step_given_signal_uncertainty(context):
    context.signal_uncertainty = np.abs(context.measured_signal) * 0.01 + 1e-6


@given("response uncertainty values")
def step_given_response_uncertainty(context):
    context.response_uncertainty = np.abs(context.frequency_response) * 0.01 + 1e-6


@given("a known input signal")
def step_given_known_input(context):
    np.random.seed(42)
    context.original_signal = np.random.randn(256)
    context.sampling_rate = 1e7


@given("a known frequency response")
def step_given_known_freq_response(context):
    np.random.seed(99)
    context.frequency_response = np.fft.fft(np.random.randn(256)) + 2.0
    signal_fft = np.fft.fft(context.original_signal)
    convolved_fft = signal_fft * context.frequency_response
    context.measured_signal = np.fft.ifft(convolved_fft).real


@when("I perform deconvolution without uncertainty")
def step_deconvolve_no_unc(context):
    context.result = deconvolve_without_uncertainty(
        context.measured_signal, context.frequency_response, context.sampling_rate
    )


@when("I convolve and then deconvolve")
def step_convolve_deconvolve(context):
    context.result = deconvolve_without_uncertainty(
        context.measured_signal, context.frequency_response, context.sampling_rate
    )


@when("I perform deconvolution with {n:d} Monte Carlo samples")
def step_deconvolve_with_unc(context, n):
    context.mean_result, context.uncertainty = deconvolve_with_uncertainty(
        context.measured_signal,
        context.signal_uncertainty,
        context.frequency_response,
        context.response_uncertainty,
        context.sampling_rate,
        num_monte_carlo=n,
    )


@then("the result should have length {n:d}")
def step_check_length(context, n):
    assert len(context.result) == n, f"Expected {n}, got {len(context.result)}"


@then("the result should be real-valued")
def step_check_real(context):
    assert np.all(np.isreal(context.result)), "Result contains imaginary parts"


@then("the recovered signal should match the original within {tol}")
def step_check_recovery(context, tol):
    tol = float(tol)
    max_err = np.max(np.abs(context.result - context.original_signal))
    assert max_err < tol, f"Max error {max_err} exceeds tolerance {tol}"


@then("I should get a mean deconvolved signal of length {n:d}")
def step_check_mean_length(context, n):
    assert len(context.mean_result) == n


@then("I should get uncertainty values of length {n:d}")
def step_check_unc_length(context, n):
    assert len(context.uncertainty) == n


@then("all uncertainty values should be non-negative")
def step_check_unc_nonneg(context):
    assert np.all(context.uncertainty >= 0), "Negative uncertainty found"
