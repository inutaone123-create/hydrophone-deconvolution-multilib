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
from deconvolution.core import pulse_parameters


@given("a time array and pressure signal of length {n:d}")
def step_given_time_pressure(context, n):
    np.random.seed(42)
    context.sampling_rate = 1e7
    context.time = np.arange(n) / context.sampling_rate
    context.pressure = np.sin(2 * np.pi * 1e5 * context.time) + 0.3 * np.random.randn(n)
    context.n = n


@given("a scalar uncertainty of {u}")
def step_given_scalar_uncertainty(context, u):
    context.u_pressure = float(u)


@given("a vector uncertainty of length {n:d}")
def step_given_vector_uncertainty(context, n):
    context.u_pressure = np.abs(context.pressure) * 0.01 + 1e-6


@given("a uniform scalar uncertainty of {u}")
def step_given_uniform_scalar(context, u):
    context.u_scalar = float(u)


@when("I calculate pulse parameters")
def step_calc_pulse_params(context):
    context.pp = pulse_parameters(context.time, context.pressure, context.u_pressure)


@when("I calculate pulse parameters with scalar and vector inputs")
def step_calc_both(context):
    u = context.u_scalar
    context.pp_scalar = pulse_parameters(context.time, context.pressure, u)
    u_vec = np.full(context.n, u)
    context.pp_vector = pulse_parameters(context.time, context.pressure, u_vec)


@then("I should get valid pc, pr, and ppsi values")
def step_check_valid(context):
    pp = context.pp
    assert pp["pc_value"] > 0, f"pc_value should be positive, got {pp['pc_value']}"
    assert pp["pr_value"] > 0, f"pr_value should be positive, got {pp['pr_value']}"
    assert pp["ppsi_value"] > 0, f"ppsi_value should be positive, got {pp['ppsi_value']}"
    assert 0 <= pp["pc_index"] < context.n
    assert 0 <= pp["pr_index"] < context.n


@then("all uncertainties should be positive")
def step_check_positive_unc(context):
    pp = context.pp
    assert pp["pc_uncertainty"] > 0
    assert pp["pr_uncertainty"] > 0
    assert pp["ppsi_uncertainty"] > 0


@then("the results should be identical")
def step_check_identical(context):
    for key in ["pc_value", "pc_uncertainty", "pr_value", "pr_uncertainty",
                "ppsi_value", "ppsi_uncertainty"]:
        v1 = context.pp_scalar[key]
        v2 = context.pp_vector[key]
        assert np.isclose(v1, v2, rtol=1e-14), \
            f"{key}: scalar={v1}, vector={v2}, diff={abs(v1-v2)}"
