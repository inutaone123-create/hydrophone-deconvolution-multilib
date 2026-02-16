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
from pathlib import Path


@given("the standard test signal")
def step_standard_test_signal(context):
    results_dir = Path("validation/results")
    context.results = {}
    for lang in ["python", "octave", "cpp", "csharp", "rust"]:
        filepath = results_dir / f"{lang}_result.csv"
        if filepath.exists():
            context.results[lang] = np.loadtxt(filepath)


@given("results from all 5 languages")
def step_all_results(context):
    step_standard_test_signal(context)


@when("I deconvolve with {lang}")
def step_deconvolve_lang(context, lang):
    lang_key = lang.lower().replace("c++", "cpp").replace("c#", "csharp")
    results_dir = Path("validation/results")
    filepath = results_dir / f"{lang_key}_result.csv"
    if filepath.exists():
        context.results[lang_key] = np.loadtxt(filepath)


@when("I compare all 10 pairs")
def step_compare_all_pairs(context):
    languages = list(context.results.keys())
    context.all_pass = True
    context.max_rel_error = 0.0
    for i, lang1 in enumerate(languages):
        for lang2 in languages[i + 1 :]:
            ref = np.max(np.abs(context.results[lang1]))
            if ref > 0:
                rel = np.max(np.abs(context.results[lang1] - context.results[lang2])) / ref
            else:
                rel = 0.0
            context.max_rel_error = max(context.max_rel_error, rel)
            if rel >= 1e-14:
                context.all_pass = False


@then("the relative error should be less than {tol}")
def step_check_rel_error(context, tol):
    tol = float(tol)
    langs = list(context.results.keys())
    if len(langs) >= 2:
        ref = np.max(np.abs(context.results[langs[0]]))
        if ref > 0:
            rel = np.max(np.abs(context.results[langs[0]] - context.results[langs[1]])) / ref
        else:
            rel = 0.0
        assert rel < tol, f"Relative error {rel} >= {tol}"


@then("all relative errors should be less than {tol}")
def step_check_all_rel_errors(context, tol):
    tol = float(tol)
    assert context.all_pass, f"Max relative error {context.max_rel_error} >= {tol}"
