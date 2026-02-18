# Hydrophone Deconvolution - Multi-language Implementation
#
# Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
# DOI: 10.5281/zenodo.10079801
# License: CC BY 4.0

import os
import subprocess
import sys

import numpy as np
from behave import given, when, then

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(WORKSPACE, "reference", "original-data")
REF_DIR = os.path.join(WORKSPACE, "reference", "results")


def load_csv(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            data.append([float(x) for x in line.strip().split(";")])
    return np.array(data)


def run_pipeline(lang, sig, noise, cal, bode, filt, fc, out):
    b = "true" if bode else "false"
    if lang == "python":
        cmd = [sys.executable, os.path.join(WORKSPACE, "python", "export_pipeline_result.py"),
               sig, noise, cal, b, filt, str(fc), out]
    elif lang == "octave":
        cmd = ["octave", "--no-gui", "--path", os.path.join(WORKSPACE, "octave"),
               os.path.join(WORKSPACE, "octave", "export_pipeline_result.m"),
               sig, noise, cal, b, filt, str(fc), out]
    elif lang == "cpp":
        cmd = [os.path.join(WORKSPACE, "cpp", "build", "export_pipeline_result"),
               sig, noise, cal, b, filt, str(fc), out]
    elif lang == "csharp":
        cmd = ["dotnet", "run", "--project",
               os.path.join(WORKSPACE, "csharp", "ExportPipelineResult", "ExportPipelineResult.csproj"),
               "-c", "Release", "--", sig, noise, cal, b, filt, str(fc), out]
    elif lang == "rust":
        cmd = [os.path.join(WORKSPACE, "rust", "target", "release", "export_pipeline_result"),
               sig, noise, cal, b, filt, str(fc), out]
    else:
        raise ValueError(f"Unknown language: {lang}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"{lang} pipeline failed: {result.stderr[:500]}"


@given("the MH44 M-Mode 3MHz measurement data")
def step_given_mh44_data(context):
    context.sig = os.path.join(DATA_DIR, "MeasuredSignals", "M-Mode 3 MHz", "M3_MH44.DAT")
    context.noise = os.path.join(DATA_DIR, "MeasuredSignals", "M-Mode 3 MHz", "M3_MH44r.DAT")
    context.cal = os.path.join(DATA_DIR, "HydrophoneCalibrationData", "MW_MH44ReIm.csv")
    context.ref_path = os.path.join(REF_DIR, "M-Mode3MHz_GAMPTMH44_Bode_LowPass100MHz.csv")
    context.ref_data = load_csv(context.ref_path)


@when("I run the {lang} pipeline with LowPass filter and Bode=true")
def step_run_pipeline(context, lang):
    lang_key = lang.lower().replace("c++", "cpp").replace("c#", "csharp")
    tmp_dir = os.path.join(WORKSPACE, "validation", "bdd-results")
    os.makedirs(tmp_dir, exist_ok=True)
    out = os.path.join(tmp_dir, f"bdd_{lang_key}_p01.csv")

    if not os.path.exists(out):
        run_pipeline(lang_key, context.sig, context.noise, context.cal,
                     True, "LowPass", 100e6, out)

    context.test_data = load_csv(out)
    context.lang = lang_key


@then("the regularized waveform should match the reference within {threshold}")
def step_check_regularized(context, threshold):
    thr = float(threshold)
    ref = context.ref_data[:, 3]
    test = context.test_data[:, 3]
    mask = np.abs(ref) > 1e-30
    rel_err = float(np.max(np.abs((test[mask] - ref[mask]) / ref[mask])))
    assert rel_err < thr, f"regularized rel_err={rel_err:.2e} > {thr:.0e}"


@then("the uncertainty should match the reference within {threshold}")
def step_check_uncertainty(context, threshold):
    thr = float(threshold)
    ref = context.ref_data[:, 4]
    test = context.test_data[:, 4]
    assert not np.any(np.isnan(test)), "uncertainty contains NaN"
    mask = np.abs(ref) > 1e-30
    rel_err = float(np.max(np.abs((test[mask] - ref[mask]) / ref[mask])))
    assert rel_err < thr, f"uncertainty rel_err={rel_err:.2e} > {thr:.0e}"


@given("pipeline results from all 5 languages for MH44 M-Mode 3MHz LowPass Bode=true")
def step_given_all_results(context):
    context.sig = os.path.join(DATA_DIR, "MeasuredSignals", "M-Mode 3 MHz", "M3_MH44.DAT")
    context.noise = os.path.join(DATA_DIR, "MeasuredSignals", "M-Mode 3 MHz", "M3_MH44r.DAT")
    context.cal = os.path.join(DATA_DIR, "HydrophoneCalibrationData", "MW_MH44ReIm.csv")

    tmp_dir = os.path.join(WORKSPACE, "validation", "bdd-results")
    os.makedirs(tmp_dir, exist_ok=True)

    context.results = {}
    for lang in ["python", "octave", "cpp", "csharp", "rust"]:
        out = os.path.join(tmp_dir, f"bdd_{lang}_p01.csv")
        if not os.path.exists(out):
            run_pipeline(lang, context.sig, context.noise, context.cal,
                         True, "LowPass", 100e6, out)
        context.results[lang] = load_csv(out)


@when("I compare all language pairs")
def step_compare_pairs(context):
    langs = list(context.results.keys())
    context.max_reg_err = 0.0
    context.max_unc_err = 0.0
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            d1 = context.results[langs[i]]
            d2 = context.results[langs[j]]
            mask = np.abs(d1[:, 3]) > 1e-30
            reg_err = float(np.max(np.abs((d2[:, 3][mask] - d1[:, 3][mask]) / d1[:, 3][mask])))
            mask2 = np.abs(d1[:, 4]) > 1e-30
            unc_err = float(np.max(np.abs((d2[:, 4][mask2] - d1[:, 4][mask2]) / d1[:, 4][mask2])))
            context.max_reg_err = max(context.max_reg_err, reg_err)
            context.max_unc_err = max(context.max_unc_err, unc_err)


@then("the maximum regularized relative error should be less than {threshold}")
def step_check_max_reg(context, threshold):
    thr = float(threshold)
    assert context.max_reg_err < thr, f"max reg err={context.max_reg_err:.2e} > {thr:.0e}"


@then("the maximum uncertainty relative error should be less than {threshold}")
def step_check_max_unc(context, threshold):
    thr = float(threshold)
    assert context.max_unc_err < thr, f"max unc err={context.max_unc_err:.2e} > {thr:.0e}"
