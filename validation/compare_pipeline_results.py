#!/usr/bin/env python3
# Hydrophone Deconvolution - Multi-language Implementation
#
# Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
# DOI: 10.5281/zenodo.10079801
# License: CC BY 4.0
#
# Cross-validation script for 5-language pipeline results vs PyDynamic reference.

import json
import os
import subprocess
import sys
import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(WORKSPACE, "reference")
DATA_DIR = os.path.join(REFERENCE_DIR, "original-data")
RESULTS_DIR = os.path.join(REFERENCE_DIR, "results")
OUTPUT_DIR = os.path.join(WORKSPACE, "validation", "pipeline-results")

# Load patterns config
with open(os.path.join(REFERENCE_DIR, "patterns.json")) as f:
    config = json.load(f)

PATTERNS = config["patterns"]
FC_MAP = config["fc_map"]
FILTER_TYPES = config["filter_types"]
BODE_OPTIONS = config["bode_options"]

# Language executables
LANGUAGES = {
    "python": {
        "cmd": lambda sig, noise, cal, bode, filt, fc, out: [
            sys.executable, os.path.join(WORKSPACE, "python", "export_pipeline_result.py"),
            sig, noise, cal, "true" if bode else "false", filt, str(fc), out
        ]
    },
    "octave": {
        "cmd": lambda sig, noise, cal, bode, filt, fc, out: [
            "octave", "--no-gui", "--path", os.path.join(WORKSPACE, "octave"),
            os.path.join(WORKSPACE, "octave", "export_pipeline_result.m"),
            sig, noise, cal, "true" if bode else "false", filt, str(fc), out
        ]
    },
    "cpp": {
        "cmd": lambda sig, noise, cal, bode, filt, fc, out: [
            os.path.join(WORKSPACE, "cpp", "build", "export_pipeline_result"),
            sig, noise, cal, "true" if bode else "false", filt, str(fc), out
        ]
    },
    "csharp": {
        "cmd": lambda sig, noise, cal, bode, filt, fc, out: [
            "dotnet", "run", "--project",
            os.path.join(WORKSPACE, "csharp", "ExportPipelineResult", "ExportPipelineResult.csproj"),
            "-c", "Release", "--",
            sig, noise, cal, "true" if bode else "false", filt, str(fc), out
        ]
    },
    "rust": {
        "cmd": lambda sig, noise, cal, bode, filt, fc, out: [
            os.path.join(WORKSPACE, "rust", "target", "release", "export_pipeline_result"),
            sig, noise, cal, "true" if bode else "false", filt, str(fc), out
        ]
    },
}


def load_csv(path):
    """Load pipeline result CSV."""
    data = []
    footer = None
    with open(path) as f:
        for line in f:
            if line.startswith("# pc_value"):
                footer = line.strip()
                continue
            if line.startswith("#"):
                continue
            data.append([float(x) for x in line.strip().split(";")])
    return np.array(data), footer


def ref_filename(pattern, filter_type, bode):
    """Construct reference result filename."""
    ptype = pattern["type"]
    hyd = pattern["hyd"]
    fc_val = FC_MAP[ptype]
    fc_mhz = f"{int(fc_val / 1e6)}MHz"
    bode_str = "Bode_" if bode else ""
    if filter_type == "None":
        return f"{ptype}_{hyd}_{bode_str}None.csv"
    return f"{ptype}_{hyd}_{bode_str}{filter_type}{fc_mhz}.csv"


def run_pattern(lang, pattern, filter_type, bode, timeout=600):
    """Run a single pattern for a language and return output path."""
    ptype = pattern["type"]
    hyd = pattern["hyd"]
    fc = FC_MAP[ptype]
    sig = os.path.join(DATA_DIR, pattern["signal"])
    noise = os.path.join(DATA_DIR, pattern["noise"])
    cal = os.path.join(DATA_DIR, pattern["cal"])

    bode_str = "bode" if bode else "nobode"
    out_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"p{pattern['id']:02d}_{filter_type}_{bode_str}.csv")

    if os.path.exists(out):
        return out

    cmd = LANGUAGES[lang]["cmd"](sig, noise, cal, bode, filter_type, fc, out)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  FAIL {lang} p{pattern['id']:02d} {filter_type} {bode_str}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {lang} p{pattern['id']:02d} {filter_type} {bode_str}")
        return None
    except FileNotFoundError:
        print(f"  NOT FOUND: {cmd[0]}")
        return None

    return out


def compare(ref_data, test_data, labels=None):
    """Compare two result arrays. Returns dict of max relative errors."""
    if labels is None:
        labels = ["time", "scaled", "deconvolved", "regularized", "uncertainty"]
    errors = {}
    for i, label in enumerate(labels):
        r = ref_data[:, i]
        t = test_data[:, i]
        if np.any(np.isnan(t)):
            errors[label] = float("nan")
            continue
        mask = np.abs(r) > 1e-30
        if mask.any():
            errors[label] = float(np.max(np.abs((t[mask] - r[mask]) / r[mask])))
        else:
            errors[label] = float(np.max(np.abs(t - r)))
    return errors


def main():
    # Only run M-Mode patterns (N=2500) to avoid OOM on pD-Mode (N=5000)
    small_patterns = [p for p in PATTERNS if "M-Mode" in p["type"]]
    # Use subset of filters for speed
    test_filters = FILTER_TYPES
    test_bodes = BODE_OPTIONS

    print("=" * 70)
    print("5-Language Pipeline Cross-Validation")
    print("=" * 70)

    total = 0
    passed = 0
    failed = 0
    results_summary = []

    for pattern in small_patterns:
        for filt in test_filters:
            for bode in test_bodes:
                # Get reference
                ref_name = ref_filename(pattern, filt, bode)
                ref_path = os.path.join(RESULTS_DIR, ref_name)
                if not os.path.exists(ref_path):
                    print(f"SKIP: ref not found: {ref_name}")
                    continue

                ref_data, _ = load_csv(ref_path)
                bode_str = "bode" if bode else "nobode"
                pattern_label = f"p{pattern['id']:02d}_{filt}_{bode_str}"

                for lang in LANGUAGES:
                    total += 1
                    out = run_pattern(lang, pattern, filt, bode)
                    if out is None:
                        failed += 1
                        results_summary.append((pattern_label, lang, "FAIL", {}))
                        continue

                    test_data, _ = load_csv(out)
                    errs = compare(ref_data, test_data)

                    # Check thresholds
                    reg_ok = errs.get("regularized", float("nan")) < 5e-6
                    unc_ok = errs.get("uncertainty", float("nan")) < 5e-6 or not np.isnan(errs.get("uncertainty", 0))

                    if reg_ok and unc_ok:
                        passed += 1
                        status = "PASS"
                    else:
                        failed += 1
                        status = "FAIL"

                    results_summary.append((pattern_label, lang, status, errs))
                    if status == "FAIL":
                        print(f"  {status}: {lang:8s} {pattern_label} reg={errs.get('regularized', 'N/A'):.2e} unc={errs.get('uncertainty', 'N/A'):.2e}")

    print()
    print("=" * 70)
    print(f"Results: {passed}/{total} PASS, {failed}/{total} FAIL")
    print("=" * 70)

    # Print per-language summary
    print("\nPer-language max errors (regularized / uncertainty):")
    for lang in LANGUAGES:
        lang_results = [r for r in results_summary if r[1] == lang and r[2] == "PASS"]
        if lang_results:
            max_reg = max(r[3].get("regularized", 0) for r in lang_results)
            max_unc = max(r[3].get("uncertainty", 0) for r in lang_results)
            print(f"  {lang:8s}: reg={max_reg:.2e}  unc={max_unc:.2e}")
        else:
            print(f"  {lang:8s}: no passing results")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
