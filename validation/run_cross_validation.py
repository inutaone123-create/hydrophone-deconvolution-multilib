#!/usr/bin/env python3
# Hydrophone Deconvolution - Multi-language Implementation
#
# Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
# DOI: 10.5281/zenodo.10079801
# License: CC BY 4.0
#
# Cross-validation: run M-Mode patterns across all 5 languages

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

with open(os.path.join(REFERENCE_DIR, "patterns.json")) as f:
    config = json.load(f)

PATTERNS = config["patterns"]
FC_MAP = config["fc_map"]


def ref_filename(pattern, filter_type, bode):
    ptype = pattern["type"]
    hyd = pattern["hyd"]
    fc_val = FC_MAP[ptype]
    fc_mhz = f"{int(fc_val / 1e6)}MHz"
    bode_str = "Bode_" if bode else ""
    if filter_type == "None":
        return f"{ptype}_{hyd}_{bode_str}None.csv"
    return f"{ptype}_{hyd}_{bode_str}{filter_type}{fc_mhz}.csv"


def load_csv(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            data.append([float(x) for x in line.strip().split(";")])
    return np.array(data)


def get_cmd(lang, sig, noise, cal, bode, filt, fc, out):
    b = "true" if bode else "false"
    if lang == "python":
        return [sys.executable, os.path.join(WORKSPACE, "python", "export_pipeline_result.py"),
                sig, noise, cal, b, filt, str(fc), out]
    elif lang == "octave":
        return ["octave", "--no-gui", "--path", os.path.join(WORKSPACE, "octave"),
                os.path.join(WORKSPACE, "octave", "export_pipeline_result.m"),
                sig, noise, cal, b, filt, str(fc), out]
    elif lang == "cpp":
        return [os.path.join(WORKSPACE, "cpp", "build", "export_pipeline_result"),
                sig, noise, cal, b, filt, str(fc), out]
    elif lang == "csharp":
        return ["dotnet", "run", "--project",
                os.path.join(WORKSPACE, "csharp", "ExportPipelineResult", "ExportPipelineResult.csproj"),
                "-c", "Release", "--", sig, noise, cal, b, filt, str(fc), out]
    elif lang == "rust":
        return [os.path.join(WORKSPACE, "rust", "target", "release", "export_pipeline_result"),
                sig, noise, cal, b, filt, str(fc), out]


def main():
    # M-Mode patterns only (N=2500, manageable memory)
    m_mode_patterns = [p for p in PATTERNS if "M-Mode" in p["type"]]
    filter_types = ["LowPass", "CriticalDamping", "Bessel", "None"]
    bode_options = [True, False]
    languages = ["python", "octave", "cpp", "csharp", "rust"]

    total_tests = 0
    passed = 0
    failed_list = []

    # Per-language max errors
    lang_max_reg = {l: 0.0 for l in languages}
    lang_max_unc = {l: 0.0 for l in languages}

    n_combos = len(m_mode_patterns) * len(filter_types) * len(bode_options)
    combo_idx = 0

    for pattern in m_mode_patterns:
        fc = FC_MAP[pattern["type"]]
        sig = os.path.join(DATA_DIR, pattern["signal"])
        noise = os.path.join(DATA_DIR, pattern["noise"])
        cal = os.path.join(DATA_DIR, pattern["cal"])

        for filt in filter_types:
            for bode in bode_options:
                combo_idx += 1
                bode_str = "bode" if bode else "nobode"
                label = f"p{pattern['id']:02d}_{filt}_{bode_str}"

                ref_name = ref_filename(pattern, filt, bode)
                ref_path = os.path.join(RESULTS_DIR, ref_name)
                if not os.path.exists(ref_path):
                    print(f"[{combo_idx}/{n_combos}] SKIP {label}: ref not found")
                    continue

                ref_data = load_csv(ref_path)
                print(f"[{combo_idx}/{n_combos}] {label}", end="", flush=True)

                for lang in languages:
                    total_tests += 1
                    out_dir = os.path.join(OUTPUT_DIR, lang)
                    os.makedirs(out_dir, exist_ok=True)
                    out = os.path.join(out_dir, f"{label}.csv")

                    if not os.path.exists(out):
                        cmd = get_cmd(lang, sig, noise, cal, bode, filt, fc, out)
                        try:
                            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                            if r.returncode != 0:
                                print(f" {lang}:FAIL", end="")
                                failed_list.append(f"{label}/{lang}: exit {r.returncode}")
                                continue
                        except subprocess.TimeoutExpired:
                            print(f" {lang}:TIMEOUT", end="")
                            failed_list.append(f"{label}/{lang}: timeout")
                            continue
                        except FileNotFoundError:
                            print(f" {lang}:NOTFOUND", end="")
                            failed_list.append(f"{label}/{lang}: not found")
                            continue

                    test_data = load_csv(out)
                    mask = np.abs(ref_data[:, 3]) > 1e-30
                    reg_err = float(np.max(np.abs((test_data[:, 3][mask] - ref_data[:, 3][mask]) / ref_data[:, 3][mask])))

                    mask2 = np.abs(ref_data[:, 4]) > 1e-30
                    has_nan = np.any(np.isnan(test_data[:, 4]))
                    unc_err = float("nan") if has_nan else float(
                        np.max(np.abs((test_data[:, 4][mask2] - ref_data[:, 4][mask2]) / ref_data[:, 4][mask2])))

                    if reg_err < 1e-6 and (not np.isnan(unc_err) and unc_err < 1e-6):
                        passed += 1
                        lang_max_reg[lang] = max(lang_max_reg[lang], reg_err)
                        lang_max_unc[lang] = max(lang_max_unc[lang], unc_err)
                    else:
                        failed_list.append(f"{label}/{lang}: reg={reg_err:.2e} unc={unc_err:.2e}")

                print()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total_tests} PASS")
    if failed_list:
        print(f"Failed ({len(failed_list)}):")
        for f in failed_list:
            print(f"  {f}")
    print("\nPer-language max relative errors:")
    for lang in languages:
        print(f"  {lang:8s}: reg={lang_max_reg[lang]:.2e}  unc={lang_max_unc[lang]:.2e}")
    print("=" * 60)
    return 0 if not failed_list else 1


if __name__ == "__main__":
    sys.exit(main())
