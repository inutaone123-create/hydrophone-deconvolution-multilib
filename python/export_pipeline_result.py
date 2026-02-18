#!/usr/bin/env python3
"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

This implementation: 2024
License: CC BY 4.0
https://creativecommons.org/licenses/by/4.0/

Export pipeline results for all 128 patterns (16 data × 4 filters × 2 Bode).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deconvolution.pipeline import full_pipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference", "original-data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation", "pipeline_results", "python")

PATTERNS = [
    (1, "MeasuredSignals/M-Mode 3 MHz/M3_MH44.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "M-Mode3MHz", True),
    (2, "MeasuredSignals/M-Mode 3 MHz/M3_MH46.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "M-Mode3MHz", True),
    (3, "MeasuredSignals/M-Mode 3 MHz/M3_ON1704.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "M-Mode3MHz", False),
    (4, "MeasuredSignals/M-Mode 3 MHz/M3_PA1434.DAT", "MeasuredSignals/M-Mode 3 MHz/M3_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "M-Mode3MHz", True),
    (5, "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "Pulse-Doppler-Mode3MHz", True),
    (6, "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "Pulse-Doppler-Mode3MHz", True),
    (7, "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "Pulse-Doppler-Mode3MHz", False),
    (8, "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434.DAT", "MeasuredSignals/pD-Mode 3 MHz/pD3_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "Pulse-Doppler-Mode3MHz", True),
    (9, "MeasuredSignals/M-Mode 6 MHz/M6_MH44.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "M-Mode6MHz", True),
    (10, "MeasuredSignals/M-Mode 6 MHz/M6_MH46.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "M-Mode6MHz", True),
    (11, "MeasuredSignals/M-Mode 6 MHz/M6_ON1704.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "M-Mode6MHz", False),
    (12, "MeasuredSignals/M-Mode 6 MHz/M6_PA1434.DAT", "MeasuredSignals/M-Mode 6 MHz/M6_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "M-Mode6MHz", True),
    (13, "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_MH44r.DAT",
     "HydrophoneCalibrationData/MW_MH44ReIm.csv", "GAMPTMH44", "Pulse-Doppler-Mode7MHz", True),
    (14, "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_MH46r.DAT",
     "HydrophoneCalibrationData/MH46_MWReIm.csv", "GAMPTMH46", "Pulse-Doppler-Mode7MHz", True),
    (15, "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_ON1704r.DAT",
     "HydrophoneCalibrationData/MW_ONDA1704_SECReIm.csv", "ONDA1704", "Pulse-Doppler-Mode7MHz", False),
    (16, "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434.DAT", "MeasuredSignals/pD-Mode 7 MHz/pD7_PA1434r.DAT",
     "HydrophoneCalibrationData/MW_PA1434ReIm.csv", "PrecisionAcoustics1434", "Pulse-Doppler-Mode7MHz", True),
]

FC_MAP = {
    "M-Mode3MHz": 100e6,
    "Pulse-Doppler-Mode3MHz": 120e6,
    "M-Mode6MHz": 150e6,
    "Pulse-Doppler-Mode7MHz": 200e6,
}

FILTER_TYPES = ["LowPass", "CriticalDamping", "Bessel", "None"]
BODE_OPTIONS = [True, False]


def result_filename(meas_type, hyd_name, usebode, filter_type, fc):
    bode_str = "_Bode" if usebode else ""
    fc_str = f"{fc / 1e6:.0f}MHz" if filter_type != "None" else ""
    return f"{meas_type}_{hyd_name}{bode_str}_{filter_type}{fc_str}.csv"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(PATTERNS) * len(FILTER_TYPES) * len(BODE_OPTIONS)
    count = 0
    ok = 0

    for pat in PATTERNS:
        idx, meas_file, noise_file, cal_file, hyd_name, meas_type, _ = pat
        fc = FC_MAP[meas_type]

        for filter_type in FILTER_TYPES:
            for usebode in BODE_OPTIONS:
                count += 1
                fname = result_filename(meas_type, hyd_name, usebode, filter_type, fc)
                outpath = os.path.join(OUTPUT_DIR, fname)

                if os.path.exists(outpath):
                    print(f"[{count:3d}/{total}] SKIP {fname}")
                    ok += 1
                    continue

                print(f"[{count:3d}/{total}] {fname} ...", end=" ", flush=True)
                try:
                    result = full_pipeline(
                        os.path.join(DATA_DIR, meas_file),
                        os.path.join(DATA_DIR, noise_file),
                        os.path.join(DATA_DIR, cal_file),
                        usebode=usebode, filter_type=filter_type, fc=fc
                    )
                    header = "time;scaled;deconvolved;regularized;uncertainty(k=1)"
                    data = np.column_stack([
                        result["time"], result["scaled"], result["deconvolved"],
                        result["regularized"], result["uncertainty"],
                    ])
                    pp = result["pulse_params"]
                    footer = (f"pc_value={pp['pc_value']};pc_uncertainty={pp['pc_uncertainty']};"
                              f"pc_time={pp['pc_time']};pr_value={pp['pr_value']};"
                              f"pr_uncertainty={pp['pr_uncertainty']};pr_time={pp['pr_time']};"
                              f"ppsi_value={pp['ppsi_value']};ppsi_uncertainty={pp['ppsi_uncertainty']}")
                    np.savetxt(outpath, data, header=header, delimiter=";", footer=footer)
                    print("OK")
                    ok += 1
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\n{ok}/{total} completed")


if __name__ == "__main__":
    main()
