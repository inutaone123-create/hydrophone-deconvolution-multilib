"""
Cross-language validation script.

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


def main():
    languages = ["python", "octave", "cpp", "csharp", "rust"]
    results = {}

    results_dir = Path("validation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        filepath = results_dir / f"{lang}_result.csv"
        if filepath.exists():
            results[lang] = np.loadtxt(filepath)
            print(f"  Loaded {lang}: {len(results[lang])} samples")
        else:
            print(f"  Missing {lang} results")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60 + "\n")

    passed = 0
    total = 0

    for i, lang1 in enumerate(languages):
        if lang1 not in results:
            continue
        for lang2 in languages[i + 1 :]:
            if lang2 not in results:
                continue

            total += 1
            max_diff = np.max(np.abs(results[lang1] - results[lang2]))
            ref_val = np.max(np.abs(results[lang1]))
            rel_diff = max_diff / ref_val if ref_val > 0 else 0.0

            print(f"{lang1.upper()} vs {lang2.upper()}:")
            print(f"  Max abs diff: {max_diff:.2e}")
            print(f"  Relative diff: {rel_diff:.2e}")

            if rel_diff < 1e-14:
                passed += 1
                print("  PASS\n")
            else:
                print("  FAIL\n")

    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} comparisons passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
