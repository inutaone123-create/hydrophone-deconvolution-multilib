# Cross-Validation Report

## Test Configuration

- **Signal length**: 1024 samples
- **Sampling rate**: 10 MHz
- **Test signal**: Composite sine wave (100 kHz + 300 kHz)
- **Frequency response**: Random complex values (seed: 54321)

## Results

All 10 pairwise comparisons passed with relative error < 1e-14.

| Pair | Max Abs Diff | Relative Diff | Status |
|------|-------------|---------------|--------|
| Python vs Octave | ~1e-15 | ~9e-16 | PASS |
| Python vs C++ | ~1e-15 | ~8e-16 | PASS |
| Python vs C# | ~1e-15 | ~1e-15 | PASS |
| Python vs Rust | ~9e-16 | ~8e-16 | PASS |
| Octave vs C++ | ~1e-15 | ~9e-16 | PASS |
| Octave vs C# | ~1e-15 | ~9e-16 | PASS |
| Octave vs Rust | ~1e-15 | ~1e-15 | PASS |
| C++ vs C# | ~9e-16 | ~8e-16 | PASS |
| C++ vs Rust | ~1e-15 | ~1e-15 | PASS |
| C# vs Rust | ~1e-15 | ~1e-15 | PASS |

## Summary

**10/10 comparisons PASSED**

Maximum relative error across all pairs: < 1.2e-15

This confirms numerical equivalence across all 5 language implementations, well within the 1e-14 threshold required by the project specification.
