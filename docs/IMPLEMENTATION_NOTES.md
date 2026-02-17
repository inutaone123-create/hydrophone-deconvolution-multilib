# Implementation Notes

## FFT Normalization Convention

All implementations follow the same normalization:
- **Forward FFT**: No scaling (unnormalized)
- **Inverse FFT**: 1/N scaling

### Per-Language Details

| Language | FFT Library | Forward | Inverse |
|----------|-------------|---------|---------|
| Python   | NumPy (PocketFFT) | No scaling | 1/N |
| Octave   | Built-in fft/ifft | No scaling | 1/N |
| C++      | Eigen FFT (kissfft) | No scaling | 1/N |
| C#       | MathNet.Numerics | NoScaling option | Manual 1/N |
| Rust     | rustfft | No scaling | Manual 1/N |

### C# Special Note
MathNet.Numerics `Fourier.Forward/Inverse` with `FourierOptions.NoScaling` performs unnormalized transforms in both directions. The 1/N scaling is applied manually in the code.

### Rust Special Note
rustfft performs unnormalized transforms in both directions. The 1/N scaling is applied manually after the inverse transform.

### C++ Special Note
Eigen FFT's `fwd/inv` with complex-to-complex transform is used to ensure full-spectrum compatibility. Real-to-complex may produce half-spectrum results.

## Numerical Precision

- All implementations use 64-bit floating point (double precision)
- Regularization epsilon: 1e-12 (added to frequency response before division)
- Cross-language relative error: < 1e-15 (well below 1e-14 threshold)

## Monte Carlo Uncertainty

- Uses Gaussian perturbation of both signal and frequency response
- Response uncertainty applied as complex perturbation: `u * N(0,1) * (1 + i)`
- Results: mean and standard deviation across all Monte Carlo iterations

## Pulse Parameters (pc, pr, ppsi)

Based on Weber & Wilkens (2023), the following pulse parameters are computed from the deconvolved pressure waveform:

### Definitions
- **pc (compressional peak pressure)**: Maximum value of the pressure waveform
- **pr (rarefactional peak pressure)**: Absolute value of the minimum pressure (positive by convention)
- **ppsi (pulse pressure-squared integral)**: `sum(p²) * dt`, a measure of pulse energy

### Uncertainty Propagation (Analytical)
Uses the law of propagation of uncertainty with sensitivity coefficients:
- `u(pc) = sqrt(U_p[i_max, i_max])` where `i_max = argmax(p)`
- `u(pr) = sqrt(U_p[i_min, i_min])` where `i_min = argmin(p)`
- `u(ppsi) = sqrt(C · U_p · Cᵀ)` where `C = 2 * |p| * dt` (sensitivity vector)

The input uncertainty `u_pressure` can be:
1. **Scalar** → uniform uncertainty, expanded to diagonal covariance matrix
2. **Vector** → per-sample uncertainty, expanded to diagonal covariance matrix
3. **Matrix** → full covariance matrix used directly

### Cross-Language Consistency
All 5 implementations (Python, Octave, C++, C#, Rust) produce identical results with relative errors below 1e-15.
