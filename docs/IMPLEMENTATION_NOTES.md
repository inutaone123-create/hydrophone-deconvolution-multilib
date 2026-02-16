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
