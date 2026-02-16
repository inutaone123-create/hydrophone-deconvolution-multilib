# API Reference

## Common API (All Languages)

### deconvolve_without_uncertainty

Deconvolve a hydrophone measurement signal using frequency-domain division.

**Parameters:**
- `measured_signal` - Time-domain measurement signal (array of float64)
- `frequency_response` - Complex frequency response of the hydrophone (array of complex128)
- `sampling_rate` - Sampling rate in Hz (float64)

**Returns:**
- `deconvolved_signal` - Deconvolved time-domain signal (array of float64)

**Algorithm:**
1. FFT of measured signal
2. Element-wise division by frequency response (with epsilon=1e-12 regularization)
3. IFFT to return to time domain

### deconvolve_with_uncertainty

Deconvolve with GUM-compliant Monte Carlo uncertainty propagation.

**Parameters:**
- `measured_signal` - Time-domain measurement signal
- `signal_uncertainty` - Standard uncertainty of each signal sample
- `frequency_response` - Complex frequency response
- `response_uncertainty` - Standard uncertainty of frequency response magnitude
- `sampling_rate` - Sampling rate in Hz
- `num_monte_carlo` - Number of Monte Carlo iterations (default: 1000)

**Returns:**
- `mean_signal` - Mean deconvolved signal
- `uncertainty` - Standard deviation (uncertainty) at each sample

## Language-Specific Notes

### Python
```python
from deconvolution import deconvolve_without_uncertainty, deconvolve_with_uncertainty
```
Uses NumPy's PocketFFT backend.

### Octave
```matlab
result = deconvolution.deconvolve_without_uncertainty(signal, freq_resp, fs);
```

### C++
```cpp
#include "deconvolution.hpp"
auto result = hydrophone::deconvolve_without_uncertainty(signal, freq_resp, fs);
```
Uses Eigen FFT (kissfft backend).

### C#
```csharp
using HydrophoneDeconvolution;
var result = Deconvolution.DeconvolveWithoutUncertainty(signal, freqResp, fs);
```
Uses MathNet.Numerics Fourier transforms.

### Rust
```rust
use hydrophone_deconvolution::deconvolve_without_uncertainty;
let result = deconvolve_without_uncertainty(&signal, &freq_resp, fs);
```
Uses rustfft with manual 1/N normalization.
