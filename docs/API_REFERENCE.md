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

### pulse_parameters

Calculate pulse parameters (pc, pr, ppsi) and their uncertainties using analytical uncertainty propagation.

**Parameters:**
- `time` - Time array (array of float64)
- `pressure` - Deconvolved pressure array (array of float64)
- `u_pressure` - Uncertainty: scalar, 1D vector, or 2D covariance matrix

**Returns:**
- `pc_value` - Compressional peak pressure (max of pressure)
- `pc_uncertainty` - Uncertainty of pc
- `pc_index` - Index of pc in the pressure array
- `pc_time` - Time of pc
- `pr_value` - Rarefactional peak pressure (positive value, = -min(pressure))
- `pr_uncertainty` - Uncertainty of pr
- `pr_index` - Index of pr in the pressure array
- `pr_time` - Time of pr
- `ppsi_value` - Pulse pressure-squared integral (sum(p²) * dt)
- `ppsi_uncertainty` - Uncertainty of ppsi

**Algorithm:**
1. Build covariance matrix U_p from input (scalar→diag, vector→diag, matrix→as-is)
2. pc: `argmax(pressure)`, uncertainty from `sqrt(U_p[i,i])`
3. pr: `argmin(pressure)`, uncertainty from `sqrt(U_p[i,i])`
4. ppsi: `sum(p²)*dt`, sensitivity `C = 2*|p|*dt`, uncertainty from `sqrt(C·U_p·Cᵀ)`

## Language-Specific Notes

### Python
```python
from deconvolution import deconvolve_without_uncertainty, deconvolve_with_uncertainty, pulse_parameters
pp = pulse_parameters(time, pressure, u_pressure)  # u_pressure: scalar, 1D, or 2D
```
Uses NumPy's PocketFFT backend.

### Octave
```matlab
result = deconvolution.deconvolve_without_uncertainty(signal, freq_resp, fs);
pp = deconvolution.pulse_parameters(time, pressure, u_pressure);
```

### C++
```cpp
#include "deconvolution.hpp"
auto result = hydrophone::deconvolve_without_uncertainty(signal, freq_resp, fs);
auto pp = hydrophone::pulse_parameters(time, pressure, u_scalar);  // or u_vector / U_matrix
```
Uses Eigen FFT (kissfft backend). `deconvolve_with_uncertainty` also available.

### C#
```csharp
using HydrophoneDeconvolution;
var result = Deconvolution.DeconvolveWithoutUncertainty(signal, freqResp, fs);
var pp = Deconvolution.PulseParameters(time, pressure, uScalar);  // or uVector / uMatrix
```
Uses MathNet.Numerics Fourier transforms. `DeconvolveWithUncertainty` also available.

### Rust
```rust
use hydrophone_deconvolution::{deconvolve_without_uncertainty, pulse_parameters, PulseUncertainty};
let result = deconvolve_without_uncertainty(&signal, &freq_resp, fs);
let pp = pulse_parameters(&time, &pressure, PulseUncertainty::Scalar(0.01));
```
Uses rustfft with manual 1/N normalization.
