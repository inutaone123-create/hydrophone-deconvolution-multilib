/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * Original License: CC BY 4.0
 *
 * This implementation: 2024
 * License: CC BY 4.0
 */

#include "deconvolution.hpp"

namespace hydrophone {

Eigen::VectorXd deconvolve_without_uncertainty(
    const Eigen::VectorXd& measured_signal,
    const Eigen::VectorXcd& frequency_response,
    double sampling_rate)
{
    int n = measured_signal.size();
    Eigen::FFT<double> fft;

    // Convert real signal to complex for full-spectrum FFT
    Eigen::VectorXcd signal_complex(n);
    for (int i = 0; i < n; ++i) {
        signal_complex(i) = std::complex<double>(measured_signal(i), 0.0);
    }

    // Forward FFT (complex-to-complex for full spectrum)
    Eigen::VectorXcd signal_fft;
    fft.fwd(signal_fft, signal_complex);

    // Deconvolution in frequency domain
    const double epsilon = 1e-12;
    Eigen::VectorXcd deconvolved_fft(n);
    for (int i = 0; i < n; ++i) {
        deconvolved_fft(i) = signal_fft(i) / (frequency_response(i) + epsilon);
    }

    // Inverse FFT
    Eigen::VectorXcd deconvolved_complex;
    fft.inv(deconvolved_complex, deconvolved_fft);

    Eigen::VectorXd deconvolved(n);
    for (int i = 0; i < n; ++i) {
        deconvolved(i) = deconvolved_complex(i).real();
    }

    return deconvolved;
}

} // namespace hydrophone
