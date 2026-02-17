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
#include <random>
#include <cmath>

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

DeconvolutionResult deconvolve_with_uncertainty(
    const Eigen::VectorXd& measured_signal,
    const Eigen::VectorXd& signal_uncertainty,
    const Eigen::VectorXcd& frequency_response,
    const Eigen::VectorXd& response_uncertainty,
    double sampling_rate,
    int num_monte_carlo)
{
    int n = measured_signal.size();
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXd mc_results(num_monte_carlo, n);

    for (int i = 0; i < num_monte_carlo; ++i) {
        Eigen::VectorXd signal_pert(n);
        for (int j = 0; j < n; ++j) {
            signal_pert(j) = measured_signal(j) + signal_uncertainty(j) * dist(rng);
        }

        Eigen::VectorXcd freq_resp_pert(frequency_response.size());
        for (int j = 0; j < frequency_response.size(); ++j) {
            double noise = response_uncertainty(j) * dist(rng);
            freq_resp_pert(j) = frequency_response(j) + std::complex<double>(noise, noise);
        }

        Eigen::VectorXd result = deconvolve_without_uncertainty(signal_pert, freq_resp_pert, sampling_rate);
        mc_results.row(i) = result.transpose();
    }

    DeconvolutionResult out;
    out.mean = mc_results.colwise().mean();
    Eigen::MatrixXd centered = mc_results.rowwise() - out.mean.transpose();
    out.std = (centered.array().square().colwise().mean()).sqrt();
    return out;
}

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    const Eigen::MatrixXd& U_p)
{
    int n = pressure.size();
    double dt = (time(n - 1) - time(0)) / (n - 1);

    PulseParameters result;

    // pc: compressional peak pressure
    pressure.maxCoeff(&result.pc_index);
    result.pc_value = pressure(result.pc_index);
    result.pc_uncertainty = std::sqrt(U_p(result.pc_index, result.pc_index));
    result.pc_time = time(result.pc_index);

    // pr: rarefactional peak pressure
    pressure.minCoeff(&result.pr_index);
    result.pr_value = -pressure(result.pr_index);
    result.pr_uncertainty = std::sqrt(U_p(result.pr_index, result.pr_index));
    result.pr_time = time(result.pr_index);

    // ppsi: pulse pressure-squared integral
    result.ppsi_value = pressure.array().square().sum() * dt;
    Eigen::VectorXd C = 2.0 * pressure.array().abs() * dt;
    result.ppsi_uncertainty = std::sqrt((C.transpose() * U_p * C)(0, 0));

    return result;
}

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    double u_scalar)
{
    int n = pressure.size();
    Eigen::MatrixXd U_p = Eigen::MatrixXd::Identity(n, n) * (u_scalar * u_scalar);
    return pulse_parameters(time, pressure, U_p);
}

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    const Eigen::VectorXd& u_vector)
{
    Eigen::MatrixXd U_p = u_vector.array().square().matrix().asDiagonal();
    return pulse_parameters(time, pressure, U_p);
}

} // namespace hydrophone
