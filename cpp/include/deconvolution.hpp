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

#ifndef DECONVOLUTION_HPP
#define DECONVOLUTION_HPP

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <unsupported/Eigen/FFT>

namespace hydrophone {

Eigen::VectorXd deconvolve_without_uncertainty(
    const Eigen::VectorXd& measured_signal,
    const Eigen::VectorXcd& frequency_response,
    double sampling_rate
);

struct DeconvolutionResult {
    Eigen::VectorXd mean;
    Eigen::VectorXd std;
};

DeconvolutionResult deconvolve_with_uncertainty(
    const Eigen::VectorXd& measured_signal,
    const Eigen::VectorXd& signal_uncertainty,
    const Eigen::VectorXcd& frequency_response,
    const Eigen::VectorXd& response_uncertainty,
    double sampling_rate,
    int num_monte_carlo = 1000
);

struct PulseParameters {
    double pc_value;
    double pc_uncertainty;
    int pc_index;
    double pc_time;
    double pr_value;
    double pr_uncertainty;
    int pr_index;
    double pr_time;
    double ppsi_value;
    double ppsi_uncertainty;
};

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    const Eigen::MatrixXd& U_p
);

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    double u_scalar
);

PulseParameters pulse_parameters(
    const Eigen::VectorXd& time,
    const Eigen::VectorXd& pressure,
    const Eigen::VectorXd& u_vector
);

} // namespace hydrophone

#endif
