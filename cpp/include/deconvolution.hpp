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

} // namespace hydrophone

#endif
