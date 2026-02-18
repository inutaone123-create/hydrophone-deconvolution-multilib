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

#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <Eigen/Dense>
#include <string>
#include <map>
#include "deconvolution.hpp"

namespace hydrophone {
namespace pipeline {

// GUM DFT result
struct DFTResult {
    Eigen::VectorXd F;   // [2M] Re/Im interleaved
    Eigen::MatrixXd UF;  // [2M x 2M] covariance
};

// Signal data from .DAT file
struct SignalData {
    Eigen::VectorXd time;
    Eigen::VectorXd voltage;
    double dt;
};

// Calibration data from .CSV
struct CalibrationData {
    Eigen::VectorXd frequency;
    Eigen::VectorXd real_part;
    Eigen::VectorXd imag_part;
    Eigen::VectorXd varreal;
    Eigen::VectorXd varimag;
    Eigen::VectorXd kovar;
};

// Pipeline result
struct PipelineResult {
    Eigen::VectorXd time;
    Eigen::VectorXd scaled;
    Eigen::VectorXd deconvolved;
    Eigen::VectorXd regularized;
    Eigen::VectorXd uncertainty;
    PulseParameters pulse_params;
    int n_samples;
};

// Frequency scale
Eigen::VectorXd calcfreqscale(const Eigen::VectorXd& timeseries);

// GUM DFT/iDFT
DFTResult gum_dft(const Eigen::VectorXd& x, const Eigen::VectorXd& Ux);
DFTResult gum_idft(const Eigen::VectorXd& F, const Eigen::MatrixXd& UF);

// DFT deconvolution and multiplication
DFTResult dft_deconv(const Eigen::VectorXd& H, const Eigen::VectorXd& Y,
                     const Eigen::MatrixXd& UH, const Eigen::MatrixXd& UY);
DFTResult dft_multiply(const Eigen::VectorXd& Y, const Eigen::VectorXd& F,
                       const Eigen::MatrixXd& UY);

// Bode equation
void bode_equation(const Eigen::VectorXd& freq, const Eigen::VectorXd& amp,
                   const Eigen::VectorXd& varamp,
                   Eigen::VectorXd& phase, Eigen::VectorXd& varphase);

// AmpPhase to DFT
DFTResult amp_phase_to_dft(const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
                           const Eigen::VectorXd& Uap);

// Regularization filter
Eigen::VectorXd regularization_filter(const Eigen::VectorXd& freq, double fc,
                                      const std::string& filter_type);

// I/O
SignalData read_dat_signal(const std::string& filepath);
CalibrationData read_calibration_csv(const std::string& filepath);

// Full pipeline
PipelineResult full_pipeline(const std::string& measurement_file,
                             const std::string& noise_file,
                             const std::string& cal_file,
                             bool usebode,
                             const std::string& filter_type,
                             double fc,
                             double fmin = 1e6,
                             double fmax = 60e6);

} // namespace pipeline
} // namespace hydrophone

#endif
