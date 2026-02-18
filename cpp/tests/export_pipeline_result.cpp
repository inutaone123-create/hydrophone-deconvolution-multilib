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

#include "pipeline.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <signal> <noise> <cal> <usebode> <filter> <fc> <outpath>"
                  << std::endl;
        return 1;
    }

    std::string signal_file = argv[1];
    std::string noise_file = argv[2];
    std::string cal_file = argv[3];
    bool usebode = std::string(argv[4]) == "true";
    std::string filter_type = argv[5];
    double fc = std::stod(argv[6]);
    std::string outpath = argv[7];

    try {
        auto result = hydrophone::pipeline::full_pipeline(
            signal_file, noise_file, cal_file,
            usebode, filter_type, fc
        );

        std::ofstream out(outpath);
        out << std::setprecision(18) << std::scientific;
        out << "# time;scaled;deconvolved;regularized;uncertainty(k=1)" << std::endl;

        for (int i = 0; i < result.n_samples; i++) {
            out << result.time(i) << ";"
                << result.scaled(i) << ";"
                << result.deconvolved(i) << ";"
                << result.regularized(i) << ";"
                << result.uncertainty(i) << std::endl;
        }

        auto& pp = result.pulse_params;
        out << "# pc_value=" << pp.pc_value
            << ";pc_uncertainty=" << pp.pc_uncertainty
            << ";pc_time=" << pp.pc_time
            << ";pr_value=" << pp.pr_value
            << ";pr_uncertainty=" << pp.pr_uncertainty
            << ";pr_time=" << pp.pr_time
            << ";ppsi_value=" << pp.ppsi_value
            << ";ppsi_uncertainty=" << pp.ppsi_uncertainty
            << std::endl;

        std::cout << "OK: " << outpath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
