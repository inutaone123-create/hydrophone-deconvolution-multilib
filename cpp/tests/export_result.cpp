/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * License: CC BY 4.0
 */

#include "deconvolution.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

Eigen::VectorXd load_csv(const std::string& path) {
    std::ifstream file(path);
    std::vector<double> values;
    double val;
    while (file >> val) {
        values.push_back(val);
    }
    Eigen::VectorXd result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result(i) = values[i];
    }
    return result;
}

int main() {
    std::string data_dir = "../../test-data/";
    std::string results_dir = "../../validation/results/";
    std::filesystem::create_directories(results_dir);

    auto measured = load_csv(data_dir + "measured_signal.csv");
    auto freq_real = load_csv(data_dir + "freq_response_real.csv");
    auto freq_imag = load_csv(data_dir + "freq_response_imag.csv");

    Eigen::VectorXcd freq_response(freq_real.size());
    for (int i = 0; i < freq_real.size(); ++i) {
        freq_response(i) = std::complex<double>(freq_real(i), freq_imag(i));
    }

    auto result = hydrophone::deconvolve_without_uncertainty(measured, freq_response, 1e7);

    std::ofstream out(results_dir + "cpp_result.csv");
    out << std::setprecision(18) << std::scientific;
    for (int i = 0; i < result.size(); ++i) {
        out << result(i) << "\n";
    }

    std::cout << "C++ result exported: " << result.size() << " samples" << std::endl;
    return 0;
}
