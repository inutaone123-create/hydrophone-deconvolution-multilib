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
#include <iostream>
#include <cmath>
#include <random>
#include <unsupported/Eigen/FFT>

int main() {
    std::cout << "=== C++ Deconvolution Tests ===" << std::endl;
    int passed = 0, failed = 0;

    // Test 1: Output length
    std::cout << "Test 1: Output length... ";
    {
        int n = 1024;
        std::mt19937 gen(42);
        std::normal_distribution<> dist(0.0, 1.0);

        Eigen::VectorXd signal(n);
        Eigen::VectorXd temp(n);
        for (int i = 0; i < n; ++i) {
            signal(i) = dist(gen);
            temp(i) = dist(gen);
        }

        Eigen::FFT<double> fft;
        Eigen::VectorXcd freq_resp;
        fft.fwd(freq_resp, temp);
        for (int i = 0; i < n; ++i) freq_resp(i) += 1.0;

        auto result = hydrophone::deconvolve_without_uncertainty(signal, freq_resp, 1e7);
        if (result.size() == n) {
            std::cout << "PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "FAILED" << std::endl;
            failed++;
        }
    }

    // Test 2: Known signal recovery
    std::cout << "Test 2: Known signal recovery... ";
    {
        int n = 256;
        std::mt19937 gen1(42);
        std::normal_distribution<> dist(0.0, 1.0);

        Eigen::VectorXd original(n);
        for (int i = 0; i < n; ++i) original(i) = dist(gen1);

        std::mt19937 gen2(99);
        Eigen::VectorXd temp(n);
        for (int i = 0; i < n; ++i) temp(i) = dist(gen2);

        Eigen::FFT<double> fft;
        Eigen::VectorXcd freq_resp;
        fft.fwd(freq_resp, temp);
        for (int i = 0; i < n; ++i) freq_resp(i) += 2.0;

        Eigen::VectorXcd original_fft;
        fft.fwd(original_fft, original);

        Eigen::VectorXcd convolved_fft(n);
        for (int i = 0; i < n; ++i) {
            convolved_fft(i) = original_fft(i) * freq_resp(i);
        }

        Eigen::VectorXd measured;
        fft.inv(measured, convolved_fft);

        auto recovered = hydrophone::deconvolve_without_uncertainty(measured, freq_resp, 1e7);

        double max_err = (recovered - original).cwiseAbs().maxCoeff();
        if (max_err < 1e-10) {
            std::cout << "PASSED (max error: " << max_err << ")" << std::endl;
            passed++;
        } else {
            std::cout << "FAILED (max error: " << max_err << ")" << std::endl;
            failed++;
        }
    }

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed > 0 ? 1 : 0;
}
