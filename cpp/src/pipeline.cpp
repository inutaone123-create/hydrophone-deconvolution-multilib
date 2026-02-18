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
#include <unsupported/Eigen/FFT>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace hydrophone {
namespace pipeline {

static const double PI = 3.14159265358979323846;

Eigen::VectorXd calcfreqscale(const Eigen::VectorXd& timeseries) {
    int n = timeseries.size();
    int M = n / 2 + 1;
    double fmax = 1.0 / ((timeseries.maxCoeff() - timeseries.minCoeff()) * n / (n - 1)) * (n - 1);
    Eigen::VectorXd f = Eigen::VectorXd::LinSpaced(n, 0, fmax);
    Eigen::VectorXd f2(2 * M);
    f2.head(M) = f.head(M);
    f2.tail(M) = f.head(M);
    return f2;
}

DFTResult gum_dft(const Eigen::VectorXd& x, const Eigen::VectorXd& Ux) {
    int N = x.size();
    int M = N / 2 + 1;

    // FFT for best estimate
    Eigen::FFT<double> fft;
    std::vector<std::complex<double>> spectrum;
    std::vector<double> xvec(x.data(), x.data() + N);
    fft.fwd(spectrum, xvec);

    Eigen::VectorXd F(2 * M);
    for (int k = 0; k < M; k++) {
        F(k) = spectrum[k].real();
        F(k + M) = spectrum[k].imag();
    }

    // Sensitivity matrix
    Eigen::MatrixXd CxCos(M, N);
    Eigen::MatrixXd CxSin(M, N);
    for (int k = 0; k < M; k++) {
        for (int n = 0; n < N; n++) {
            double phase = 2.0 * PI * k * n / N;
            CxCos(k, n) = cos(phase);
            CxSin(k, n) = -sin(phase);
        }
    }

    Eigen::MatrixXd C(2 * M, N);
    C.topRows(M) = CxCos;
    C.bottomRows(M) = CxSin;

    // UF = C * diag(Ux) * C'
    Eigen::MatrixXd CU = C.array().rowwise() * Ux.transpose().array();
    Eigen::MatrixXd UF = CU * C.transpose();

    return {F, UF};
}

DFTResult gum_idft(const Eigen::VectorXd& F, const Eigen::MatrixXd& UF) {
    int M = F.size() / 2;
    int N = 2 * (M - 1);

    // irfft for best estimate
    Eigen::FFT<double> fft;
    std::vector<std::complex<double>> spec(M);
    for (int k = 0; k < M; k++) {
        spec[k] = std::complex<double>(F(k), F(k + M));
    }
    std::vector<double> xvec;
    fft.inv(xvec, spec, N);

    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(xvec.data(), N);

    // Sensitivity matrices
    Eigen::MatrixXd Cc(N, M);
    Eigen::MatrixXd Cs(N, M);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < M; k++) {
            double phase = 2.0 * PI * n * k / N;
            Cc(n, k) = cos(phase);
            Cs(n, k) = -sin(phase);
        }
    }

    // Adjust for rfft convention
    Cc.rightCols(M - 1) *= 2.0;
    Cs.rightCols(M - 1) *= 2.0;
    if (N % 2 == 0) {
        Cc.col(M - 1) *= 0.5;
        Cs.col(M - 1) *= 0.5;
    }

    Eigen::MatrixXd RR = UF.topLeftCorner(M, M);
    Eigen::MatrixXd RI = UF.topRightCorner(M, M);
    Eigen::MatrixXd II = UF.bottomRightCorner(M, M);

    Eigen::MatrixXd Ux = Cc * RR * Cc.transpose();
    Eigen::MatrixXd term2 = Cc * RI * Cs.transpose();
    Ux += term2 + term2.transpose();
    Ux += Cs * II * Cs.transpose();
    Ux /= (double)(N * N);

    return {x, Ux};
}

DFTResult dft_deconv(const Eigen::VectorXd& H, const Eigen::VectorXd& Y,
                     const Eigen::MatrixXd& UH, const Eigen::MatrixXd& UY) {
    int M = H.size() / 2;

    Eigen::VectorXd Hr = H.head(M), Hi = H.tail(M);
    Eigen::VectorXd Yr = Y.head(M), Yi = Y.tail(M);
    Eigen::VectorXd D = Hr.array().square() + Hi.array().square();
    D = D.array().max(1e-30);

    Eigen::VectorXd Xr = (Yr.array() * Hr.array() + Yi.array() * Hi.array()) / D.array();
    Eigen::VectorXd Xi = (Yi.array() * Hr.array() - Yr.array() * Hi.array()) / D.array();

    Eigen::VectorXd X(2 * M);
    X.head(M) = Xr;
    X.tail(M) = Xi;

    Eigen::MatrixXd UX = Eigen::MatrixXd::Zero(2 * M, 2 * M);
    for (int k = 0; k < M; k++) {
        Eigen::Matrix2d JY, JH, UY_k, UH_k;
        JY << Hr(k)/D(k), Hi(k)/D(k), -Hi(k)/D(k), Hr(k)/D(k);
        JH << (Yr(k)-2*Hr(k)*Xr(k))/D(k), (Yi(k)-2*Hi(k)*Xr(k))/D(k),
               (Yi(k)-2*Hr(k)*Xi(k))/D(k), (-Yr(k)-2*Hi(k)*Xi(k))/D(k);
        UY_k << UY(k,k), UY(k,k+M), UY(k+M,k), UY(k+M,k+M);
        UH_k << UH(k,k), UH(k,k+M), UH(k+M,k), UH(k+M,k+M);

        Eigen::Matrix2d UX_k = JY * UY_k * JY.transpose() + JH * UH_k * JH.transpose();
        UX(k, k) = UX_k(0, 0);
        UX(k, k+M) = UX_k(0, 1);
        UX(k+M, k) = UX_k(1, 0);
        UX(k+M, k+M) = UX_k(1, 1);
    }
    return {X, UX};
}

DFTResult dft_multiply(const Eigen::VectorXd& Y, const Eigen::VectorXd& F,
                       const Eigen::MatrixXd& UY) {
    int M = Y.size() / 2;
    Eigen::VectorXd Yr = Y.head(M), Yi = Y.tail(M);
    Eigen::VectorXd Fr = F.head(M), Fi = F.tail(M);

    Eigen::VectorXd Zr = Yr.array() * Fr.array() - Yi.array() * Fi.array();
    Eigen::VectorXd Zi = Yr.array() * Fi.array() + Yi.array() * Fr.array();
    Eigen::VectorXd Z(2 * M);
    Z.head(M) = Zr;
    Z.tail(M) = Zi;

    Eigen::MatrixXd UZ = Eigen::MatrixXd::Zero(2 * M, 2 * M);
    for (int k = 0; k < M; k++) {
        Eigen::Matrix2d JY, UY_k;
        JY << Fr(k), -Fi(k), Fi(k), Fr(k);
        UY_k << UY(k,k), UY(k,k+M), UY(k+M,k), UY(k+M,k+M);
        Eigen::Matrix2d UZ_k = JY * UY_k * JY.transpose();
        UZ(k, k) = UZ_k(0, 0);
        UZ(k, k+M) = UZ_k(0, 1);
        UZ(k+M, k) = UZ_k(1, 0);
        UZ(k+M, k+M) = UZ_k(1, 1);
    }
    return {Z, UZ};
}

void bode_equation(const Eigen::VectorXd& freq, const Eigen::VectorXd& amp,
                   const Eigen::VectorXd& varamp,
                   Eigen::VectorXd& phase, Eigen::VectorXd& varphase) {
    int n = freq.size();
    double df = freq(1) - freq(0);
    phase.resize(n);
    varphase.resize(n);

    Eigen::VectorXd logamp = amp.array().log();

    for (int i = 0; i < n; i++) {
        Eigen::VectorXd numerator = logamp.array() - logamp(i);
        Eigen::VectorXd denom = freq.array().square() - freq(i) * freq(i);
        denom(i) = 1.0;

        double coeff = 2.0 * freq(i) / PI * df;
        phase(i) = coeff * (numerator.array() / denom.array()).sum();

        Eigen::VectorXd denomu = amp.array() * denom.array();
        Eigen::VectorXd numeratoru = Eigen::VectorXd::Ones(n);
        numeratoru(i) = 0.0;
        varphase(i) = coeff * coeff * ((numeratoru.array() / denomu.array()).square() * varamp.array()).sum();
    }
}

DFTResult amp_phase_to_dft(const Eigen::VectorXd& amp, const Eigen::VectorXd& phase,
                           const Eigen::VectorXd& Uap) {
    int M = amp.size();
    Eigen::VectorXd re = amp.array() * phase.array().cos();
    Eigen::VectorXd im = amp.array() * phase.array().sin();
    Eigen::VectorXd x(2 * M);
    x.head(M) = re;
    x.tail(M) = im;

    Eigen::VectorXd var_amp = Uap.head(M);
    Eigen::VectorXd var_phase = Uap.tail(M);

    Eigen::MatrixXd ux = Eigen::MatrixXd::Zero(2 * M, 2 * M);
    for (int k = 0; k < M; k++) {
        double cp = cos(phase(k)), sp = sin(phase(k));
        Eigen::Matrix2d J;
        J << cp, -amp(k)*sp, sp, amp(k)*cp;
        Eigen::Matrix2d U_in = Eigen::Matrix2d::Zero();
        U_in(0, 0) = var_amp(k);
        U_in(1, 1) = var_phase(k);
        Eigen::Matrix2d U_out = J * U_in * J.transpose();
        ux(k, k) = U_out(0, 0);
        ux(k, k+M) = U_out(0, 1);
        ux(k+M, k) = U_out(1, 0);
        ux(k+M, k+M) = U_out(1, 1);
    }
    return {x, ux};
}

Eigen::VectorXd regularization_filter(const Eigen::VectorXd& freq, double fc,
                                      const std::string& filter_type) {
    int M = freq.size();
    Eigen::VectorXcd Hc(M);

    if (filter_type == "LowPass") {
        for (int i = 0; i < M; i++) {
            std::complex<double> s = 1.0 + std::complex<double>(0, 1) * freq(i) / (fc * 1.555);
            Hc(i) = 1.0 / (s * s);
        }
    } else if (filter_type == "CriticalDamping") {
        for (int i = 0; i < M; i++) {
            std::complex<double> jf = std::complex<double>(0, 1) * freq(i) / fc;
            Hc(i) = 1.0 / (1.0 + 1.28719 * jf + 0.41421 * jf * jf);
        }
    } else if (filter_type == "Bessel") {
        for (int i = 0; i < M; i++) {
            std::complex<double> jf = std::complex<double>(0, 1) * freq(i) / fc;
            Hc(i) = 1.0 / (1.0 + 1.3617 * jf - 0.6180 * freq(i) * freq(i) / (fc * fc));
        }
    } else if (filter_type == "None") {
        Hc.setOnes();
    } else {
        throw std::runtime_error("Unknown filter type: " + filter_type);
    }

    // Phase correction
    if (filter_type != "None") {
        int ind3dB = 0;
        double minDiff = 1e30;
        for (int i = 0; i < M; i++) {
            double diff = std::abs(std::abs(Hc(i)) - std::sqrt(0.5));
            if (diff < minDiff) {
                minDiff = diff;
                ind3dB = i;
            }
        }
        if (ind3dB > 1) {
            // lstsq: f[:ind3dB] \ angle(Hc[:ind3dB])
            Eigen::VectorXd f_sub = freq.head(ind3dB);
            Eigen::VectorXd ang_sub(ind3dB);
            for (int i = 0; i < ind3dB; i++) {
                ang_sub(i) = std::arg(Hc(i));
            }
            double w = f_sub.dot(ang_sub) / f_sub.dot(f_sub);
            for (int i = 0; i < M; i++) {
                Hc(i) *= std::exp(std::complex<double>(0, -w * freq(i)));
            }
        }
    }

    Eigen::VectorXd filt(2 * M);
    for (int i = 0; i < M; i++) {
        filt(i) = Hc(i).real();
        filt(i + M) = Hc(i).imag();
    }
    return filt;
}

SignalData read_dat_signal(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filepath);

    std::vector<double> values;
    double val;
    while (file >> val) values.push_back(val);

    int n_samples = (int)values[0];
    double dt = values[1];

    Eigen::VectorXd voltage(n_samples);
    for (int i = 0; i < n_samples; i++) {
        voltage(i) = values[4 + i];
    }
    voltage.array() -= voltage.mean();

    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(n_samples, 0, (n_samples - 1) * dt);

    return {time, voltage, dt};
}

CalibrationData read_calibration_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filepath);

    std::string line;
    std::getline(file, line); // skip header

    std::vector<std::vector<double>> rows;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string token;
        while (std::getline(iss, token, ',')) {
            row.push_back(std::stod(token));
        }
        if (row.size() >= 6) rows.push_back(row);
    }

    int n = rows.size();
    CalibrationData cal;
    cal.frequency.resize(n);
    cal.real_part.resize(n);
    cal.imag_part.resize(n);
    cal.varreal.resize(n);
    cal.varimag.resize(n);
    cal.kovar.resize(n);

    for (int i = 0; i < n; i++) {
        cal.frequency(i) = rows[i][0] * 1e6;
        cal.real_part(i) = rows[i][1];
        cal.imag_part(i) = rows[i][2];
        cal.varreal(i) = rows[i][3];
        cal.varimag(i) = rows[i][4];
        cal.kovar(i) = rows[i][5];
    }
    return cal;
}

// Linear interpolation helper
static Eigen::VectorXd interp1(const Eigen::VectorXd& xp, const Eigen::VectorXd& fp,
                                const Eigen::VectorXd& x) {
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++) {
        double xi = x(i);
        if (xi <= xp(0)) {
            result(i) = fp(0);
        } else if (xi >= xp(xp.size() - 1)) {
            result(i) = fp(fp.size() - 1);
        } else {
            auto it = std::lower_bound(xp.data(), xp.data() + xp.size(), xi);
            int idx = it - xp.data();
            if (idx == 0) idx = 1;
            double t = (xi - xp(idx - 1)) / (xp(idx) - xp(idx - 1));
            result(i) = fp(idx - 1) + t * (fp(idx) - fp(idx - 1));
        }
    }
    return result;
}

PipelineResult full_pipeline(const std::string& measurement_file,
                             const std::string& noise_file,
                             const std::string& cal_file,
                             bool usebode,
                             const std::string& filter_type,
                             double fc,
                             double fmin,
                             double fmax) {
    // 1. Read signal
    auto sig = read_dat_signal(measurement_file);
    int N = sig.voltage.size();

    // 2. Noise uncertainty
    auto noise_sig = read_dat_signal(noise_file);
    double stdev = 0;
    {
        double mean = noise_sig.voltage.mean();
        Eigen::VectorXd diff = noise_sig.voltage.array() - mean;
        // Note: read_dat_signal already removed mean, so use raw noise
        std::ifstream nf(noise_file);
        std::vector<double> nvals;
        double v;
        while (nf >> v) nvals.push_back(v);
        int nn = (int)nvals[0];
        Eigen::VectorXd nvoltage(nn);
        for (int i = 0; i < nn; i++) nvoltage(i) = nvals[4 + i];
        double nmean = nvoltage.mean();
        stdev = std::sqrt((nvoltage.array() - nmean).square().mean());
    }
    Eigen::VectorXd uncertainty_vec = Eigen::VectorXd::Constant(N, stdev);

    // 3. Frequency scale
    auto frequency = calcfreqscale(sig.time);
    int M = N / 2 + 1;
    Eigen::VectorXd half_freq = frequency.head(M);

    // 4. GUM_DFT
    auto [spectrum, varspec] = gum_dft(sig.voltage, uncertainty_vec.array().square().matrix());
    spectrum *= 2.0 / N;
    varspec *= 4.0 / (N * N);

    // 5. Calibration
    auto cal = read_calibration_csv(cal_file);

    // 6-7. Calibration processing
    int imin = 0, imax = 0;
    {
        double minDiff = 1e30;
        for (int i = 0; i < cal.frequency.size(); i++) {
            double d = std::abs(cal.frequency(i) - fmin);
            if (d < minDiff) { minDiff = d; imin = i; }
        }
        minDiff = 1e30;
        for (int i = 0; i < cal.frequency.size(); i++) {
            double d = std::abs(cal.frequency(i) - fmax);
            if (d < minDiff) { minDiff = d; imax = i; }
        }
    }

    Eigen::VectorXd x, ux_placeholder;
    Eigen::MatrixXd ux;

    if (usebode) {
        Eigen::VectorXd amp = (cal.real_part.array().square() + cal.imag_part.array().square()).sqrt();
        Eigen::VectorXd varamp = (cal.real_part.array() / amp.array()).square() * cal.varreal.array()
            + (cal.imag_part.array() / amp.array()).square() * cal.varimag.array()
            + 2.0 * (cal.real_part.array() / amp.array()) * (cal.imag_part.array() / amp.array()) * cal.kovar.array();

        int range_n = imax - imin + 1;
        Eigen::VectorXd freq_trim = cal.frequency.segment(imin, range_n);
        Eigen::VectorXd amp_trim = amp.segment(imin, range_n);
        Eigen::VectorXd varamp_trim = varamp.segment(imin, range_n);

        Eigen::VectorXd ampip = interp1(freq_trim, amp_trim, half_freq);
        Eigen::VectorXd varampip = interp1(freq_trim, varamp_trim, half_freq);

        Eigen::VectorXd phaseip, varphaseip;
        bode_equation(half_freq, ampip, varampip, phaseip, varphaseip);

        Eigen::VectorXd Uap(2 * M);
        Uap.head(M) = varampip;
        Uap.tail(M) = varphaseip;
        auto [x_bode, ux_bode] = amp_phase_to_dft(ampip, phaseip, Uap);
        x = x_bode;
        ux = ux_bode;
    } else {
        int range_n = imax - imin + 1;
        Eigen::VectorXd src_freq = cal.frequency.segment(imin, range_n);

        Eigen::VectorXd real_ip = interp1(src_freq, cal.real_part.segment(imin, range_n), half_freq);
        Eigen::VectorXd imag_ip = interp1(src_freq, cal.imag_part.segment(imin, range_n), half_freq);
        Eigen::VectorXd varreal_ip = interp1(src_freq, cal.varreal.segment(imin, range_n), half_freq);
        Eigen::VectorXd varimag_ip = interp1(src_freq, cal.varimag.segment(imin, range_n), half_freq);
        Eigen::VectorXd kovar_ip = interp1(src_freq, cal.kovar.segment(imin, range_n), half_freq);

        x.resize(2 * M);
        x.head(M) = real_ip;
        x.tail(M) = imag_ip;

        ux = Eigen::MatrixXd::Zero(2 * M, 2 * M);
        for (int i = 0; i < M; i++) {
            ux(i, i) = varreal_ip(i);
            ux(i, i + M) = kovar_ip(i);
            ux(i + M, i) = kovar_ip(i);
            ux(i + M, i + M) = varimag_ip(i);
        }
    }

    // 8. DFT_deconv
    auto [deconv_p, deconv_Up] = dft_deconv(x, spectrum, ux, varspec);

    // 9. Filter
    auto filt = regularization_filter(half_freq, fc, filter_type);

    // 10. DFT_multiply
    auto [regul_p, regul_Up] = dft_multiply(deconv_p, filt, deconv_Up);

    // 11. GUM_iDFT
    auto [sigp_raw, Usigp_raw] = gum_idft(regul_p, regul_Up);
    int N_sig = sigp_raw.size();
    Eigen::VectorXd sigp = sigp_raw * N_sig / 2.0;
    Eigen::MatrixXd Usigp = Usigp_raw * (N_sig * N_sig) / 4.0;

    // Unfiltered deconvolution
    auto [deconv_raw, deconv_U_raw] = gum_idft(deconv_p, deconv_Up);
    Eigen::VectorXd deconv_time_p = deconv_raw * deconv_raw.size() / 2.0;

    // Scaled signal
    Eigen::VectorXd hyd_amp = (x.head(M).array().square() + x.tail(M).array().square()).sqrt();
    Eigen::VectorXd sig_amp = (spectrum.head(M).array().square() + spectrum.tail(M).array().square()).sqrt();
    int iffun;
    sig_amp.maxCoeff(&iffun);
    double hydempffun = hyd_amp(iffun);
    Eigen::VectorXd scaled = sig.voltage / hydempffun;

    // Uncertainty
    Eigen::VectorXd deltasigp = Usigp.diagonal().array().sqrt();

    // Pulse parameters
    auto pp = hydrophone::pulse_parameters(sig.time, sigp, Usigp);

    return {sig.time, scaled, deconv_time_p, sigp, deltasigp, pp, N};
}

} // namespace pipeline
} // namespace hydrophone
