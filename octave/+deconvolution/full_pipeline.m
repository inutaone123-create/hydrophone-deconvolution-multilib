% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function result = full_pipeline(measurement_file, noise_file, cal_file, ...
                                usebode, filter_type, fc, fmin, fmax)
    % Run complete deconvolution pipeline.
    if nargin < 7; fmin = 1e6; end
    if nargin < 8; fmax = 60e6; end

    % 1. Read measurement signal
    [time_vec, voltage, dt] = deconvolution.read_dat_signal(measurement_file);
    N = length(voltage);

    % 2. Read noise and estimate uncertainty
    [~, noise_voltage, ~] = deconvolution.read_dat_signal(noise_file);
    % Re-read raw values for std computation (before DC removal)
    noise_raw = load(noise_file);
    nn = noise_raw(1);
    noise_vals = noise_raw(5:4+nn);
    stdev = std(noise_vals, 1);  % population std (divide by N, not N-1)
    uncertainty_vec = ones(N, 1) * stdev;

    % 3. Frequency scale
    frequency = deconvolution.calcfreqscale(time_vec);
    M = floor(N/2) + 1;
    half_freq = frequency(1:M);

    % 4. GUM_DFT
    [spectrum, varspec] = deconvolution.gum_dft(voltage, uncertainty_vec.^2);
    spectrum = 2 * spectrum / N;
    varspec = 4 * varspec / N^2;

    % 5. Read calibration
    hyd = deconvolution.read_calibration_csv(cal_file);

    % 6-7. Calibration
    [~, imin] = min(abs(hyd.frequency - fmin));
    [~, imax] = min(abs(hyd.frequency - fmax));

    if usebode
        amp = sqrt(hyd.real_part.^2 + hyd.imag_part.^2);
        summand_real = (hyd.real_part ./ amp).^2 .* hyd.varreal;
        summand_imag = (hyd.imag_part ./ amp).^2 .* hyd.varimag;
        summand_corr = 2 * (hyd.real_part ./ amp) .* (hyd.imag_part ./ amp) .* hyd.kovar;
        varamp = summand_real + summand_imag + summand_corr;

        freq_trim = hyd.frequency(imin:imax);
        amp_trim = amp(imin:imax);
        varamp_trim = varamp(imin:imax);

        % Clamp interpolation (match numpy.interp behavior: no extrapolation)
        hf_clamped = max(min(half_freq, freq_trim(end)), freq_trim(1));
        ampip = interp1(freq_trim, amp_trim, hf_clamped, 'linear');
        varampip = interp1(freq_trim, varamp_trim, hf_clamped, 'linear');

        [phaseip, varphaseip] = deconvolution.bode_equation(half_freq, ampip, varampip);
        [x, ux] = deconvolution.amp_phase_to_dft(ampip, phaseip, [varampip; varphaseip]);
    else
        src_freq = hyd.frequency(imin:imax);
        hf_clamped = max(min(half_freq, src_freq(end)), src_freq(1));
        real_ip = interp1(src_freq, hyd.real_part(imin:imax), hf_clamped, 'linear');
        imag_ip = interp1(src_freq, hyd.imag_part(imin:imax), hf_clamped, 'linear');
        varreal_ip = interp1(src_freq, hyd.varreal(imin:imax), hf_clamped, 'linear');
        varimag_ip = interp1(src_freq, hyd.varimag(imin:imax), hf_clamped, 'linear');
        kovar_ip = interp1(src_freq, hyd.kovar(imin:imax), hf_clamped, 'linear');

        x = [real_ip; imag_ip];
        a = [diag(varreal_ip), diag(kovar_ip)];
        b = [diag(kovar_ip), diag(varimag_ip)];
        ux = [a; b];
    end

    % 8. DFT_deconv
    [deconv_p, deconv_Up] = deconvolution.dft_deconv(x, spectrum, ux, varspec);

    % 9. Regularization filter
    filt = deconvolution.regularization_filter(half_freq, fc, filter_type);

    % 10. DFT_multiply
    [regul_p, regul_Up] = deconvolution.dft_multiply(deconv_p, filt, deconv_Up);

    % 11. GUM_iDFT
    [sigp_raw, Usigp_raw] = deconvolution.gum_idft(regul_p, regul_Up);
    N_sig = length(sigp_raw);
    sigp = sigp_raw * N_sig / 2;
    Usigp = Usigp_raw * (N_sig^2) / 4;

    % Unfiltered deconvolution
    [deconv_raw, ~] = deconvolution.gum_idft(deconv_p, deconv_Up);
    deconv_time_p = deconv_raw * length(deconv_raw) / 2;

    % Scaled signal
    hyd_amp = sqrt(x(1:M).^2 + x(M+1:end).^2);
    sig_amp = sqrt(spectrum(1:M).^2 + spectrum(M+1:end).^2);
    [~, iffun] = max(sig_amp);
    hydempffun = hyd_amp(iffun);
    scaled = voltage / hydempffun;

    % Uncertainty
    deltasigp = sqrt(diag(Usigp));

    % Pulse parameters
    pp = deconvolution.pulse_parameters(time_vec, sigp, Usigp);

    result.time = time_vec;
    result.scaled = scaled;
    result.deconvolved = deconv_time_p;
    result.regularized = sigp;
    result.uncertainty = deltasigp;
    result.pulse_params = pp;
    result.n_samples = N;
end
