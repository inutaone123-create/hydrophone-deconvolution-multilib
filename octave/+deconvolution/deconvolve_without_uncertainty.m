% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function deconvolved = deconvolve_without_uncertainty(measured_signal, frequency_response, sampling_rate)
    signal_fft = fft(measured_signal);
    epsilon = 1e-12;
    deconvolved_fft = signal_fft ./ (frequency_response + epsilon);
    deconvolved = real(ifft(deconvolved_fft));
end
