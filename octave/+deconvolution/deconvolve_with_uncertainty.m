% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [deconvolved, uncertainty] = deconvolve_with_uncertainty(...
    measured_signal, signal_uncertainty, ...
    frequency_response, response_uncertainty, ...
    sampling_rate, num_monte_carlo)

    if nargin < 6
        num_monte_carlo = 1000;
    end

    n_samples = length(measured_signal);
    mc_results = zeros(num_monte_carlo, n_samples);

    for i = 1:num_monte_carlo
        signal_pert = measured_signal + randn(size(measured_signal)) .* signal_uncertainty;
        resp_pert = frequency_response + randn(size(response_uncertainty)) .* response_uncertainty .* (1 + 1i);
        mc_results(i, :) = deconvolution.deconvolve_without_uncertainty(signal_pert, resp_pert, sampling_rate);
    end

    deconvolved = mean(mc_results, 1)';
    uncertainty = std(mc_results, 0, 1)';
end
