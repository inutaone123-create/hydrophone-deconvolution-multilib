% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% License: CC BY 4.0

addpath(fullfile(fileparts(mfilename('fullpath'))));

% Load test data
measured_signal = load('../test-data/measured_signal.csv')';
freq_real = load('../test-data/freq_response_real.csv')';
freq_imag = load('../test-data/freq_response_imag.csv')';
frequency_response = freq_real + 1i * freq_imag;

% Deconvolve
result = deconvolution.deconvolve_without_uncertainty(measured_signal, frequency_response, 1e7);

% Save result
results_dir = '../validation/results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
dlmwrite(fullfile(results_dir, 'octave_result.csv'), result', '%.18e');
fprintf('Octave result exported: %d samples\n', length(result));
