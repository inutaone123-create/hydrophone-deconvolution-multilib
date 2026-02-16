% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% License: CC BY 4.0

% Add parent directory to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fprintf('=== Octave Deconvolution Tests ===\n\n');
passed = 0;
failed = 0;

% Test 1: Output length
fprintf('Test 1: Output length... ');
n = 1024;
signal = randn(1, n);
freq_resp = fft(randn(1, n)) + 1.0;
result = deconvolution.deconvolve_without_uncertainty(signal, freq_resp, 1e7);
if length(result) == n
    fprintf('PASSED\n');
    passed = passed + 1;
else
    fprintf('FAILED\n');
    failed = failed + 1;
end

% Test 2: Output is real
fprintf('Test 2: Output is real... ');
if all(imag(result) == 0)
    fprintf('PASSED\n');
    passed = passed + 1;
else
    fprintf('FAILED\n');
    failed = failed + 1;
end

% Test 3: Known signal recovery
fprintf('Test 3: Known signal recovery... ');
rand('state', 42);
original = randn(1, 256);
rand('state', 99);
freq_resp2 = fft(randn(1, 256)) + 2.0;
signal_fft = fft(original);
measured = real(ifft(signal_fft .* freq_resp2));
recovered = deconvolution.deconvolve_without_uncertainty(measured, freq_resp2, 1e7);
max_err = max(abs(recovered - original));
if max_err < 1e-10
    fprintf('PASSED (max error: %.2e)\n', max_err);
    passed = passed + 1;
else
    fprintf('FAILED (max error: %.2e)\n', max_err);
    failed = failed + 1;
end

fprintf('\n=== Results: %d passed, %d failed ===\n', passed, failed);
if failed > 0
    exit(1);
end
