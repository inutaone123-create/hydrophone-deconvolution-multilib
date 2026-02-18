% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function f2 = calcfreqscale(timeseries)
    % Calculate frequency scale matching PyDynamic's GUM_DFT output format.

    n = length(timeseries);
    fmax = 1 / ((max(timeseries) - min(timeseries)) * n / (n-1)) * (n-1);
    f = linspace(0, fmax, n)';
    M = floor(n/2) + 1;
    f2 = [f(1:M); f(1:M)];
end
