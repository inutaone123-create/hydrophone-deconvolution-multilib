% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [phase, varphase] = bode_equation(frequencies, amplitudes, varamplitudes)
    % Reconstruct phase from amplitude using Bode integral equation.

    frequencies = frequencies(:);
    amplitudes = amplitudes(:);
    varamplitudes = varamplitudes(:);
    n = length(frequencies);
    df = frequencies(2) - frequencies(1);

    phase = zeros(n, 1);
    varphase = zeros(n, 1);

    for i = 1:n
        numerator = log(amplitudes) - log(amplitudes(i));
        denominator = frequencies.^2 - frequencies(i)^2;
        denominator(i) = 1;
        phase(i) = 2 * frequencies(i) / pi * df * sum(numerator ./ denominator);

        denominatoru = amplitudes .* denominator;
        numeratoru = ones(n, 1);
        numeratoru(i) = 0;
        varphase(i) = (2 * frequencies(i) / pi * df)^2 * ...
            sum(((numeratoru ./ denominatoru).^2) .* varamplitudes);
    end
end
