% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function filt = regularization_filter(freq, fc, filter_type)
    % Create regularization filter in PyDynamic format [Re; Im].
    %
    % Args:
    %   freq        - Positive frequencies [M x 1]
    %   fc          - Cutoff frequency [Hz]
    %   filter_type - 'LowPass', 'CriticalDamping', 'Bessel', or 'None'

    freq = freq(:);
    f = freq;

    if strcmp(filter_type, 'LowPass')
        Hc = 1 ./ (1 + 1i * f / (fc * 1.555)).^2;
    elseif strcmp(filter_type, 'CriticalDamping')
        Hc = 1 ./ (1 + 1.28719 * 1i * f / fc + 0.41421 * (1i * f / fc).^2);
    elseif strcmp(filter_type, 'Bessel')
        Hc = 1 ./ (1 + 1.3617 * 1i * f / fc - 0.6180 * f.^2 / fc^2);
    elseif strcmp(filter_type, 'None')
        Hc = ones(size(f));
    else
        error('Unknown filter type: %s', filter_type);
    end

    % Phase correction
    if ~strcmp(filter_type, 'None')
        [~, ind3dB] = min(abs(abs(Hc) - sqrt(0.5)));
        if ind3dB > 2
            w = f(1:ind3dB-1) \ angle(Hc(1:ind3dB-1));
            Hc = Hc .* exp(-1i * w * f);
        end
    end

    filt = [real(Hc); imag(Hc)];
end
