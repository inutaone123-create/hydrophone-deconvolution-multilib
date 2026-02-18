% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [x, Ux] = gum_idft(F, UF)
    % GUM-compliant inverse DFT with uncertainty propagation.
    %
    % Args:
    %   F  - Spectrum [2M x 1] in PyDynamic format [Re; Im]
    %   UF - Covariance matrix [2M x 2M]
    %
    % Returns:
    %   x  - Time-domain signal [N x 1]
    %   Ux - Covariance matrix [N x N]

    F = F(:);
    total = length(F);
    M = total / 2;
    N = 2 * (M - 1);

    % Best estimate via irfft (reconstruct full symmetric spectrum)
    F_complex = F(1:M) + 1i * F(M+1:end);
    full_spec = zeros(N, 1);
    full_spec(1) = F_complex(1);
    full_spec(2:M) = F_complex(2:M);
    full_spec(N:-1:M+1) = conj(F_complex(2:M-1));
    x = real(ifft(full_spec));

    % Sensitivity matrices (without 1/N factor)
    beta = 2 * pi * (0:N-1)' / N;  % [N x 1]
    k_idx = (0:M-1);               % [1 x M]
    k_beta = beta * k_idx;         % [N x M]

    Cc = cos(k_beta);              % [N x M]
    Cs = -sin(k_beta);             % [N x M]

    % Adjust for rfft convention: multiply interior bins by 2
    Cc(:, 2:end) = Cc(:, 2:end) * 2;
    Cs(:, 2:end) = Cs(:, 2:end) * 2;
    % Undo factor 2 for Nyquist (even N)
    if mod(N, 2) == 0
        Cc(:, M) = Cc(:, M) * 0.5;
        Cs(:, M) = Cs(:, M) * 0.5;
    end

    % Uncertainty propagation
    RR = UF(1:M, 1:M);
    RI = UF(1:M, M+1:end);
    II = UF(M+1:end, M+1:end);

    Ux = Cc * RR * Cc';
    term2 = Cc * RI * Cs';
    Ux = Ux + term2 + term2';
    Ux = Ux + Cs * II * Cs';
    Ux = Ux / (N^2);
end
