% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [F, UF] = gum_dft(x, Ux)
    % GUM-compliant DFT with uncertainty propagation.
    %
    % Args:
    %   x  - Time-domain signal [N x 1]
    %   Ux - Variance vector [N x 1] (diagonal of covariance matrix)
    %
    % Returns:
    %   F  - Spectrum [2M x 1] in PyDynamic format [Re; Im]
    %   UF - Covariance matrix [2M x 2M]

    x = x(:);
    Ux = Ux(:);
    N = length(x);
    M = floor(N/2) + 1;

    % Best estimate via rfft
    F_complex = fft(x);
    F_complex = F_complex(1:M);  % take positive frequencies only
    F = [real(F_complex); imag(F_complex)];

    % Sensitivity matrix
    beta = 2 * pi * (0:N-1)' / N;  % [N x 1]
    k_idx = (0:M-1);               % [1 x M]
    phase = beta * k_idx;           % [N x M]

    CxCos = cos(phase)';            % [M x N]
    CxSin = -sin(phase)';           % [M x N]

    C = [CxCos; CxSin];            % [2M x N]

    % UF = C * diag(Ux) * C'
    CU = C .* Ux';                  % [2M x N] element-wise
    UF = CU * C';                   % [2M x 2M]
end
