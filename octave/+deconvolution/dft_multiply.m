% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [Z, UZ] = dft_multiply(Y, F, UY)
    % Frequency-domain multiplication Z = Y * F with uncertainty propagation.
    %
    % F is deterministic (no uncertainty).

    Y = Y(:); F = F(:);
    M = length(Y) / 2;
    Yr = Y(1:M); Yi = Y(M+1:end);
    Fr = F(1:M); Fi = F(M+1:end);

    Zr = Yr .* Fr - Yi .* Fi;
    Zi = Yr .* Fi + Yi .* Fr;
    Z = [Zr; Zi];

    UZ = zeros(2*M, 2*M);
    for k = 1:M
        JY = [Fr(k), -Fi(k); Fi(k), Fr(k)];
        UY_k = [UY(k,k), UY(k,k+M); UY(k+M,k), UY(k+M,k+M)];
        UZ_k = JY * UY_k * JY';
        UZ(k,k) = UZ_k(1,1);
        UZ(k,k+M) = UZ_k(1,2);
        UZ(k+M,k) = UZ_k(2,1);
        UZ(k+M,k+M) = UZ_k(2,2);
    end
end
