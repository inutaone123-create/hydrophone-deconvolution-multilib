% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [X, UX] = dft_deconv(H, Y, UH, UY)
    % Frequency-domain deconvolution X = Y / H with uncertainty propagation.
    %
    % All inputs in PyDynamic format [Re; Im].

    H = H(:); Y = Y(:);
    M = length(H) / 2;
    Hr = H(1:M); Hi = H(M+1:end);
    Yr = Y(1:M); Yi = Y(M+1:end);

    % Complex division
    D = Hr.^2 + Hi.^2;
    D(D == 0) = 1e-30;

    Xr = (Yr .* Hr + Yi .* Hi) ./ D;
    Xi = (Yi .* Hr - Yr .* Hi) ./ D;
    X = [Xr; Xi];

    % Uncertainty propagation per bin
    UX = zeros(2*M, 2*M);
    for k = 1:M
        JY = [Hr(k)/D(k), Hi(k)/D(k); -Hi(k)/D(k), Hr(k)/D(k)];
        JH = [(Yr(k) - 2*Hr(k)*Xr(k))/D(k), (Yi(k) - 2*Hi(k)*Xr(k))/D(k);
               (Yi(k) - 2*Hr(k)*Xi(k))/D(k), (-Yr(k) - 2*Hi(k)*Xi(k))/D(k)];

        UY_k = [UY(k,k), UY(k,k+M); UY(k+M,k), UY(k+M,k+M)];
        UH_k = [UH(k,k), UH(k,k+M); UH(k+M,k), UH(k+M,k+M)];

        UX_k = JY * UY_k * JY' + JH * UH_k * JH';
        UX(k,k) = UX_k(1,1);
        UX(k,k+M) = UX_k(1,2);
        UX(k+M,k) = UX_k(2,1);
        UX(k+M,k+M) = UX_k(2,2);
    end
end
