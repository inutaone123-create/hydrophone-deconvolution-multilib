% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [x, ux] = amp_phase_to_dft(amp, phase, Uap)
    % Convert amplitude+phase to real+imag with uncertainty.
    %
    % Args:
    %   amp   - Amplitude [M x 1]
    %   phase - Phase [M x 1]
    %   Uap   - Variance vector [2M x 1] = [var_amp; var_phase]

    amp = amp(:); phase = phase(:); Uap = Uap(:);
    M = length(amp);

    re = amp .* cos(phase);
    im = amp .* sin(phase);
    x = [re; im];

    var_amp = Uap(1:M);
    var_phase = Uap(M+1:end);

    ux = zeros(2*M, 2*M);
    for k = 1:M
        J = [cos(phase(k)), -amp(k)*sin(phase(k));
             sin(phase(k)),  amp(k)*cos(phase(k))];
        U_in = diag([var_amp(k), var_phase(k)]);
        U_out = J * U_in * J';
        ux(k,k) = U_out(1,1);
        ux(k,k+M) = U_out(1,2);
        ux(k+M,k) = U_out(2,1);
        ux(k+M,k+M) = U_out(2,2);
    end
end
