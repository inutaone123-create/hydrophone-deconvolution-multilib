% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function hyd = read_calibration_csv(filepath)
    % Read hydrophone calibration CSV.
    %
    % Returns struct with frequency, real_part, imag_part, varreal, varimag, kovar.

    data = csvread(filepath, 1, 0);
    hyd.frequency = data(:, 1) * 1e6;  % MHz -> Hz
    hyd.real_part = data(:, 2);
    hyd.imag_part = data(:, 3);
    hyd.varreal = data(:, 4);
    hyd.varimag = data(:, 5);
    hyd.kovar = data(:, 6);
end
