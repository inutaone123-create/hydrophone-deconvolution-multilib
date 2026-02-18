% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function [time_vec, voltage, dt] = read_dat_signal(filepath)
    % Read .DAT measurement signal file.
    %
    % Format: line0=n_samples, line1=dt, line2-3=skip, line4+=voltage

    rawdata = load(filepath);
    n_samples = rawdata(1);
    dt = rawdata(2);
    voltage = rawdata(5:end);
    voltage = voltage - mean(voltage);
    time_vec = (0:n_samples-1)' * dt;
end
