% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0

function result = pulse_parameters(time, pressure, u_pressure)
    % Calculate pulse parameters (pc, pr, ppsi) and their uncertainties.
    %
    % Args:
    %   time      - time vector
    %   pressure  - pressure vector (deconvolved)
    %   u_pressure - uncertainty: scalar, vector, or covariance matrix
    %
    % Returns:
    %   result - struct with pc_value, pc_uncertainty, pc_index, pc_time,
    %            pr_value, pr_uncertainty, pr_index, pr_time,
    %            ppsi_value, ppsi_uncertainty

    n = length(pressure);

    % Build covariance matrix U_p
    if isscalar(u_pressure)
        U_p = diag(repmat(u_pressure^2, n, 1));
    elseif isvector(u_pressure)
        U_p = diag(u_pressure(:).^2);
    else
        U_p = u_pressure;
    end

    dt = (time(end) - time(1)) / (n - 1);

    % pc: compressional peak pressure
    [pc_value, pc_index] = max(pressure(:));
    pc_uncertainty = sqrt(U_p(pc_index, pc_index));
    pc_time = time(pc_index);

    % pr: rarefactional peak pressure (positive value)
    [pr_min, pr_index] = min(pressure(:));
    pr_value = -pr_min;
    pr_uncertainty = sqrt(U_p(pr_index, pr_index));
    pr_time = time(pr_index);

    % ppsi: pulse pressure-squared integral
    ppsi_value = sum(pressure(:).^2) * dt;
    C = 2.0 * abs(pressure(:)) * dt;
    ppsi_uncertainty = sqrt(C' * U_p * C);

    result.pc_value = pc_value;
    result.pc_uncertainty = pc_uncertainty;
    result.pc_index = pc_index;
    result.pc_time = pc_time;
    result.pr_value = pr_value;
    result.pr_uncertainty = pr_uncertainty;
    result.pr_index = pr_index;
    result.pr_time = pr_time;
    result.ppsi_value = ppsi_value;
    result.ppsi_uncertainty = ppsi_uncertainty;
end
