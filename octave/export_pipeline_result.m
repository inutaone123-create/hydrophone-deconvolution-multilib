% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0
%
% Export pipeline results for validation.
% Usage: octave export_pipeline_result.m <signal> <noise> <cal> <usebode> <filter> <fc> <outpath>

args = argv();
if length(args) < 7
    error('Usage: octave export_pipeline_result.m <signal> <noise> <cal> <usebode> <filter> <fc> <outpath>');
end

signal_file = args{1};
noise_file = args{2};
cal_file = args{3};
usebode = strcmp(args{4}, 'true');
filter_type = args{5};
fc = str2double(args{6});
outpath = args{7};

result = deconvolution.full_pipeline(signal_file, noise_file, cal_file, usebode, filter_type, fc);

% Save
data = [result.time, result.scaled, result.deconvolved, result.regularized, result.uncertainty];
fid = fopen(outpath, 'w');
fprintf(fid, '# time;scaled;deconvolved;regularized;uncertainty(k=1)\n');
for i = 1:size(data, 1)
    fprintf(fid, '%.18e;%.18e;%.18e;%.18e;%.18e\n', data(i,:));
end
pp = result.pulse_params;
fprintf(fid, '# pc_value=%.18e;pc_uncertainty=%.18e;pc_time=%.18e;pr_value=%.18e;pr_uncertainty=%.18e;pr_time=%.18e;ppsi_value=%.18e;ppsi_uncertainty=%.18e\n', ...
    pp.pc_value, pp.pc_uncertainty, pp.pc_time, pp.pr_value, pp.pr_uncertainty, pp.pr_time, pp.ppsi_value, pp.ppsi_uncertainty);
fclose(fid);

printf('OK: %s\n', outpath);
