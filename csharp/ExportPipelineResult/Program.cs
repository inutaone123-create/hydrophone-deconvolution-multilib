/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * License: CC BY 4.0
 */

using System.Globalization;
using HydrophoneDeconvolution;

if (args.Length < 7)
{
    Console.Error.WriteLine("Usage: ExportPipelineResult <signal> <noise> <cal> <usebode> <filter> <fc> <outpath>");
    return 1;
}

string signalFile = args[0];
string noiseFile = args[1];
string calFile = args[2];
bool usebode = args[3] == "true";
string filterType = args[4];
double fc = double.Parse(args[5], CultureInfo.InvariantCulture);
string outpath = args[6];

try
{
    var result = Pipeline.FullPipeline(signalFile, noiseFile, calFile, usebode, filterType, fc);

    using var writer = new StreamWriter(outpath);
    writer.WriteLine("# time;scaled;deconvolved;regularized;uncertainty(k=1)");
    for (int i = 0; i < result.NSamples; i++)
    {
        writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0:E18};{1:E18};{2:E18};{3:E18};{4:E18}",
            result.Time[i], result.Scaled[i], result.Deconvolved[i],
            result.Regularized[i], result.Uncertainty[i]));
    }
    var pp = result.PulseParams;
    writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "# pc_value={0:E18};pc_uncertainty={1:E18};pc_time={2:E18};pr_value={3:E18};pr_uncertainty={4:E18};pr_time={5:E18};ppsi_value={6:E18};ppsi_uncertainty={7:E18}",
        pp.PcValue, pp.PcUncertainty, pp.PcTime, pp.PrValue, pp.PrUncertainty, pp.PrTime, pp.PpsiValue, pp.PpsiUncertainty));

    Console.WriteLine($"OK: {outpath}");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"ERROR: {ex.Message}");
    return 1;
}
