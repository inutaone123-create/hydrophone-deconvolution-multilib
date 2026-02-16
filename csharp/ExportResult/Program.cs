/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * License: CC BY 4.0
 */

using System.Globalization;
using System.Numerics;
using HydrophoneDeconvolution;

string dataDir = "/workspace/test-data";
string resultsDir = "/workspace/validation/results";
Directory.CreateDirectory(resultsDir);

double[] LoadCsv(string path) =>
    File.ReadAllLines(path)
        .Where(l => !string.IsNullOrWhiteSpace(l))
        .Select(l => double.Parse(l.Trim(), CultureInfo.InvariantCulture))
        .ToArray();

var measured = LoadCsv(Path.Combine(dataDir, "measured_signal.csv"));
var freqReal = LoadCsv(Path.Combine(dataDir, "freq_response_real.csv"));
var freqImag = LoadCsv(Path.Combine(dataDir, "freq_response_imag.csv"));

var freqResponse = new Complex[freqReal.Length];
for (int i = 0; i < freqReal.Length; i++)
    freqResponse[i] = new Complex(freqReal[i], freqImag[i]);

var result = Deconvolution.DeconvolveWithoutUncertainty(measured, freqResponse, 1e7);

using var writer = new StreamWriter(Path.Combine(resultsDir, "csharp_result.csv"));
foreach (var val in result)
    writer.WriteLine(val.ToString("E18", CultureInfo.InvariantCulture));

Console.WriteLine($"C# result exported: {result.Length} samples");
