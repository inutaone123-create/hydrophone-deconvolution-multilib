/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * Original License: CC BY 4.0
 *
 * This implementation: 2024
 * License: CC BY 4.0
 */

using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace HydrophoneDeconvolution;

public static class Deconvolution
{
    public static double[] DeconvolveWithoutUncertainty(
        double[] measuredSignal,
        Complex[] frequencyResponse,
        double samplingRate)
    {
        int n = measuredSignal.Length;
        var signal = new Complex[n];
        for (int i = 0; i < n; i++)
            signal[i] = new Complex(measuredSignal[i], 0);

        Fourier.Forward(signal, FourierOptions.NoScaling);

        const double epsilon = 1e-12;
        var deconvolved = new Complex[n];
        for (int i = 0; i < n; i++)
            deconvolved[i] = signal[i] / (frequencyResponse[i] + epsilon);

        Fourier.Inverse(deconvolved, FourierOptions.NoScaling);

        var result = new double[n];
        double scale = 1.0 / n;
        for (int i = 0; i < n; i++)
            result[i] = deconvolved[i].Real * scale;

        return result;
    }
}
