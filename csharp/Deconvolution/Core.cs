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

public class DeconvolutionResult
{
    public double[] Mean { get; set; } = Array.Empty<double>();
    public double[] Std { get; set; } = Array.Empty<double>();
}

public class PulseParametersResult
{
    public double PcValue { get; set; }
    public double PcUncertainty { get; set; }
    public int PcIndex { get; set; }
    public double PcTime { get; set; }
    public double PrValue { get; set; }
    public double PrUncertainty { get; set; }
    public int PrIndex { get; set; }
    public double PrTime { get; set; }
    public double PpsiValue { get; set; }
    public double PpsiUncertainty { get; set; }
}

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

    public static DeconvolutionResult DeconvolveWithUncertainty(
        double[] measuredSignal,
        double[] signalUncertainty,
        Complex[] frequencyResponse,
        double[] responseUncertainty,
        double samplingRate,
        int numMonteCarlo = 1000)
    {
        int n = measuredSignal.Length;
        var rng = new Random(42);
        var mcResults = new double[numMonteCarlo][];

        for (int i = 0; i < numMonteCarlo; i++)
        {
            var signalPert = new double[n];
            for (int j = 0; j < n; j++)
                signalPert[j] = measuredSignal[j] + signalUncertainty[j] * NextGaussian(rng);

            var freqRespPert = new Complex[frequencyResponse.Length];
            for (int j = 0; j < frequencyResponse.Length; j++)
            {
                double noise = responseUncertainty[j] * NextGaussian(rng);
                freqRespPert[j] = frequencyResponse[j] + new Complex(noise, noise);
            }

            mcResults[i] = DeconvolveWithoutUncertainty(signalPert, freqRespPert, samplingRate);
        }

        var mean = new double[n];
        var std = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int i = 0; i < numMonteCarlo; i++)
                sum += mcResults[i][j];
            mean[j] = sum / numMonteCarlo;

            double varSum = 0;
            for (int i = 0; i < numMonteCarlo; i++)
                varSum += (mcResults[i][j] - mean[j]) * (mcResults[i][j] - mean[j]);
            std[j] = Math.Sqrt(varSum / numMonteCarlo);
        }

        return new DeconvolutionResult { Mean = mean, Std = std };
    }

    public static PulseParametersResult PulseParameters(
        double[] time, double[] pressure, double[,] uP)
    {
        int n = pressure.Length;
        double dt = (time[n - 1] - time[0]) / (n - 1);

        var result = new PulseParametersResult();

        // pc
        int pcIdx = 0;
        double pcMax = pressure[0];
        for (int i = 1; i < n; i++)
        {
            if (pressure[i] > pcMax) { pcMax = pressure[i]; pcIdx = i; }
        }
        result.PcIndex = pcIdx;
        result.PcValue = pcMax;
        result.PcUncertainty = Math.Sqrt(uP[pcIdx, pcIdx]);
        result.PcTime = time[pcIdx];

        // pr
        int prIdx = 0;
        double prMin = pressure[0];
        for (int i = 1; i < n; i++)
        {
            if (pressure[i] < prMin) { prMin = pressure[i]; prIdx = i; }
        }
        result.PrIndex = prIdx;
        result.PrValue = -prMin;
        result.PrUncertainty = Math.Sqrt(uP[prIdx, prIdx]);
        result.PrTime = time[prIdx];

        // ppsi
        double ppsiVal = 0;
        for (int i = 0; i < n; i++)
            ppsiVal += pressure[i] * pressure[i];
        ppsiVal *= dt;
        result.PpsiValue = ppsiVal;

        double[] C = new double[n];
        for (int i = 0; i < n; i++)
            C[i] = 2.0 * Math.Abs(pressure[i]) * dt;

        double ppsiVar = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                ppsiVar += C[i] * uP[i, j] * C[j];
        result.PpsiUncertainty = Math.Sqrt(ppsiVar);

        return result;
    }

    public static PulseParametersResult PulseParameters(
        double[] time, double[] pressure, double uScalar)
    {
        int n = pressure.Length;
        var uP = new double[n, n];
        double uSq = uScalar * uScalar;
        for (int i = 0; i < n; i++)
            uP[i, i] = uSq;
        return PulseParameters(time, pressure, uP);
    }

    public static PulseParametersResult PulseParameters(
        double[] time, double[] pressure, double[] uVector)
    {
        int n = pressure.Length;
        var uP = new double[n, n];
        for (int i = 0; i < n; i++)
            uP[i, i] = uVector[i] * uVector[i];
        return PulseParameters(time, pressure, uP);
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
