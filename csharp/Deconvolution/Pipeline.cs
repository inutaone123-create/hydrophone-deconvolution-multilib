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

using System.Globalization;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace HydrophoneDeconvolution;

public class DFTResult
{
    public double[] F { get; set; } = Array.Empty<double>();
    public double[,] UF { get; set; } = new double[0, 0];
}

public class SignalData
{
    public double[] Time { get; set; } = Array.Empty<double>();
    public double[] Voltage { get; set; } = Array.Empty<double>();
    public double Dt { get; set; }
}

public class CalibrationData
{
    public double[] Frequency { get; set; } = Array.Empty<double>();
    public double[] RealPart { get; set; } = Array.Empty<double>();
    public double[] ImagPart { get; set; } = Array.Empty<double>();
    public double[] VarReal { get; set; } = Array.Empty<double>();
    public double[] VarImag { get; set; } = Array.Empty<double>();
    public double[] Kovar { get; set; } = Array.Empty<double>();
}

public class PipelineResultData
{
    public double[] Time { get; set; } = Array.Empty<double>();
    public double[] Scaled { get; set; } = Array.Empty<double>();
    public double[] Deconvolved { get; set; } = Array.Empty<double>();
    public double[] Regularized { get; set; } = Array.Empty<double>();
    public double[] Uncertainty { get; set; } = Array.Empty<double>();
    public PulseParametersResult PulseParams { get; set; } = new();
    public int NSamples { get; set; }
}

public static class Pipeline
{
    public static double[] CalcFreqScale(double[] timeseries)
    {
        int n = timeseries.Length;
        int M = n / 2 + 1;
        double tMin = timeseries[0], tMax = timeseries[0];
        for (int i = 1; i < n; i++) { if (timeseries[i] < tMin) tMin = timeseries[i]; if (timeseries[i] > tMax) tMax = timeseries[i]; }
        double fmax = 1.0 / ((tMax - tMin) * n / (n - 1)) * (n - 1);
        var f2 = new double[2 * M];
        for (int i = 0; i < M; i++)
        {
            double fi = fmax * i / (n - 1.0);
            f2[i] = fi;
            f2[i + M] = fi;
        }
        return f2;
    }

    public static DFTResult GumDft(double[] x, double[] Ux)
    {
        int N = x.Length;
        int M = N / 2 + 1;

        // FFT for best estimate
        var signal = new Complex[N];
        for (int i = 0; i < N; i++) signal[i] = new Complex(x[i], 0);
        Fourier.Forward(signal, FourierOptions.NoScaling);

        var F = new double[2 * M];
        for (int k = 0; k < M; k++)
        {
            F[k] = signal[k].Real;
            F[k + M] = signal[k].Imaginary;
        }

        // For uniform variance (iid noise), UF is diagonal due to DFT orthogonality.
        // UF[k,k] = σ² * Σ cos²(2πkn/N) = σ² * N/2 (interior), σ² * N (DC, Nyquist)
        // UF[k+M,k+M] = σ² * Σ sin²(2πkn/N) = σ² * N/2 (interior), 0 (DC, Nyquist)
        // All off-diagonal elements are zero.
        bool uniform = true;
        double ux0 = Ux[0];
        for (int i = 1; i < N && uniform; i++)
            if (Math.Abs(Ux[i] - ux0) > 1e-30 * Math.Max(Math.Abs(ux0), 1e-300)) uniform = false;

        if (uniform)
        {
            var UF2 = new double[2 * M, 2 * M];
            // Compute diagonal elements exactly
            for (int k = 0; k < M; k++)
            {
                double sumCC = 0, sumSS = 0;
                for (int n = 0; n < N; n++)
                {
                    double phase = 2.0 * Math.PI * k * n / N;
                    double c = Math.Cos(phase);
                    double s = -Math.Sin(phase);
                    sumCC += c * c;
                    sumSS += s * s;
                }
                UF2[k, k] = ux0 * sumCC;
                UF2[k + M, k + M] = ux0 * sumSS;
            }
            return new DFTResult { F = F, UF = UF2 };
        }

        // UF = C * diag(Ux) * C' where C = [CxCos; CxSin]
        // Compute CxCos and CxSin arrays, then multiply
        double[,] UF = new double[2 * M, 2 * M];

        // Precompute CU = C * diag(Ux) as [2M x N] then UF = CU * C'
        // But 2M*N can be large. Use block approach:
        // UF = [CC*diag*CC', CC*diag*CS'; CS*diag*CC', CS*diag*CS']
        // For each block, UF_block[k1,k2] = Σ_n cos/sin(k1*β_n) * Ux[n] * cos/sin(k2*β_n)

        // Precompute cos/sin values [M x N] - store as flat arrays for cache efficiency
        var cosvals = new double[M * N];
        var sinvals = new double[M * N];
        for (int k = 0; k < M; k++)
        {
            for (int n = 0; n < N; n++)
            {
                double phase = 2.0 * Math.PI * k * n / N;
                cosvals[k * N + n] = Math.Cos(phase);
                sinvals[k * N + n] = -Math.Sin(phase);
            }
        }

        // Compute UF blocks: CC, CS, SS
        for (int k1 = 0; k1 < M; k1++)
        {
            int off1c = k1 * N;
            for (int k2 = k1; k2 < M; k2++)
            {
                int off2c = k2 * N;
                double cc = 0, cs = 0, ss = 0;
                for (int n = 0; n < N; n++)
                {
                    double c1 = cosvals[off1c + n];
                    double s1 = sinvals[off1c + n];
                    double c2 = cosvals[off2c + n];
                    double s2 = sinvals[off2c + n];
                    double u = Ux[n];
                    cc += c1 * u * c2;
                    cs += c1 * u * s2;
                    ss += s1 * u * s2;
                }
                UF[k1, k2] = cc; UF[k2, k1] = cc;
                UF[k1, k2 + M] = cs; UF[k2 + M, k1] = cs;
                // SC block: UF[k1+M, k2] = Σ s1 * u * c2
                double sc = 0;
                if (k1 != k2)
                {
                    for (int n = 0; n < N; n++)
                        sc += sinvals[off1c + n] * Ux[n] * cosvals[off2c + n];
                    UF[k1 + M, k2] = sc; UF[k2, k1 + M] = sc;
                }
                else
                {
                    UF[k1 + M, k2] = cs; UF[k2, k1 + M] = cs;
                }
                UF[k1 + M, k2 + M] = ss; UF[k2 + M, k1 + M] = ss;
            }
            // Fill remaining SC for k2 < k1
            for (int k2 = 0; k2 < k1; k2++)
            {
                // UF[k1+M, k2] = Σ sin(k1) * u * cos(k2)
                int off2c2 = k2 * N;
                double sc2 = 0;
                for (int n = 0; n < N; n++)
                    sc2 += sinvals[off1c + n] * Ux[n] * cosvals[off2c2 + n];
                UF[k1 + M, k2] = sc2;
                UF[k2, k1 + M] = sc2;
            }
        }

        return new DFTResult { F = F, UF = UF };
    }

    public static (double[] x, double[,] Ux) GumIdft(double[] F, double[,] UF)
    {
        int M = F.Length / 2;
        int N = 2 * (M - 1);

        // irfft for best estimate
        var spec = new Complex[N];
        spec[0] = new Complex(F[0], F[M]);
        for (int k = 1; k < M; k++)
        {
            spec[k] = new Complex(F[k], F[k + M]);
            spec[N - k] = new Complex(F[k], -F[k + M]);
        }
        Fourier.Inverse(spec, FourierOptions.NoScaling);

        var x = new double[N];
        double scale = 1.0 / N;
        for (int i = 0; i < N; i++) x[i] = spec[i].Real * scale;

        // Precompute adjusted sensitivity coefficients
        // Cc[n,k] = cos(2πnk/N) * mult[k], Cs[n,k] = -sin(2πnk/N) * mult[k]
        // where mult[0]=1, mult[1..M-2]=2, mult[M-1]=1 (for even N)
        var mult = new double[M];
        mult[0] = 1.0;
        for (int k = 1; k < M - 1; k++) mult[k] = 2.0;
        mult[M - 1] = (N % 2 == 0) ? 1.0 : 2.0;

        // Check if UF is diagonal (common fast path)
        bool isDiag = true;
        for (int i = 0; i < 2 * M && isDiag; i++)
            for (int j = 0; j < 2 * M && isDiag; j++)
                if (i != j && Math.Abs(UF[i, j]) > 1e-30 * Math.Max(Math.Abs(UF[i, i]), 1e-300))
                    isDiag = false;

        var Ux = new double[N, N];
        double invN2 = 1.0 / (N * N);

        if (isDiag)
        {
            // Fast path: UF diagonal → Ux[n1,n2] = (1/N²) * Σ_k (Cc[n1,k]*UF[k,k]*Cc[n2,k] + Cs[n1,k]*UF[k+M,k+M]*Cs[n2,k])
            for (int n1 = 0; n1 < N; n1++)
            {
                for (int n2 = n1; n2 < N; n2++)
                {
                    double sum = 0;
                    for (int k = 0; k < M; k++)
                    {
                        double p1 = 2.0 * Math.PI * n1 * k / N;
                        double p2 = 2.0 * Math.PI * n2 * k / N;
                        double c1 = Math.Cos(p1) * mult[k];
                        double s1 = -Math.Sin(p1) * mult[k];
                        double c2 = Math.Cos(p2) * mult[k];
                        double s2 = -Math.Sin(p2) * mult[k];
                        sum += c1 * UF[k, k] * c2 + s1 * UF[k + M, k + M] * s2;
                    }
                    Ux[n1, n2] = sum * invN2;
                    Ux[n2, n1] = sum * invN2;
                }
            }
        }
        else
        {
            // General path using intermediate products
            double[,] Cc = new double[N, M];
            double[,] Cs = new double[N, M];
            for (int n = 0; n < N; n++)
            {
                for (int k = 0; k < M; k++)
                {
                    double phase = 2.0 * Math.PI * n * k / N;
                    Cc[n, k] = Math.Cos(phase) * mult[k];
                    Cs[n, k] = -Math.Sin(phase) * mult[k];
                }
            }

            var Ac = new double[N, M];
            var As = new double[N, M];
            for (int n1 = 0; n1 < N; n1++)
            {
                for (int k2 = 0; k2 < M; k2++)
                {
                    double sumC = 0, sumS = 0;
                    for (int k1 = 0; k1 < M; k1++)
                    {
                        sumC += Cc[n1, k1] * UF[k1, k2] + Cs[n1, k1] * UF[k1 + M, k2];
                        sumS += Cc[n1, k1] * UF[k1, k2 + M] + Cs[n1, k1] * UF[k1 + M, k2 + M];
                    }
                    Ac[n1, k2] = sumC;
                    As[n1, k2] = sumS;
                }
            }

            for (int n1 = 0; n1 < N; n1++)
            {
                for (int n2 = n1; n2 < N; n2++)
                {
                    double sum = 0;
                    for (int k = 0; k < M; k++)
                        sum += Ac[n1, k] * Cc[n2, k] + As[n1, k] * Cs[n2, k];
                    Ux[n1, n2] = sum * invN2;
                    Ux[n2, n1] = sum * invN2;
                }
            }
        }

        return (x, Ux);
    }

    public static DFTResult DftDeconv(double[] H, double[] Y, double[,] UH, double[,] UY)
    {
        int M = H.Length / 2;
        var X = new double[2 * M];
        var UX = new double[2 * M, 2 * M];

        for (int k = 0; k < M; k++)
        {
            double Hr = H[k], Hi = H[k + M];
            double Yr = Y[k], Yi = Y[k + M];
            double D = Hr * Hr + Hi * Hi;
            if (D == 0) D = 1e-30;

            double Xr = (Yr * Hr + Yi * Hi) / D;
            double Xi = (Yi * Hr - Yr * Hi) / D;
            X[k] = Xr;
            X[k + M] = Xi;

            // Jacobians
            double jy00 = Hr / D, jy01 = Hi / D;
            double jy10 = -Hi / D, jy11 = Hr / D;
            double jh00 = (Yr - 2 * Hr * Xr) / D, jh01 = (Yi - 2 * Hi * Xr) / D;
            double jh10 = (Yi - 2 * Hr * Xi) / D, jh11 = (-Yr - 2 * Hi * Xi) / D;

            double uy00 = UY[k, k], uy01 = UY[k, k + M], uy10 = UY[k + M, k], uy11 = UY[k + M, k + M];
            double uh00 = UH[k, k], uh01 = UH[k, k + M], uh10 = UH[k + M, k], uh11 = UH[k + M, k + M];

            // UX_k = JY*UY_k*JY' + JH*UH_k*JH'
            // (r,c) = Σ_a Σ_b J[r,a]*U[a,b]*J[c,b]
            // tmp = U*J' where J'[b,c] = J[c,b]
            // JY: tmp_y[a,c] = Σ_b UY[a,b]*JY[c,b]
            double ty00 = uy00 * jy00 + uy01 * jy01, ty01 = uy00 * jy10 + uy01 * jy11;
            double ty10 = uy10 * jy00 + uy11 * jy01, ty11 = uy10 * jy10 + uy11 * jy11;
            // JH: tmp_h[a,c] = Σ_b UH[a,b]*JH[c,b]
            double th00 = uh00 * jh00 + uh01 * jh01, th01 = uh00 * jh10 + uh01 * jh11;
            double th10 = uh10 * jh00 + uh11 * jh01, th11 = uh10 * jh10 + uh11 * jh11;
            // result[r,c] = Σ_a J[r,a]*tmp[a,c]
            double ux00 = jy00 * ty00 + jy01 * ty10 + jh00 * th00 + jh01 * th10;
            double ux01 = jy00 * ty01 + jy01 * ty11 + jh00 * th01 + jh01 * th11;
            double ux10 = jy10 * ty00 + jy11 * ty10 + jh10 * th00 + jh11 * th10;
            double ux11 = jy10 * ty01 + jy11 * ty11 + jh10 * th01 + jh11 * th11;

            UX[k, k] = ux00;
            UX[k, k + M] = ux01;
            UX[k + M, k] = ux10;
            UX[k + M, k + M] = ux11;
        }
        return new DFTResult { F = X, UF = UX };
    }

    public static DFTResult DftMultiply(double[] Y, double[] F, double[,] UY)
    {
        int M = Y.Length / 2;
        var Z = new double[2 * M];
        var UZ = new double[2 * M, 2 * M];

        for (int k = 0; k < M; k++)
        {
            double Yr = Y[k], Yi = Y[k + M];
            double Fr = F[k], Fi = F[k + M];
            Z[k] = Yr * Fr - Yi * Fi;
            Z[k + M] = Yr * Fi + Yi * Fr;

            // Jacobian: JY = [[Fr, -Fi],[Fi, Fr]], JY' = [[Fr, Fi],[-Fi, Fr]]
            // UZ_k = JY * UY_k * JY'
            // tmp = UY * JY' → tmp[0,0] = uy00*Fr - uy01*Fi, tmp[0,1] = uy00*Fi + uy01*Fr
            //                   tmp[1,0] = uy10*Fr - uy11*Fi, tmp[1,1] = uy10*Fi + uy11*Fr
            double uy00 = UY[k, k], uy01 = UY[k, k + M], uy10 = UY[k + M, k], uy11 = UY[k + M, k + M];
            double t00 = uy00 * Fr - uy01 * Fi, t01 = uy00 * Fi + uy01 * Fr;
            double t10 = uy10 * Fr - uy11 * Fi, t11 = uy10 * Fi + uy11 * Fr;
            double uz00 = Fr * t00 + (-Fi) * t10;
            double uz01 = Fr * t01 + (-Fi) * t11;
            double uz10 = Fi * t00 + Fr * t10;
            double uz11 = Fi * t01 + Fr * t11;

            UZ[k, k] = uz00;
            UZ[k, k + M] = uz01;
            UZ[k + M, k] = uz10;
            UZ[k + M, k + M] = uz11;
        }
        return new DFTResult { F = Z, UF = UZ };
    }

    public static (double[] phase, double[] varphase) BodeEquation(
        double[] freq, double[] amp, double[] varamp)
    {
        int n = freq.Length;
        double df = freq[1] - freq[0];
        var phase = new double[n];
        var varphase = new double[n];

        var logamp = new double[n];
        for (int i = 0; i < n; i++) logamp[i] = Math.Log(amp[i]);

        for (int i = 0; i < n; i++)
        {
            double coeff = 2.0 * freq[i] / Math.PI * df;
            double sumP = 0, sumV = 0;
            for (int j = 0; j < n; j++)
            {
                double denom = freq[j] * freq[j] - freq[i] * freq[i];
                if (j == i) denom = 1.0;
                sumP += (logamp[j] - logamp[i]) / denom;

                double denomu = amp[j] * denom;
                double numu = j == i ? 0.0 : 1.0;
                sumV += (numu / denomu) * (numu / denomu) * varamp[j];
            }
            phase[i] = coeff * sumP;
            varphase[i] = coeff * coeff * sumV;
        }
        return (phase, varphase);
    }

    public static DFTResult AmpPhaseToDft(double[] amp, double[] phase, double[] Uap)
    {
        int M = amp.Length;
        var x = new double[2 * M];
        var ux = new double[2 * M, 2 * M];

        for (int k = 0; k < M; k++)
        {
            double cp = Math.Cos(phase[k]), sp = Math.Sin(phase[k]);
            x[k] = amp[k] * cp;
            x[k + M] = amp[k] * sp;

            double va = Uap[k], vp = Uap[k + M];
            ux[k, k] = cp * cp * va + amp[k] * amp[k] * sp * sp * vp;
            ux[k, k + M] = cp * sp * va - amp[k] * amp[k] * sp * cp * vp;
            ux[k + M, k] = ux[k, k + M];
            ux[k + M, k + M] = sp * sp * va + amp[k] * amp[k] * cp * cp * vp;
        }
        return new DFTResult { F = x, UF = ux };
    }

    public static double[] RegularizationFilter(double[] freq, double fc, string filterType)
    {
        int M = freq.Length;
        var Hc = new Complex[M];

        for (int i = 0; i < M; i++)
        {
            Complex jf = new Complex(0, freq[i] / fc);
            switch (filterType)
            {
                case "LowPass":
                    Complex s = 1.0 + new Complex(0, freq[i] / (fc * 1.555));
                    Hc[i] = 1.0 / (s * s);
                    break;
                case "CriticalDamping":
                    Hc[i] = 1.0 / (1.0 + 1.28719 * jf + 0.41421 * jf * jf);
                    break;
                case "Bessel":
                    Hc[i] = 1.0 / (1.0 + 1.3617 * jf - 0.6180 * freq[i] * freq[i] / (fc * fc));
                    break;
                case "None":
                    Hc[i] = Complex.One;
                    break;
            }
        }

        if (filterType != "None")
        {
            int ind3dB = 0;
            double minDiff = 1e30;
            for (int i = 0; i < M; i++)
            {
                double diff = Math.Abs(Hc[i].Magnitude - Math.Sqrt(0.5));
                if (diff < minDiff) { minDiff = diff; ind3dB = i; }
            }
            if (ind3dB > 1)
            {
                double sumFAng = 0, sumFF = 0;
                for (int i = 0; i < ind3dB; i++)
                {
                    sumFAng += freq[i] * Hc[i].Phase;
                    sumFF += freq[i] * freq[i];
                }
                double w = sumFAng / sumFF;
                for (int i = 0; i < M; i++)
                    Hc[i] *= Complex.Exp(new Complex(0, -w * freq[i]));
            }
        }

        var filt = new double[2 * M];
        for (int i = 0; i < M; i++)
        {
            filt[i] = Hc[i].Real;
            filt[i + M] = Hc[i].Imaginary;
        }
        return filt;
    }

    public static SignalData ReadDatSignal(string filepath)
    {
        var lines = File.ReadAllLines(filepath);
        var values = new List<double>();
        foreach (var line in lines)
        {
            if (double.TryParse(line.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v))
                values.Add(v);
        }

        int nSamples = (int)values[0];
        double dt = values[1];
        var voltage = new double[nSamples];
        double mean = 0;
        for (int i = 0; i < nSamples; i++)
        {
            voltage[i] = values[4 + i];
            mean += voltage[i];
        }
        mean /= nSamples;
        for (int i = 0; i < nSamples; i++) voltage[i] -= mean;

        var time = new double[nSamples];
        for (int i = 0; i < nSamples; i++) time[i] = i * dt;

        return new SignalData { Time = time, Voltage = voltage, Dt = dt };
    }

    public static CalibrationData ReadCalibrationCsv(string filepath)
    {
        var lines = File.ReadAllLines(filepath);
        var freq = new List<double>();
        var realP = new List<double>();
        var imagP = new List<double>();
        var varR = new List<double>();
        var varI = new List<double>();
        var kov = new List<double>();

        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < 6) continue;
            freq.Add(double.Parse(parts[0].Trim(), CultureInfo.InvariantCulture) * 1e6);
            realP.Add(double.Parse(parts[1].Trim(), CultureInfo.InvariantCulture));
            imagP.Add(double.Parse(parts[2].Trim(), CultureInfo.InvariantCulture));
            varR.Add(double.Parse(parts[3].Trim(), CultureInfo.InvariantCulture));
            varI.Add(double.Parse(parts[4].Trim(), CultureInfo.InvariantCulture));
            kov.Add(double.Parse(parts[5].Trim(), CultureInfo.InvariantCulture));
        }

        return new CalibrationData
        {
            Frequency = freq.ToArray(), RealPart = realP.ToArray(),
            ImagPart = imagP.ToArray(), VarReal = varR.ToArray(),
            VarImag = varI.ToArray(), Kovar = kov.ToArray()
        };
    }

    static double[] Interp1(double[] xp, double[] fp, double[] x)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double xi = x[i];
            if (xi <= xp[0]) { result[i] = fp[0]; continue; }
            if (xi >= xp[^1]) { result[i] = fp[^1]; continue; }
            int idx = Array.BinarySearch(xp, xi);
            if (idx < 0) idx = ~idx;
            if (idx == 0) idx = 1;
            double t = (xi - xp[idx - 1]) / (xp[idx] - xp[idx - 1]);
            result[i] = fp[idx - 1] + t * (fp[idx] - fp[idx - 1]);
        }
        return result;
    }

    static int FindNearest(double[] arr, double val)
    {
        int idx = 0; double minD = Math.Abs(arr[0] - val);
        for (int i = 1; i < arr.Length; i++)
        {
            double d = Math.Abs(arr[i] - val);
            if (d < minD) { minD = d; idx = i; }
        }
        return idx;
    }

    public static PipelineResultData FullPipeline(string measurementFile, string noiseFile,
        string calFile, bool usebode, string filterType, double fc,
        double fmin = 1e6, double fmax = 60e6)
    {
        var sig = ReadDatSignal(measurementFile);
        int N = sig.Voltage.Length;

        // Noise uncertainty
        var noiseRaw = ReadDatSignal(noiseFile);
        // Re-read raw noise (without DC removal) to compute std
        var noiseLines = File.ReadAllLines(noiseFile);
        var nvals = new List<double>();
        foreach (var line in noiseLines)
            if (double.TryParse(line.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v))
                nvals.Add(v);
        int nn = (int)nvals[0];
        double nMean = 0;
        for (int i = 0; i < nn; i++) nMean += nvals[4 + i];
        nMean /= nn;
        double nVar = 0;
        for (int i = 0; i < nn; i++) nVar += (nvals[4 + i] - nMean) * (nvals[4 + i] - nMean);
        double stdev = Math.Sqrt(nVar / nn);

        var uxVec = new double[N];
        for (int i = 0; i < N; i++) uxVec[i] = stdev * stdev;

        var frequency = CalcFreqScale(sig.Time);
        int M = N / 2 + 1;
        var halfFreq = new double[M];
        Array.Copy(frequency, halfFreq, M);

        // GUM_DFT
        var dftResult = GumDft(sig.Voltage, uxVec);
        for (int i = 0; i < dftResult.F.Length; i++) dftResult.F[i] *= 2.0 / N;
        for (int i = 0; i < 2 * M; i++)
            for (int j = 0; j < 2 * M; j++)
                dftResult.UF[i, j] *= 4.0 / (N * N);

        var cal = ReadCalibrationCsv(calFile);
        int imin = FindNearest(cal.Frequency, fmin);
        int imax = FindNearest(cal.Frequency, fmax);
        int rangeN = imax - imin + 1;

        double[] x;
        double[,] ux;

        if (usebode)
        {
            var amp = new double[cal.Frequency.Length];
            var varamp = new double[cal.Frequency.Length];
            for (int i = 0; i < amp.Length; i++)
            {
                amp[i] = Math.Sqrt(cal.RealPart[i] * cal.RealPart[i] + cal.ImagPart[i] * cal.ImagPart[i]);
                varamp[i] = (cal.RealPart[i] / amp[i]) * (cal.RealPart[i] / amp[i]) * cal.VarReal[i]
                          + (cal.ImagPart[i] / amp[i]) * (cal.ImagPart[i] / amp[i]) * cal.VarImag[i]
                          + 2 * (cal.RealPart[i] / amp[i]) * (cal.ImagPart[i] / amp[i]) * cal.Kovar[i];
            }

            var freqTrim = new double[rangeN];
            var ampTrim = new double[rangeN];
            var varampTrim = new double[rangeN];
            Array.Copy(cal.Frequency, imin, freqTrim, 0, rangeN);
            Array.Copy(amp, imin, ampTrim, 0, rangeN);
            Array.Copy(varamp, imin, varampTrim, 0, rangeN);

            var ampip = Interp1(freqTrim, ampTrim, halfFreq);
            var varampip = Interp1(freqTrim, varampTrim, halfFreq);

            var (phaseip, varphaseip) = BodeEquation(halfFreq, ampip, varampip);

            var Uap = new double[2 * M];
            Array.Copy(varampip, 0, Uap, 0, M);
            Array.Copy(varphaseip, 0, Uap, M, M);

            var apResult = AmpPhaseToDft(ampip, phaseip, Uap);
            x = apResult.F;
            ux = apResult.UF;
        }
        else
        {
            var srcFreq = new double[rangeN];
            var srcReal = new double[rangeN];
            var srcImag = new double[rangeN];
            var srcVR = new double[rangeN];
            var srcVI = new double[rangeN];
            var srcKov = new double[rangeN];
            Array.Copy(cal.Frequency, imin, srcFreq, 0, rangeN);
            Array.Copy(cal.RealPart, imin, srcReal, 0, rangeN);
            Array.Copy(cal.ImagPart, imin, srcImag, 0, rangeN);
            Array.Copy(cal.VarReal, imin, srcVR, 0, rangeN);
            Array.Copy(cal.VarImag, imin, srcVI, 0, rangeN);
            Array.Copy(cal.Kovar, imin, srcKov, 0, rangeN);

            var realIp = Interp1(srcFreq, srcReal, halfFreq);
            var imagIp = Interp1(srcFreq, srcImag, halfFreq);
            var vrIp = Interp1(srcFreq, srcVR, halfFreq);
            var viIp = Interp1(srcFreq, srcVI, halfFreq);
            var kovIp = Interp1(srcFreq, srcKov, halfFreq);

            x = new double[2 * M];
            Array.Copy(realIp, 0, x, 0, M);
            Array.Copy(imagIp, 0, x, M, M);

            ux = new double[2 * M, 2 * M];
            for (int i = 0; i < M; i++)
            {
                ux[i, i] = vrIp[i];
                ux[i, i + M] = kovIp[i];
                ux[i + M, i] = kovIp[i];
                ux[i + M, i + M] = viIp[i];
            }
        }

        // DFT_deconv
        var deconvResult = DftDeconv(x, dftResult.F, ux, dftResult.UF);

        // Filter
        var filt = RegularizationFilter(halfFreq, fc, filterType);

        // DFT_multiply
        var regulResult = DftMultiply(deconvResult.F, filt, deconvResult.UF);

        // GUM_iDFT
        var (sigpRaw, UsigpRaw) = GumIdft(regulResult.F, regulResult.UF);
        int NSig = sigpRaw.Length;
        var sigp = new double[NSig];
        var Usigp = new double[NSig, NSig];
        for (int i = 0; i < NSig; i++) sigp[i] = sigpRaw[i] * NSig / 2.0;
        for (int i = 0; i < NSig; i++)
            for (int j = 0; j < NSig; j++)
                Usigp[i, j] = UsigpRaw[i, j] * NSig * NSig / 4.0;

        // Unfiltered deconvolution
        var (deconvRaw, _) = GumIdft(deconvResult.F, deconvResult.UF);
        var deconvTimeP = new double[deconvRaw.Length];
        for (int i = 0; i < deconvRaw.Length; i++)
            deconvTimeP[i] = deconvRaw[i] * deconvRaw.Length / 2.0;

        // Scaled signal
        var hydAmp = new double[M];
        var sigAmp = new double[M];
        for (int i = 0; i < M; i++)
        {
            hydAmp[i] = Math.Sqrt(x[i] * x[i] + x[i + M] * x[i + M]);
            sigAmp[i] = Math.Sqrt(dftResult.F[i] * dftResult.F[i] + dftResult.F[i + M] * dftResult.F[i + M]);
        }
        int iffun = 0;
        double maxSigAmp = sigAmp[0];
        for (int i = 1; i < M; i++)
            if (sigAmp[i] > maxSigAmp) { maxSigAmp = sigAmp[i]; iffun = i; }
        double hydempffun = hydAmp[iffun];
        var scaled = new double[N];
        for (int i = 0; i < N; i++) scaled[i] = sig.Voltage[i] / hydempffun;

        // Uncertainty
        var deltasigp = new double[NSig];
        for (int i = 0; i < NSig; i++) deltasigp[i] = Math.Sqrt(Usigp[i, i]);

        var pp = Deconvolution.PulseParameters(sig.Time, sigp, Usigp);

        return new PipelineResultData
        {
            Time = sig.Time, Scaled = scaled, Deconvolved = deconvTimeP,
            Regularized = sigp, Uncertainty = deltasigp, PulseParams = pp, NSamples = N
        };
    }
}
