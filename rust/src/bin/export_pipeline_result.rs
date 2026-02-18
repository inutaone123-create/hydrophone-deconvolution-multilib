// Hydrophone Deconvolution - Multi-language Implementation
//
// Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
// DOI: 10.5281/zenodo.10079801
// Original License: CC BY 4.0
//
// This implementation: 2024
// License: CC BY 4.0

use std::env;
use std::fs::File;
use std::io::Write;

use hydrophone_deconvolution::pipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!("Usage: export_pipeline_result <signal> <noise> <cal> <usebode> <filter> <fc> <outpath>");
        std::process::exit(1);
    }

    let signal_file = &args[1];
    let noise_file = &args[2];
    let cal_file = &args[3];
    let usebode = args[4] == "true";
    let filter_type = &args[5];
    let fc: f64 = args[6].parse()?;
    let outpath = &args[7];

    let result = pipeline::full_pipeline(signal_file, noise_file, cal_file, usebode, filter_type, fc)?;

    let mut f = File::create(outpath)?;
    writeln!(f, "# time;scaled;deconvolved;regularized;uncertainty(k=1)")?;
    for i in 0..result.n_samples {
        writeln!(f, "{:.18E};{:.18E};{:.18E};{:.18E};{:.18E}",
            result.time[i], result.scaled[i], result.deconvolved[i],
            result.regularized[i], result.uncertainty[i])?;
    }
    let pp = &result.pulse_params;
    writeln!(f, "# pc_value={:.18E};pc_uncertainty={:.18E};pc_time={:.18E};pr_value={:.18E};pr_uncertainty={:.18E};pr_time={:.18E};ppsi_value={:.18E};ppsi_uncertainty={:.18E}",
        pp.pc_value, pp.pc_uncertainty, pp.pc_time, pp.pr_value, pp.pr_uncertainty, pp.pr_time, pp.ppsi_value, pp.ppsi_uncertainty)?;

    println!("OK: {}", outpath);
    Ok(())
}
