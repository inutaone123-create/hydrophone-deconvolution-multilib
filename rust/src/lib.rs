//! Hydrophone Deconvolution - Multi-language Implementation
//!
//! Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
//! DOI: 10.5281/zenodo.10079801
//! Original License: CC BY 4.0
//!
//! This implementation: 2024
//! License: CC BY 4.0

pub mod core;
pub mod pipeline;
pub use crate::core::{deconvolve_without_uncertainty, deconvolve_with_uncertainty, pulse_parameters, PulseParameters, PulseUncertainty};
