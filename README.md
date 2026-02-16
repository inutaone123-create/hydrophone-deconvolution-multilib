# Hydrophone Deconvolution Multi-Language Library

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dev Container](https://img.shields.io/badge/Dev%20Container-Ready-blue.svg)](https://code.visualstudio.com/docs/remote/containers)

Multi-language implementation of hydrophone measurement deconvolution with uncertainty propagation.

## Features

- **5 Language Implementations**: Python, Octave, C++, C#, Rust
- **PocketFFT Compatible**: Consistent FFT across all languages
- **Uncertainty Propagation**: GUM-compliant Monte Carlo method
- **Spec-Driven Development**: BDD with Gherkin features
- **Cross-Validated**: < 1e-14 relative error between languages
- **Dev Container Ready**: Instant development environment

## Quick Start

### Using Dev Container (Recommended)

```bash
# Open in VSCode
code .

# Reopen in Container
# F1 -> "Dev Containers: Reopen in Container"

# Inside container
claude code  # Start development
```

### Running Tests

```bash
# BDD tests
behave features/

# Language-specific tests
cd python && pytest tests/
cd octave && octave tests/test_deconvolution.m
cd cpp && mkdir build && cd build && cmake .. && make && ./test_deconvolution
cd csharp && dotnet test
cd rust && cargo test

# Cross-validation
python3 validation/compare_results.py
```

## Credits and Attribution

### Original Tutorial

This project is based on the hydrophone deconvolution tutorial:

**Authors**:
- Martin Weber (University of Helsinki)
  - ORCID: [0000-0001-5919-5808](https://orcid.org/0000-0001-5919-5808)
- Volker Wilkens (Physikalisch-Technische Bundesanstalt)
  - ORCID: [0000-0002-7815-1330](https://orcid.org/0000-0002-7815-1330)

**Publication**:
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (Version v1.4.1) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.10079801

**Original License**: CC BY 4.0

### This Implementation

Multi-language implementation and extensions by Inuta, 2024.

**Changes from Original**:
1. Refactored tutorial code into reusable library functions
2. Implemented in 5 languages with identical numerical behavior
3. Added PocketFFT alignment for cross-language consistency
4. Developed comprehensive BDD testing framework
5. Created cross-validation suite
6. Added Dev Container environment

## License

CC BY 4.0 - See [LICENSE](LICENSE) for full text.

## Citation

If you use this library in research, please cite:

**Original tutorial**:
```
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (v1.4.1).
Zenodo. https://doi.org/10.5281/zenodo.10079801
```

**This implementation**:
```
Inuta. (2024). Hydrophone Deconvolution Multi-Language Library.
GitHub. https://github.com/inuta/hydrophone-deconvolution-multilib
```
