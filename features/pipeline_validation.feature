@requires_reference_data
Feature: GUM Pipeline Validation
  As a developer
  I want the GUM-based deconvolution pipeline in all 5 languages
  to produce results matching the PyDynamic reference
  So that users can trust the complete pipeline implementation

  Scenario: Python pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the Python pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-9
    And the uncertainty should match the reference within 1e-12

  Scenario: Octave pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the Octave pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-9
    And the uncertainty should match the reference within 1e-12

  Scenario: C++ pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the C++ pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-9
    And the uncertainty should match the reference within 1e-12

  Scenario: C# pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the C# pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-6
    And the uncertainty should match the reference within 1e-12

  Scenario: Rust pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the Rust pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-9
    And the uncertainty should match the reference within 1e-12

  Scenario: 5-language pipeline cross-validation
    Given pipeline results from all 5 languages for MH44 M-Mode 3MHz LowPass Bode=true
    When I compare all language pairs
    Then the maximum regularized relative error should be less than 1e-6
    And the maximum uncertainty relative error should be less than 1e-6
