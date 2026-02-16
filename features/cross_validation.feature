Feature: Cross-Language Validation
  As a developer
  I want all language implementations to produce identical results
  So that users can trust any implementation

  Scenario: Python and Octave produce identical results
    Given the standard test signal
    When I deconvolve with Python
    And I deconvolve with Octave
    Then the relative error should be less than 1e-14

  Scenario: Python and C++ produce identical results
    Given the standard test signal
    When I deconvolve with Python
    And I deconvolve with C++
    Then the relative error should be less than 1e-14

  Scenario: Python and C# produce identical results
    Given the standard test signal
    When I deconvolve with Python
    And I deconvolve with C#
    Then the relative error should be less than 1e-14

  Scenario: Python and Rust produce identical results
    Given the standard test signal
    When I deconvolve with Python
    And I deconvolve with Rust
    Then the relative error should be less than 1e-14

  Scenario: All 10 pairwise comparisons pass
    Given results from all 5 languages
    When I compare all 10 pairs
    Then all relative errors should be less than 1e-14
