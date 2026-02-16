Feature: Uncertainty Propagation
  As a metrologist
  I want to propagate measurement uncertainties through deconvolution
  So that I can assess the reliability of my results

  Scenario: Monte Carlo uncertainty propagation
    Given a measured signal of length 1024
    And signal uncertainty values
    And a frequency response of length 1024
    And response uncertainty values
    And a sampling rate of 10000000 Hz
    When I perform deconvolution with 100 Monte Carlo samples
    Then I should get a mean deconvolved signal of length 1024
    And I should get uncertainty values of length 1024
    And all uncertainty values should be non-negative
