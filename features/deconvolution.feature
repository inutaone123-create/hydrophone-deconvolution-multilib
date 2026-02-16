Feature: Hydrophone Deconvolution
  As a metrologist
  I want to deconvolve hydrophone measurements
  So that I can obtain the true acoustic pressure signal

  Scenario: Deconvolution without uncertainty
    Given a measured signal of length 1024
    And a frequency response of length 1024
    And a sampling rate of 10000000 Hz
    When I perform deconvolution without uncertainty
    Then the result should have length 1024
    And the result should be real-valued

  Scenario: Known signal recovery
    Given a known input signal
    And a known frequency response
    When I convolve and then deconvolve
    Then the recovered signal should match the original within 1e-10
