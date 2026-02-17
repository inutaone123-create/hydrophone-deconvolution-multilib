Feature: Pulse Parameters
  As a metrologist
  I want to calculate pulse parameters from deconvolved pressure
  So that I can characterize acoustic pulses with uncertainties

  Scenario: Calculate pulse parameters with scalar uncertainty
    Given a time array and pressure signal of length 256
    And a scalar uncertainty of 0.01
    When I calculate pulse parameters
    Then I should get valid pc, pr, and ppsi values
    And all uncertainties should be positive

  Scenario: Calculate pulse parameters with vector uncertainty
    Given a time array and pressure signal of length 256
    And a vector uncertainty of length 256
    When I calculate pulse parameters
    Then I should get valid pc, pr, and ppsi values
    And all uncertainties should be positive

  Scenario: Pulse parameters consistency across uncertainty types
    Given a time array and pressure signal of length 256
    And a uniform scalar uncertainty of 0.05
    When I calculate pulse parameters with scalar and vector inputs
    Then the results should be identical
