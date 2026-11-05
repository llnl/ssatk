"""Step-14 calibration against rocket motion and closed-form Gaussian regression."""

import numpy as np

from demos.launch.demo_launch_calibration import predict, reference_case, run_reference
from ssatk.launch import calibrate_launch_measurements


def _design(case):
    time = np.asarray(case["time_gps_s"])
    elapsed = np.r_[time[0]-.25, time]+.02-case["model"]["epoch_gps_s"]
    mass, flow = case["model"]["initial_mass_kg"], case["model"]["flow_kg_s"]
    response = np.log(mass/(mass-flow*elapsed))/flow
    return np.column_stack(((response[1:]+response[:-1])/2, np.ones(time.size)))


def test_rocket_calibration_and_held_out_covariance_against_gls(gps_epoch):
    """R1: thrust/bias within 2e-6 N and 2e-7 m/s; normalized covariance within 2e-5.

    Oracle is the constant-flow rocket equation and (X.T R^-1 X)^-1.
    The held-out 0.3 m/s offset must remain in residuals, not refit the sensor.
    """
    report = run_reference(epoch=gps_epoch)
    np.testing.assert_allclose(report["parameters"]["thrust_n"], 10., atol=2e-6, rtol=0)
    np.testing.assert_allclose(report["parameters"]["velocity_bias_m_s"], .04, atol=2e-7, rtol=0)
    training, held_out, failed = report["cases"]
    design = _design(training["input"])
    noise = np.asarray(training["input"]["covariance"])
    covariance = np.linalg.inv(design.T@np.linalg.solve(noise, design))
    sigma = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose((report["parameter_covariance"]-covariance)/np.outer(sigma, sigma),
                               0., atol=2e-5, rtol=0)
    design = _design(held_out["input"])
    predictive = held_out["input"]["covariance"]+design@covariance@design.T
    np.testing.assert_allclose(held_out["predictive_covariance"], predictive, atol=2e-9, rtol=0)
    # Standard-normal 97.5th percentile; interval width includes fitted-parameter uncertainty.
    expected_width = 2*1.959963984540054*np.sqrt(np.diag(predictive))
    np.testing.assert_allclose(np.subtract(held_out["interval_upper"], held_out["interval_lower"]),
                               expected_width, atol=2e-7, rtol=0)
    np.testing.assert_allclose(failed["residual"], .3, atol=2e-7, rtol=0)
    np.testing.assert_allclose(failed["bias"], .3, atol=2e-7, rtol=0)
    np.testing.assert_allclose([held_out["coverage"], failed["coverage"]], [1., 0.], atol=0., rtol=0)
    assert held_out["status"] == "PASS" and failed["status"] == "FAIL"
    assert report["status"] == "FAIL" and report["measurement_evidence_status"] == "UNAVAILABLE"


def test_unidentifiable_motor_scale_has_zero_fisher_information():
    """R1: product of two free thrust scales supplies only one independent sensitivity."""
    case = reference_case("calibration")

    def coupled(parameters, case):
        return predict({"thrust_n": parameters["gain"]*parameters["thrust_n"],
                        "velocity_bias_m_s": .04}, case)

    report = calibrate_launch_measurements(
        [case], [reference_case("held_out", epoch=1.4e9+10)], coupled,
        {"thrust_n": 10., "gain": 1.}, parameter_units=["N", "1"], parameter_scale=[10., 1.],
        parameter_bounds=([1., .1], [20., 2.]), coverage_probability=.95,
    )
    information = np.asarray(report["singular_values"])
    np.testing.assert_allclose(information[1]/information[0], 0., atol=1e-8, rtol=0)
    assert report["rank"] == 1 and report["parameter_covariance"] is None
    assert report["calibration_status"] == "FAIL" and report["validation_status"] == "UNAVAILABLE"
