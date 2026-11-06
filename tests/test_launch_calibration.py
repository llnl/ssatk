"""Step-14 calibration against rocket motion and closed-form Gaussian regression."""

import json

import numpy as np

from demos.launch.demo_launch_calibration import main, predict, reference_case, run_reference
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


def test_custom_dataset_recovers_rocket_thrust_and_sensor_bias(tmp_path):
    """R1: edited JSON recovers 7.5 N / -0.03 m/s within 2e-6 N / 2e-7 m/s.

    Observations independently use the rocket equation and circular lever-arm
    motion with changed mass, flow, spin, timing and unequal filter weights.
    """
    dataset_path, report_path = tmp_path/"sample.json", tmp_path/"report.json"
    main(["--write-sample", str(dataset_path)])
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["validation_cases"] = dataset["validation_cases"][:1]
    dataset["parameters0"] = {"thrust_n": 6., "velocity_bias_m_s": .01}
    for case in dataset["calibration_cases"]+dataset["validation_cases"]:
        model, sensor = case["model"], case["sensor"]
        model.update(initial_mass_kg=20., flow_kg_s=.2, spin_rad_s=.15)
        sensor.update(position_body_m=[.2, -.1, 0.], clock_offset_s=-.01,
                      first_sample_lead_s=.1, filter_weights=[.25, .75])
        tags = np.asarray(case["time_gps_s"])
        elapsed = np.r_[tags[0]-.1, tags]-.01-model["epoch_gps_s"]
        velocity = 7.5/.2*np.log(20./(20.-.2*elapsed))
        velocity -= .15*(.2*np.sin(.15*elapsed)-.1*np.cos(.15*elapsed))
        case["observed"] = (.25*velocity[:-1]+.75*velocity[1:]-.03).tolist()
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    main(["--dataset", str(dataset_path), "--output", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(report["parameters"]["thrust_n"], 7.5, atol=2e-6, rtol=0)
    np.testing.assert_allclose(report["parameters"]["velocity_bias_m_s"], -.03, atol=2e-7, rtol=0)
    for case in report["cases"]:
        np.testing.assert_allclose(case["residual"], 0., atol=2e-7, rtol=0)
    assert report["validation_status"] == "PASS" and report["status"] == "UNAVAILABLE"
