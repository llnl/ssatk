"""Calibration and held-out checks for explicitly modelled launch observables."""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2, norm

from .preflight import collect_software_baseline

__all__ = ["calibrate_launch_measurements"]


def _prepare_case(case):
    # A JSON snapshot freezes the split, noise model and budgets before fitting.
    case = json.loads(json.dumps(case, allow_nan=False))
    for name in ("id", "flight_id", "observable", "unit", "independence", "model_source"):
        if not isinstance(case.get(name), str) or not case[name].strip():
            raise ValueError(f"case requires a nonempty {name}")
    if case.get("evidence") not in ("measured", "synthetic"):
        raise ValueError("case evidence must be measured or synthetic")
    time = np.asarray(case["time_gps_s"], dtype=float)
    observed = np.asarray(case["observed"], dtype=float)
    covariance = np.asarray(case["covariance"], dtype=float)
    if (time.ndim != 1 or not time.size or observed.shape != time.shape
            or not np.isfinite(time).all() or not np.isfinite(observed).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError("case needs increasing finite GPS times and one finite scalar observable per time")
    if covariance.shape != (time.size, time.size) or not np.isfinite(covariance).all():
        raise ValueError("covariance must cover all samples in squared observable units")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12*scale):
        raise ValueError("covariance must be symmetric")
    covariance = (covariance+covariance.T)/2
    whitener = np.linalg.cholesky(covariance)
    sensor = case["sensor"]
    for name in ("frame", "filter", "time_alignment"):
        if not isinstance(sensor.get(name), str) or not sensor[name].strip():
            raise ValueError(f"sensor requires {name}")
    position = np.asarray(sensor["position_body_m"], dtype=float)
    support = np.asarray(sensor["support_gps_s"], dtype=float)
    if (position.shape != (3,) or not np.isfinite(position).all()
            or support.shape != (2,) or not np.isfinite(support).all()
            or support[0] > time[0] or support[1] < time[-1]):
        raise ValueError("sensor requires a finite body position and full observation/filter time support")
    requirements = case["requirements"]
    for name in ("max_abs_residual", "max_abs_bias", "min_coverage"):
        value = requirements[name]
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if requirements["min_coverage"] > 1 or not requirements.get("source"):
        raise ValueError("requirements need coverage in [0, 1] and a declared source")
    missing = []
    for name in ("thrust", "mass_properties", "atmosphere", "telemetry"):
        evidence = case.get("inputs", {}).get(name)
        if evidence is None:
            missing.append(name)
            continue
        if not isinstance(evidence, Mapping) or not evidence.get("source") or not evidence.get("uncertainty"):
            raise ValueError(f"{name} input needs source and uncertainty information")
        digest = evidence.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{name} input needs a lowercase SHA-256 fingerprint")
    return case, observed, covariance, whitener, missing


def calibrate_launch_measurements(
    calibration_cases, validation_cases, predict, parameters0, *,
    parameter_units, parameter_scale, parameter_bounds, coverage_probability,
    rank_rtol=1e-8, max_nfev=None,
):
    """Fit named parameters on calibration cases, then freeze them for validation.

    ``predict(parameters, case)`` returns a scalar history in the case's units,
    frame and sample order. Reuse the launch propagator and a calibrated sensor
    operator, including its position, filtering and clock alignment. The callback
    receives copies of cases and a read-only parameter mapping. It must not use
    observed values to predict them. Angles must be consistently unwrapped.

    Cases are JSON mappings documented in ``docs/launch_baseline.rst``. Each
    covariance includes instrument/input uncertainty and temporal correlation;
    cases must be independent conditional on the fitted parameters. Overlapping
    support from the same flight cannot cross the calibration/validation split.
    Omitted input provenance is retained as unavailable measured evidence.

    Scales, bounds and units follow ``parameters0`` insertion order. Rank is
    assessed in these scaled coordinates with the declared relative SVD cutoff.
    Only an interior, converged, full-rank fit supplies formal covariance. It is
    inverse Fisher information, without residual-variance rescaling. Held-out
    predictive covariance adds parameter uncertainty to the supplied covariance.
    Gaussian marginal intervals and a joint chi-square check assume a locally
    linear model and known noise; observed coverage alone is not a confidence
    bound on population coverage. No flight qualification is inferred.
    """
    calibration = [_prepare_case(c) for c in calibration_cases]
    validation = [_prepare_case(c) for c in validation_cases]
    cases = calibration + validation
    if not calibration or not callable(predict):
        raise ValueError("at least one calibration case and a prediction callback are required")
    if len({c[0]["id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be unique across both splits")
    for training in calibration:
        for held_out in validation:
            a, b = training[0], held_out[0]
            if a["flight_id"] == b["flight_id"]:
                left, right = a["sensor"]["support_gps_s"], b["sensor"]["support_gps_s"]
                if max(left[0], right[0]) <= min(left[1], right[1]):
                    raise ValueError("calibration and held-out support overlap within a flight")
    names = tuple(parameters0)
    initial = np.asarray(list(parameters0.values()), dtype=float)
    scale = np.asarray(parameter_scale, dtype=float)
    bounds = np.asarray(parameter_bounds, dtype=float)
    units = tuple(parameter_units)
    size = len(names)
    if (not size or any(not isinstance(n, str) or not n.strip() for n in names)
            or initial.shape != (size,) or not np.isfinite(initial).all()
            or scale.shape != (size,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or bounds.shape != (2, size) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1]) or np.any(initial < bounds[0]) or np.any(initial > bounds[1])
            or len(units) != size or any(not isinstance(u, str) or not u.strip() for u in units)):
        raise ValueError("parameters need finite guesses, positive scales, finite ordered bounds and units")
    if not 0 < coverage_probability < 1 or not 0 < rank_rtol < 1:
        raise ValueError("coverage_probability and rank_rtol must lie strictly between zero and one")
    if sum(c[1].size for c in calibration) < size:
        raise ValueError("calibration has fewer observations than estimated parameters")
    lower, upper = bounds/scale

    def prediction(scaled, case):
        parameters = MappingProxyType(dict(zip(names, scaled*scale)))
        value = np.asarray(predict(parameters, deepcopy(case[0])), dtype=float)
        if value.shape != case[1].shape or not np.isfinite(value).all():
            raise ValueError(f"{case[0]['id']}: prediction must match the finite observed history")
        return value

    def residual(scaled):
        return np.concatenate([solve_triangular(c[3], c[1]-prediction(scaled, c), lower=True)
                               for c in calibration])

    fit = least_squares(residual, initial/scale, jac="3-point", bounds=(lower, upper), max_nfev=max_nfev)
    _, singular, vh = np.linalg.svd(fit.jac, full_matrices=False)
    rank = int(np.count_nonzero(singular > rank_rtol*singular[0]))
    interior = not np.any(fit.active_mask)
    usable = bool(fit.success and rank == size and interior)
    covariance_scaled = (vh.T / singular**2) @ vh if usable else None
    covariance = scale[:, None]*covariance_scaled*scale[None, :] if usable else None
    parameters = dict(zip(names, (fit.x*scale).tolist()))
    report = {
        "schema": "ssatk.launch_calibration.v1", "status": "UNAVAILABLE",
        "qualification_status": "NOT RUN", "software": collect_software_baseline(),
        "coverage_probability": float(coverage_probability), "rank_rtol": float(rank_rtol),
        "parameters": parameters, "parameter_names": list(names), "parameter_units": list(units),
        "parameter_scale": scale.tolist(), "parameter_bounds": bounds.tolist(),
        "parameters0": dict(zip(names, initial.tolist())), "max_nfev": max_nfev,
        "calibration_status": "PASS" if usable else "FAIL", "optimizer_success": bool(fit.success),
        "message": str(fit.message), "nfev": int(fit.nfev), "rank": rank,
        "singular_values": singular.tolist(), "active_mask": fit.active_mask.tolist(),
        "parameter_covariance": None if covariance is None else covariance.tolist(),
        "cases": [],
        "limits": ["Conditional independence between cases is supplied, not inferred",
                   "Local Gaussian uncertainty is conditional on the supplied model and covariance",
                   "Marginal coverage is descriptive; temporal correlation is included in the joint check",
                   "Synthetic cases cannot establish measured flight validation"],
    }
    quantile = norm.isf((1-coverage_probability)/2)
    for split, group in (("calibration", calibration), ("validation", validation)):
        for case in group:
            specification, observed, noise, whitener, missing = case
            record = {"id": specification["id"], "split": split, "input": specification,
                      "input_sha256": hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest(),
                      "missing_inputs": missing, "status": "UNAVAILABLE", "checks": {}}
            report["cases"].append(record)
            try:
                predicted = prediction(fit.x, case)
                difference = observed-predicted
                record.update(predicted=predicted.tolist(), residual=difference.tolist(),
                              whitened_residual=solve_triangular(whitener, difference, lower=True).tolist(),
                              bias=float(np.mean(difference)), rms=float(np.sqrt(np.mean(difference**2))),
                              max_abs_residual=float(np.max(np.abs(difference))))
                requirements = specification["requirements"]
                record["checks"] = {
                    "max_abs_residual": record["max_abs_residual"] <= requirements["max_abs_residual"],
                    "max_abs_bias": abs(record["bias"]) <= requirements["max_abs_bias"],
                }
                if split == "validation" and usable:
                    derivative = np.empty((observed.size, size))
                    for j in range(size):
                        step = min(np.cbrt(np.finfo(float).eps)*max(1., abs(fit.x[j])),
                                   (fit.x[j]-lower[j])/2, (upper[j]-fit.x[j])/2)
                        plus, minus = fit.x.copy(), fit.x.copy()
                        plus[j] += step
                        minus[j] -= step
                        if plus[j] == minus[j]:
                            raise ValueError("no representable interior sensitivity step")
                        derivative[:, j] = (prediction(plus, case)-prediction(minus, case))/(plus[j]-minus[j])
                    predictive = noise+derivative@covariance_scaled@derivative.T
                    sigma = np.sqrt(np.diag(predictive))
                    normalized = solve_triangular(np.linalg.cholesky(predictive), difference, lower=True)
                    coverage = float(np.mean(np.abs(difference) <= quantile*sigma))
                    joint = float(normalized@normalized)
                    joint_limit = float(chi2.ppf(coverage_probability, observed.size))
                    record.update(predictive_covariance=predictive.tolist(),
                                  interval_lower=(predicted-quantile*sigma).tolist(),
                                  interval_upper=(predicted+quantile*sigma).tolist(),
                                  coverage=coverage, joint_statistic=joint, joint_limit=joint_limit,
                                  bias_standard_error=float(np.sqrt(np.sum(predictive))/observed.size))
                    record["checks"].update(coverage=coverage >= requirements["min_coverage"],
                                             joint_consistency=joint <= joint_limit)
                if not all(record["checks"].values()):
                    record["status"] = "FAIL"
                elif split == "calibration" or usable:
                    record["status"] = "PASS"
            except Exception as error:
                # Retain failed held-out cases instead of dropping them from coverage.
                record.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    validation_records = [c for c in report["cases"] if c["split"] == "validation"]
    report["validation_status"] = next((s for s in ("ERROR", "FAIL", "UNAVAILABLE")
                                        if any(c["status"] == s for c in validation_records)),
                                       "PASS" if validation_records else "UNAVAILABLE")
    measured = bool(validation) and all(c[0]["evidence"] == "measured" and not c[4] for c in cases)
    report["measurement_evidence_status"] = "AVAILABLE" if measured else "UNAVAILABLE"
    statuses = {report["calibration_status"], report["validation_status"]} | {c["status"] for c in report["cases"]}
    report["status"] = next((s for s in ("ERROR", "FAIL", "UNAVAILABLE") if s in statuses),
                            "PASS" if measured else "UNAVAILABLE")
    return report
