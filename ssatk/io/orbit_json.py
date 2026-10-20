"""Load the sampled-orbit JSON format used by the Moon WebGL viewer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import numpy as np


def _json_timing_metadata(config, parent=None):
    """Return optional sampled-track timing metadata in seconds."""
    sources = (config, parent or {})
    definitions = (
        (("duration_seconds", "period_seconds", "duration_s", "period_s",
          "durationSeconds", "periodSeconds"), 1.0, "duration_seconds"),
        (("duration_days", "period_days", "durationDays", "periodDays"),
         86400.0, "duration_seconds"),
        (("step_seconds", "sample_step_seconds", "time_step_seconds",
          "stepSeconds"), 1.0, "step_seconds"),
        (("stride_hours",), 3600.0, "step_seconds"),
        (("step_days", "sample_step_days", "stepDays"), 86400.0, "step_seconds"),
    )
    for source in sources:
        for keys, scale, output_key in definitions:
            for key in keys:
                if key not in source:
                    continue
                value = float(source[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"orbit JSON {key} must be a positive finite number"
                    )
                return {output_key: value * scale}
    return {}


def _json_times(value):
    """Normalize numeric GPS or ISO-UTC JSON times to GPS seconds."""
    values = np.asarray(value, dtype=object)
    flat = values.reshape(-1)
    if not len(flat):
        raise ValueError("orbit JSON time data cannot be empty")
    if all(isinstance(item, (int, float, np.integer, np.floating)) for item in flat):
        return np.asarray(flat, dtype=float).reshape(values.shape)
    try:
        from astropy.time import Time
    except ImportError as exc:  # pragma: no cover - SSAPy depends on Astropy
        raise ImportError("ISO orbit JSON times require Astropy") from exc
    parsed = Time(flat.tolist(), scale="utc").gps
    return np.asarray(parsed, dtype=float).reshape(values.shape)


def load_orbit_json(source):
    """Load sampled positions or Keplerian elements from a JSON object/file.

    Sampled JSON accepts ``r`` (or ``positions``), ``t`` (or ``times``), and
    optional ``r_frame``/``frame`` plus ``units`` (``"m"`` or ``"km"``).
    A set of sampled tracks accepts ``orbits``; each item may use the same
    fields, or the LLNL cislunar export's ``xyz`` field. ``xyz`` defaults to
    Moon-centered kilometres because that is the coordinate convention of the
    GDO cislunar catalogue. Optional period or step metadata drives the browser
    timeline when position timestamps are unavailable.

    Keplerian JSON accepts an ``elements`` object containing ``a``, ``e``,
    ``i``, ``pa``, ``raan`` and ``nu``. Angles are radians by default; set
    ``angle_units`` to ``"deg"``. ``a_units`` defaults to metres, ``mu`` to
    the canonical SSAPy Moon value, and ``r_frame`` to ``"moon_centered"``.
    """
    if isinstance(source, Mapping):
        payload = dict(source)
    elif isinstance(source, (str, bytes, os.PathLike)):
        path = os.path.expanduser(os.fspath(source))
        with open(path, encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    else:
        raise TypeError("orbit_json must be a mapping or JSON file path")
    if not isinstance(payload, Mapping):
        raise ValueError("orbit JSON root must be an object")

    nested = payload.get("orbit")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update({key: value for key, value in payload.items() if key != "orbit"})
        payload = merged

    def units_alias(config=None, default="auto"):
        aliases = {
            "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
            "km": "km", "kilometer": "km", "kilometers": "km",
            "kilometre": "km", "kilometres": "km", "auto": "auto",
        }
        config = payload if config is None else config
        value = str(config.get("units", config.get("position_units", default))).lower()
        if value not in aliases:
            raise ValueError("orbit JSON units must be 'm', 'km', or 'auto'")
        return aliases[value]

    records = payload.get("orbits")
    if records is not None:
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("orbit JSON 'orbits' must be a non-empty array")
        default_frame = str(payload.get("r_frame", payload.get("frame", "moon_centered")))
        default_units = units_alias(payload) if (
            "units" in payload or "position_units" in payload
        ) else None
        default_times = next(
            (payload[key] for key in ("t", "times", "time") if key in payload),
            None,
        )
        tracks = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise ValueError(f"orbit JSON orbits[{index - 1}] must be an object")
            positions_key = next(
                (key for key in ("r", "positions", "position", "xyz") if key in record),
                None,
            )
            if positions_key is None:
                raise ValueError(f"orbit JSON orbits[{index - 1}] has no positions/xyz field")
            frame = str(record.get("r_frame", record.get("frame", default_frame)))
            if frame == "moon_centered_earth_moon_rotating":
                frame = "moon_centered"
            units = units_alias(
                record,
                default=default_units or ("km" if positions_key == "xyz" else "auto"),
            )
            times = next(
                (record[key] for key in ("t", "times", "time") if key in record),
                default_times,
            )
            if frame != "moon_centered" and times is None:
                raise ValueError(f"orbit JSON orbits[{index - 1}] requires times for frame '{frame}'")
            track = {
                "r": record[positions_key],
                "t": None if times is None else _json_times(times),
                "r_frame": frame,
                "units": units,
                "name": str(record.get("name", record.get("oid", record.get("id", f"Orbit {index}")))),
            }
            track.update(_json_timing_metadata(record, payload))
            tracks.append(track)
        return {"tracks": tracks}

    positions = next((payload[key] for key in ("r", "positions", "position") if key in payload), None)
    if positions is not None:
        times = next((payload[key] for key in ("t", "times", "time") if key in payload), None)
        if times is None:
            raise ValueError("sampled orbit JSON requires t/times alongside r/positions")
        sampled = {
            "r": positions,
            "t": _json_times(times),
            "r_frame": str(payload.get("r_frame", payload.get("frame", "gcrf"))),
            "units": units_alias(payload),
            "name": str(payload.get("name", payload.get("oid", payload.get("id", "Orbit")))),
        }
        sampled.update(_json_timing_metadata(payload))
        return sampled

    elements = payload.get("elements", payload.get("keplerian", payload))
    if not isinstance(elements, Mapping) or "a" not in elements:
        raise ValueError("orbit JSON must contain sampled r/t data or Keplerian a/elements data")
    try:
        import ssapy
        from ssatk.constants import MOON_MU
    except ImportError as exc:  # pragma: no cover - SSAPy is a package dependency
        raise ImportError("Keplerian orbit JSON requires SSAPy") from exc

    def pick(*names, default=0.0):
        return next((elements[name] for name in names if name in elements), default)

    a_units = str(elements.get("a_units", payload.get("a_units", "m"))).lower()
    if a_units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        a = float(elements["a"]) * 1e3
    elif a_units in {"m", "meter", "meters", "metre", "metres"}:
        a = float(elements["a"])
    else:
        raise ValueError("Keplerian JSON a_units must be 'm' or 'km'")
    angle_units = str(elements.get("angle_units", payload.get("angle_units", "rad"))).lower()
    if angle_units in {"deg", "degree", "degrees"}:
        angle_scale = np.pi / 180.0
    elif angle_units in {"rad", "radian", "radians"}:
        angle_scale = 1.0
    else:
        raise ValueError("Keplerian JSON angle_units must be 'rad' or 'deg'")
    epoch = pick("t", "epoch", default=0.0)
    epoch_gps = float(np.asarray(_json_times(epoch), dtype=float).reshape(-1)[0])
    orbit = ssapy.Orbit.fromKeplerianElements(
        a, float(pick("e", default=0.0)),
        float(pick("i", "inclination", default=0.0)) * angle_scale,
        float(pick("pa", "argp", "argument_of_periapsis", default=0.0)) * angle_scale,
        float(pick("raan", default=0.0)) * angle_scale,
        float(pick("nu", "true_anomaly", "trueAnomaly", default=0.0)) * angle_scale,
        t=epoch_gps,
        mu=float(elements.get("mu", payload.get("mu", MOON_MU))),
    )
    return {
        "orbit": orbit,
        "r_frame": str(payload.get("r_frame", payload.get("frame", "moon_centered"))),
    }
