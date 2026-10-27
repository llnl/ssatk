"""Compare Demo-2 launch initialization with the optional SpacePy backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy import units as units
from astropy.time import Time

from ssatk.launch.preflight import launch_site_state

GALLERY_INCLUDE = False
NASA_REFERENCE = "https://www.nasa.gov/image-article/demo-2-launch-setting-forth-historic-journey/"


def run():
    """Return a numerical preflight comparison, not a reconstructed flight."""
    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock

    state = launch_site_state("Kennedy Space Center LC-39A", "2020-05-30T19:22:00")
    positions = []
    fixed_position = None
    for offset in (-1, 0, 1):
        epoch = Time(state["epoch_utc"]) + offset * units.s
        coordinate = Coords(
            [[0.0, state["latitude_deg"], state["longitude_deg"]]],
            "GDZ", "sph", units=["km", "deg", "deg"], use_irbem=False,
        )
        coordinate.ticks = Ticktock([epoch.isot], "ISO")
        fixed = coordinate.convert("GEO", "car")
        inertial = coordinate.convert("ECI2000", "car")
        if offset == 0:
            fixed_position = fixed.data[0] * fixed.Re * 1000.0
            julian_day_quantum_s = float(np.spacing(coordinate.ticks.JD[0]) * 86400.0)
        positions.append(inertial.data[0] * inertial.Re * 1000.0)

    reference_velocity = (positions[2] - positions[0]) / 2.0
    speed = float(np.linalg.norm(state["gcrf_velocity_m_s"]))
    reference_speed = float(np.linalg.norm(reference_velocity))
    position_error = float(np.linalg.norm(fixed_position - state["itrf_position_m"]))
    speed_error = abs(speed - reference_speed)
    speed_budget = reference_speed * julian_day_quantum_s
    return {
        "status": "PASS" if position_error <= 1.0e-6 and speed_error <= speed_budget else "FAIL",
        "scope": "REAL-LAUNCH PREFLIGHT ONLY; NO ASCENT OR TELEMETRY MATCH",
        "nasa_reference": NASA_REFERENCE,
        "state": state,
        "spacepy_fixed_position_m": fixed_position.tolist(),
        "ssatk_speed_m_s": speed,
        "spacepy_speed_m_s": reference_speed,
        "fixed_position_error_m": position_error,
        "fixed_position_tolerance_m": 1.0e-6,
        "speed_error_m_s": speed_error,
        "speed_numerical_budget_m_s": speed_budget,
        "inertial_position_difference_m": float(np.linalg.norm(positions[1] - state["gcrf_position_m"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New HTML report in a data/output location.")
    arguments = parser.parse_args()
    if arguments.output.suffix.lower() != ".html":
        parser.error("--output must name an HTML file")
    result = run()
    state = result["state"]
    rows = "".join(
        f"<tr><td>{axis}</td><td>{actual:.3f}</td><td>{reference:.3f}</td></tr>"
        for axis, actual, reference in zip(
            ("Earth-fixed X (m)", "Earth-fixed Y (m)", "Earth-fixed Z (m)"),
            state["itrf_position_m"], result["spacepy_fixed_position_m"],
        )
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo-2 preflight comparison</title><style>
body{{font:17px system-ui;max-width:950px;margin:40px auto;padding:0 20px;line-height:1.6}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccc;text-align:left}}
.status{{font-size:24px;font-weight:bold}}aside{{background:#fff2cf;padding:16px}}
</style><h1>SpaceX Demo-2: preflight comparison</h1>
<p class="status">{result['status']} - launch initialization only</p>
<p>NASA records launch from LC-39A at 3:22 p.m. EDT on May 30, 2020.
This check uses {state['epoch_utc']} UTC, the published launch minute,
not a telemetry-derived subsecond timestamp.</p>
<p><a href="{NASA_REFERENCE}">NASA mission reference</a></p>
<table><tr><th>Quantity</th><th>SSATK</th><th>SpacePy</th></tr>{rows}
<tr><td>Inertial pad speed (m/s)</td><td>{result['ssatk_speed_m_s']:.6f}</td>
<td>{result['spacepy_speed_m_s']:.6f}</td></tr></table>
<h2>What passed?</h2>
<p>Independent WGS84 conversion of the same catalog latitude/longitude:
{result['fixed_position_error_m']:.3e} m difference; numerical tolerance 1 micrometre.</p>
<p>Earth-rotation speed: {result['speed_error_m_s']:.6f} m/s difference;
budget {result['speed_numerical_budget_m_s']:.6f} m/s from SpacePy's floating-point
Julian-day resolution and its one-second central-difference velocity.</p>
<aside><strong>Not a flight match.</strong> Catalog coordinates are not an independent
pad survey. SpacePy assumes UT1 = UTC and omits Earth-orientation corrections.
Its J2000 position differs from SSATK GCRF by {result['inertial_position_difference_m']:.3f} m;
this direction/frame difference is reported, not scored as agreement.
No ascent altitude, thrust, staging or telemetry was validated. GMAT was not run.</aside>
</html>"""
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(html)
    print(f"{result['status']}: {arguments.output}")
    return result


if __name__ == "__main__":
    main()
