"""Export a mesh and attitude history with a catalog sky and Earth/Moon limb.

Earth: a two-body GEO orbit with nadir-pointing attitude over 24 hours.
Moon: a prescribed one-minute circular arc and body spin. Neither is telemetry.
"""

from pathlib import Path

import numpy as np
from astropy.time import Time
from scipy.spatial.transform import Rotation
from ssapy import Orbit, get_body, rv

from ssatk.constants import EARTH_MU, MOON_MU, MOON_RADIUS, SIDEREAL_DAY_SECONDS
from ssatk.coordinates.frames import eci_to_ecf_matrix
from ssatk.plots.spacecraft_view import SpacecraftMesh, SpacecraftView
from ssatk.satellites import SpacecraftBody, Facet

TITLE = "Spacecraft Mesh, Attitude, and Sun Direction"
DESCRIPTION = (
    "A 45-second GEO chase-camera movie spanning 24 hours, with a nadir-pointing mesh, "
    "body axes, Sun direction, catalog stars and textured Earth. Includes a short lunar example."
)
GALLERY_CATEGORY = "six_dof"
GEO_SPAN_SECONDS = 86400.0
GEO_PLAYBACK_SECONDS = 45.0


def geo_states(count=540):
    """Return epochs, GCRF r/v and body-to-GCRF quaternions for a 24-hour GEO view.

    LLNL SSAPy's default Keplerian propagator samples a circular orbit at the
    radius implied by the mean sidereal period. Its plane is the terrestrial
    equator at the initial epoch, not the J2000 XY plane. The initial subpoint
    is Greenwich. Precession/nutation evolve afterwards, so this unperturbed
    illustration is nearly geostationary, not a station-keeping simulation.

    Body +X is along track, +Y opposite the orbit normal, +Z toward Earth.
    """
    epoch = Time("2026-10-11T12:00:00", scale="utc")
    times = Time(epoch.gps + np.linspace(0, GEO_SPAN_SECONDS, count), format="gps")
    radius = (EARTH_MU * (SIDEREAL_DAY_SECONDS / (2 * np.pi))**2)**(1 / 3)
    terrestrial_to_gcrf = eci_to_ecf_matrix(float(epoch.gps)).T
    radial = terrestrial_to_gcrf[:, 0]
    radial = radial / np.linalg.norm(radial)
    normal = terrestrial_to_gcrf[:, 2]
    normal = normal - np.dot(normal, radial) * radial
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    orbit = Orbit(radius * radial, np.sqrt(EARTH_MU / radius) * along, epoch, mu=EARTH_MU)
    r, v = rv(orbit, times)
    z = -r / np.linalg.norm(r, axis=1)[:, None]
    y = -np.cross(r, v)
    y /= np.linalg.norm(y, axis=1)[:, None]
    x = np.cross(y, z)
    q = Rotation.from_matrix(np.stack((x, y, z), axis=-1)).as_quat()[:, [3, 0, 1, 2]]
    return times, r, v, q


def example_mesh():
    """A 1.2 x 1.0 x 0.8 m bus with two 2.0 x 1.4 m subdivided panels."""
    body = SpacecraftBody.box(name="illustrative spacecraft", mass=100, size=(1.2, 1.0, 0.8))
    facets = list(body.facets)
    for sign in (-1, 1):
        # Cell boundaries are actual coplanar mesh edges, not a pasted image.
        for a in range(6):
            for b in range(3):
                x0, x1 = sign * (0.6 + a / 3), sign * (0.6 + (a + 1) / 3)
                y0, y1 = -0.7 + b * 1.4 / 3, -0.7 + (b + 1) * 1.4 / 3
                vertices = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
                if sign < 0:
                    vertices.reverse()
                facets.append(Facet(name=f"panel_{sign}_{a}_{b}", area=(1 / 3) * (1.4 / 3),
                                    normal_body=(0, 0, 1), vertices_body=vertices))
    return SpacecraftMesh.from_body(body.with_updates(facets=tuple(facets)))


def example_view(environment="earth", count=None, mesh=None, camera="limb"):
    """24-hour GEO chase view or one-minute lunar arc, with any viewing mesh.

    For alternatives use mesh="cubesat", mesh="boxwing", or mesh=example_mesh().
    """
    if environment == "earth":
        times, r, _, q = geo_states(540 if count is None else count)
        return SpacecraftView(r, times, q, mesh=mesh, camera=camera, environment="earth",
                              title="GEO chase camera / 24 hours / nadir pointing")
    if environment != "moon":
        raise ValueError("environment must be 'earth' or 'moon'")
    count = 25 if count is None else count
    times = Time(Time("2026-10-11T12:00:00", scale="utc").gps + np.linspace(0, 60, count), format="gps")
    radius = MOON_RADIUS + 3_000e3
    sun = get_body("sun")
    sun0 = np.asarray(sun.position(float(times.gps[0])))
    # A specified sunlit initial longitude makes the example body recognizable.
    # The view itself also supports eclipse/night-side input trajectories.
    phase0 = np.arctan2(sun0[1], sun0[0]) + 0.3
    phase = phase0 + np.sqrt(MOON_MU / radius**3) * (times.gps - times.gps[0])
    r = radius * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(count)))
    moon = get_body("moon")
    r += np.asarray(moon.position(times.gps)).T
    theta = np.deg2rad(np.linspace(25, 145, count))
    spin_axis = np.array([0.3, 0.5, 0.8])
    spin_axis /= np.linalg.norm(spin_axis)
    q = np.column_stack((np.cos(theta / 2), np.sin(theta / 2)[:, None] * spin_axis))
    return SpacecraftView(r, times, q, mesh=mesh, camera=camera, environment=environment,
                          title=f"{environment.title()} orbit / spacecraft attitude")


def main(make_figures=True, fast=False, output_dir=None):
    if not make_figures:
        return {"skipped": True, "reason": "figures_disabled"}
    if output_dir is None:
        from ssatk.plots.figpath import figpath
        output_dir = Path(figpath("six_dof/demo_spacecraft_view.png")).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # Sample the propagator at every encoded frame epoch. This avoids replacing
    # a full-day orbit with chords between a handful of trajectory samples.
    fps = 12 if fast else 24
    earth = example_view("earth", count=round(fps * GEO_PLAYBACK_SECONDS))
    paths.append(str(earth.save(output_dir / "spacecraft_earth.png")))
    paths.append(str(earth.save(output_dir / "spacecraft_earth.mp4", fps=fps,
                               duration_s=GEO_PLAYBACK_SECONDS,
                               width=960 if fast else 1280, height=640 if fast else 720)))
    moon = example_view("moon")
    paths.append(str(moon.save(output_dir / "spacecraft_moon.png")))
    paths.append(str(moon.save(output_dir / "spacecraft_moon.gif", width=720, height=480,
                              fps=8, duration_s=2 if fast else 6)))
    return {"title": TITLE, "files": paths}


if __name__ == "__main__":
    main()
