"""Export a mesh and attitude history with a catalog sky and Earth/Moon limb.

The orbit and spin are prescribed analytic examples, not observed trajectories.
"""

from pathlib import Path

import numpy as np
from astropy.time import Time
from ssapy import get_body

from ssatk.constants import EARTH_MU, EARTH_RADIUS, MOON_MU, MOON_RADIUS
from ssatk.plots.spacecraft_view import SpacecraftMesh, SpacecraftView
from ssatk.satellites import SpacecraftBody, Facet

TITLE = "Spacecraft Mesh, Attitude, and Sun Direction"
DESCRIPTION = "PNG/JPG and GIF/MP4 views of supplied trajectory and quaternion samples."
GALLERY_CATEGORY = "six_dof"


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


def example_view(environment="earth", count=25):
    """One-minute circular arc and prescribed 120-degree body spin, in GCRF."""
    times = Time(Time("2026-10-11T12:00:00", scale="utc").gps + np.linspace(0, 60, count), format="gps")
    earth = environment == "earth"
    radius = (EARTH_RADIUS + 20_200e3) if earth else (MOON_RADIUS + 3_000e3)
    mu = EARTH_MU if earth else MOON_MU
    sun = get_body("sun")
    sun0 = np.asarray(sun.position(float(times.gps[0])))
    # A specified sunlit initial longitude makes the example body recognizable.
    # The view itself also supports eclipse/night-side input trajectories.
    phase0 = np.arctan2(sun0[1], sun0[0]) + 0.3
    phase = phase0 + np.sqrt(mu / radius**3) * (times.gps - times.gps[0])
    r = radius * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(count)))
    if not earth:
        moon = get_body("moon")
        r += np.asarray(moon.position(times.gps)).T
    theta = np.deg2rad(np.linspace(25, 145, count))
    spin_axis = np.array([0.3, 0.5, 0.8])
    spin_axis /= np.linalg.norm(spin_axis)
    q = np.column_stack((np.cos(theta / 2), np.sin(theta / 2)[:, None] * spin_axis))
    return SpacecraftView(r, times, q, mesh=example_mesh(), environment=environment,
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
    for environment in ("earth", "moon"):
        view = example_view(environment)
        paths.append(str(view.save(output_dir / f"spacecraft_{environment}.png")))
        paths.append(str(view.save(output_dir / f"spacecraft_{environment}.gif",
                                   width=720, height=480, fps=8, duration_s=2 if fast else 6)))
    return {"title": TITLE, "files": paths}


if __name__ == "__main__":
    main()
