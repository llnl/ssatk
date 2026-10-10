"""Multi-band light curves of faceted satellite bodies, ground versus space.

Four bodies with per-component albedo are spun about a body axis and observed
over one hour in four bands. Each body gets a subplot; each band is a colour;
the ground observer is solid and the space-based observer dashed.

The bodies differ in the way geometry and material combine:

* a box-wing communications satellite, whose large dark solar arrays dominate
  the visible while the multi-layer insulation on the bus carries the SWIR
* a spent upper stage, modelled as a twelve-sided cylinder with end caps, whose
  flat-sided barrel gives the broad specular-free plateaus typical of a
  tumbling rocket body
* a 3U CubeSat with body-mounted cells, three orders of magnitude fainter
* a flat debris panel, which is invisible edge-on and flashes twice per
  rotation

Ground photometry runs through ``ssapy.EarthObserver`` and carries Kasten &
Young airmass extinction; the space-based curve passes an
``ssapy.OrbitalObserver`` position vector straight through, so it has no
extinction and a different range history.

Run as a script to write the figure; ``GALLERY_INCLUDE`` keeps it in the
gallery because it produces a useful artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import astropy.units as u
from astropy.time import Time

import ssapy
from ssatk.compute.faceted_magnitude import (
    faceted_light_curve,
    line_of_sight_blocked,
)

GALLERY_INCLUDE = True

BANDS = ("V", "R", "I", "SWIR")
BAND_COLORS = {"V": "#3b6ea5", "R": "#c0392b", "I": "#8e5aa8", "SWIR": "#d68910"}

# Diffuse reflectivity per band. Solar cells are dark and blue-suppressed;
# multi-layer insulation is strongly red/SWIR bright; aluminium and white paint
# are comparatively flat.
SOLAR_CELL = {"V": 0.04, "R": 0.06, "I": 0.12, "SWIR": 0.22}
MLI_GOLD = {"V": 0.22, "R": 0.45, "I": 0.62, "SWIR": 0.78}
WHITE_PAINT = {"V": 0.72, "R": 0.74, "I": 0.70, "SWIR": 0.42}
ALUMINIUM = {"V": 0.55, "R": 0.58, "I": 0.60, "SWIR": 0.65}


@dataclass(frozen=True)
class Facet:
    """Photometric facet with real polygon geometry.

    ``normal_body``, ``area``, and ``center_of_pressure`` are derived from
    ``vertices_body`` so they cannot drift out of agreement, and the vertex
    list is what both the renderer and the self-shadowing test consume.
    """

    vertices_body: np.ndarray
    normal_body: tuple[float, float, float]
    area: float
    center_of_pressure: tuple[float, float, float]
    diffuse_reflectivity: dict = field(default_factory=dict)


def facet_from_polygon(vertices, material, outward_hint):
    """Build a facet from a planar polygon, orienting it along ``outward_hint``."""

    vertices = np.asarray(vertices, dtype=float)
    normal = np.zeros(3)
    for index in range(len(vertices)):
        current, following = vertices[index], vertices[(index + 1) % len(vertices)]
        normal += np.cross(current, following)
    area = 0.5 * float(np.linalg.norm(normal))
    normal = normal / np.linalg.norm(normal)
    if float(np.dot(normal, np.asarray(outward_hint, dtype=float))) < 0.0:
        normal, vertices = -normal, vertices[::-1]
    return Facet(
        vertices_body=vertices,
        normal_body=tuple(float(component) for component in normal),
        area=area,
        center_of_pressure=tuple(float(component) for component in vertices.mean(axis=0)),
        diffuse_reflectivity=material,
    )


def _box_faces(half_x, half_y, half_z, material, offset=(0.0, 0.0, 0.0)):
    """Six outward-facing rectangles of a box centred on ``offset``."""

    offset = np.asarray(offset, dtype=float)
    corners = {
        (axis, sign): None for axis in range(3) for sign in (1.0, -1.0)
    }
    half = np.array([half_x, half_y, half_z], dtype=float)
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            other = [index for index in range(3) if index != axis]
            square = []
            for corner_a, corner_b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = np.zeros(3)
                point[axis] = sign * half[axis]
                point[other[0]] = corner_a * half[other[0]]
                point[other[1]] = corner_b * half[other[1]]
                square.append(point + offset)
            hint = np.zeros(3)
            hint[axis] = sign
            faces.append(facet_from_polygon(square, material, hint))
    del corners
    return faces


def box_wing_comsat():
    """2.5 m bus in MLI with two 4 m x 2 m solar wings on a +/-Y boom."""

    facets = _box_faces(1.25, 1.25, 1.25, MLI_GOLD)
    for sign in (1.0, -1.0):
        inner, outer = sign * 1.6, sign * 5.6
        panel = [
            np.array([-1.0, inner, 0.0]), np.array([1.0, inner, 0.0]),
            np.array([1.0, outer, 0.0]), np.array([-1.0, outer, 0.0]),
        ]
        facets.append(facet_from_polygon(panel, SOLAR_CELL, (0.0, 0.0, 1.0)))
        facets.append(facet_from_polygon(panel, MLI_GOLD, (0.0, 0.0, -1.0)))
    return facets


def spent_upper_stage(sides=12, radius=1.65, length=8.0):
    """Twelve-sided prism approximating a cylindrical stage with end caps."""

    half = 0.5 * length
    angles = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    facets = []
    for index in range(sides):
        first, second = ring[index], ring[(index + 1) % sides]
        quad = [
            np.array([first[0], first[1], -half]), np.array([second[0], second[1], -half]),
            np.array([second[0], second[1], half]), np.array([first[0], first[1], half]),
        ]
        hint = np.array([0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]), 0.0])
        facets.append(facet_from_polygon(quad, ALUMINIUM, hint))
    for sign in (1.0, -1.0):
        cap = [np.array([point[0], point[1], sign * half]) for point in ring]
        facets.append(facet_from_polygon(cap, WHITE_PAINT, (0.0, 0.0, sign)))
    return facets


def cubesat_3u():
    """0.1 x 0.1 x 0.34 m bus with body-mounted cells on the +/-X long faces."""

    facets = _box_faces(0.05, 0.05, 0.17, ALUMINIUM)
    cells = []
    for facet in facets:
        if abs(facet.normal_body[0]) > 0.5:
            cells.append(
                facet_from_polygon(facet.vertices_body, SOLAR_CELL, facet.normal_body)
            )
    return [facet for facet in facets if abs(facet.normal_body[0]) <= 0.5] + cells


def debris_panel():
    """A detached 1.5 m x 1.0 m panel, cells one side and insulation the other."""

    panel = [
        np.array([-0.75, -0.5, 0.0]), np.array([0.75, -0.5, 0.0]),
        np.array([0.75, 0.5, 0.0]), np.array([-0.75, 0.5, 0.0]),
    ]
    return [
        facet_from_polygon(panel, SOLAR_CELL, (0.0, 0.0, 1.0)),
        facet_from_polygon(panel, MLI_GOLD, (0.0, 0.0, -1.0)),
    ]


def spin_quaternions(times_s, period_s, axis=(0.0, 1.0, 0.0)):
    """Body-to-inertial quaternions for a constant spin about ``axis``."""

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angles = 2.0 * np.pi * np.asarray(times_s, dtype=float) / float(period_s)
    half = 0.5 * angles
    return np.column_stack([np.cos(half), *(np.sin(half) * component for component in axis)])


def _orbit_positions(orbit, times):
    """Return (N, 3) GCRF positions in metres for an SSAPy orbit."""

    position, _velocity = ssapy.rv(orbit, times)
    return np.atleast_2d(np.asarray(position, dtype=float))


def geostationary_over(observer, epoch, longitude_offset_deg=0.0):
    """Return a circular GEO orbit parked near the observer's longitude.

    Choosing Keplerian angles by hand puts a fixed GEO slot at an arbitrary
    longitude, which for most sites is permanently below the horizon.
    """

    time = Time(epoch)
    site = np.asarray(observer.getRV(time)[0], dtype=float).ravel()
    equatorial = np.array([site[0], site[1], 0.0])
    equatorial /= np.linalg.norm(equatorial)
    angle = np.radians(longitude_offset_deg)
    rotated = np.array([
        equatorial[0] * np.cos(angle) - equatorial[1] * np.sin(angle),
        equatorial[0] * np.sin(angle) + equatorial[1] * np.cos(angle),
        0.0,
    ])
    radius = 4.21644e7
    speed = np.sqrt(3.986004418e14 / radius)
    along_track = np.cross(np.array([0.0, 0.0, 1.0]), rotated)
    return ssapy.Orbit(radius * rotated, speed * along_track, time.gps)


def find_visibility_window(
    orbit, observer, platform, epoch, span_s, search_hours=24.0, step_s=30.0
):
    """Return the start offset and usable duration of the longest mutual pass.

    Both observers have to see the target for the comparison to mean anything.
    A low-orbit platform spends most of a low-orbit pass behind the Earth, so
    requiring only ground visibility silently produces an all-occulted
    space-based curve. The window is also trimmed to what the geometry
    supports, since a pass is often shorter than the span requested.
    """

    steps = int(search_hours * 3600.0 / step_s)
    offsets = np.arange(steps) * step_s
    times = Time(epoch) + offsets * u.s
    positions = _orbit_positions(orbit, times)
    site = np.atleast_2d(np.asarray(observer.getRV(times)[0], dtype=float))

    # Elevation above the local horizon, using the geocentric up direction.
    # A 15 degree mask is a realistic site constraint and avoids the grazing
    # geometry where extinction dominates.
    up = site / np.linalg.norm(site, axis=1)[:, None]
    line = positions - site
    line = line / np.linalg.norm(line, axis=1)[:, None]
    elevation = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", up, line), -1.0, 1.0)))
    platform_positions = _orbit_positions(platform, times)
    unblocked = np.array(
        [not line_of_sight_blocked(positions[i], platform_positions[i]) for i in range(steps)]
    )
    visible = (elevation >= 15.0) & unblocked
    best_start, best_length, run = None, 0, 0
    for index, flag in enumerate(visible):
        run = run + 1 if flag else 0
        if run > best_length:
            best_length, best_start = run, index - run + 1
    if best_start is None or best_length < 2:
        raise RuntimeError("no visibility window found in the search interval.")
    return float(offsets[best_start]), min(float(span_s), (best_length - 1) * step_s)


ROTATIONS_SHOWN = 8
SAMPLES_PER_ROTATION = 45
LIMITING_MAGNITUDE = 19.0


def build_cases(epoch, observer, platform, samples=None):
    """Return the four demo cases, each centred on a real visibility window.

    The plotted span is whichever is shorter: the pass itself, or enough time
    for ``ROTATIONS_SHOWN`` rotations. Sampling is tied to the spin period
    rather than to a fixed count, because an 11 s spin sampled on an 8 minute
    grid aliases into noise.
    """

    gps = Time(epoch).gps
    specs = [
        ("Box-wing comsat (GEO)", box_wing_comsat(),
         geostationary_over(observer, epoch, -5.0), 600.0, 3600.0),
        ("Spent upper stage (LEO)", spent_upper_stage(),
         ssapy.Orbit.fromKeplerianElements(7.0e6, 1.0e-3, 1.20, 0.0, 0.5, 0.0, gps), 24.0, 480.0),
        ("3U CubeSat (LEO)", cubesat_3u(),
         ssapy.Orbit.fromKeplerianElements(6.95e6, 1.0e-3, 1.71, 0.0, 2.0, 0.0, gps), 11.0, 480.0),
        ("Debris panel (MEO)", debris_panel(),
         ssapy.Orbit.fromKeplerianElements(2.0e7, 2.0e-3, 0.95, 0.0, 1.1, 0.7, gps), 42.0, 1800.0),
    ]

    cases = []
    for title, facets, orbit, spin_period, span_s in specs:
        start, usable_s = find_visibility_window(orbit, observer, platform, epoch, span_s)
        shown_s = min(usable_s, ROTATIONS_SHOWN * spin_period)
        count = int(round(SAMPLES_PER_ROTATION * shown_s / spin_period)) + 1
        offsets = np.linspace(0.0, shown_s, count)
        times = Time(epoch) + (start + offsets) * u.s
        cases.append((title, facets, orbit, spin_period, offsets, times))
    return cases


def _detectable(magnitudes, limit=None):
    """Blank samples fainter than the detection floor.

    An edge-on facet drives the scattering area to zero and the magnitude to
    several tens, which is arithmetically correct and physically meaningless.
    Masking keeps the axis on the range an instrument could actually record.
    """

    limit = LIMITING_MAGNITUDE if limit is None else limit
    values = np.asarray(magnitudes, dtype=float).copy()
    values[~np.isfinite(values) | (values > limit)] = np.nan
    return values


def draw_body(axis, facets, band="V"):
    """Render the facet geometry, shaded by that band's diffuse reflectivity."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    reflectivities = [facet.diffuse_reflectivity[band] for facet in facets]
    colormap = plt.get_cmap("copper")
    polygons = Poly3DCollection(
        [facet.vertices_body for facet in facets],
        facecolors=[colormap(min(1.0, value / 0.8)) for value in reflectivities],
        edgecolors="0.35",
        linewidths=0.35,
        alpha=0.97,
    )
    axis.add_collection3d(polygons)

    points = np.vstack([facet.vertices_body for facet in facets])
    extent = float(np.abs(points).max()) * 1.05
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1, 1, 1), zoom=1.45)
    axis.view_init(elev=22, azim=35)
    axis.set_axis_off()


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ssatk.plots import figsave

    epoch = "2026-06-09T06:00:00"
    ground = ssapy.EarthObserver(lon=-121.7, lat=37.7, elevation=200.0)
    # A mid-orbit platform: high enough to keep a low-orbit target in view for
    # most of a ground pass, where a co-altitude platform would be occulted.
    platform = ssapy.Orbit.fromKeplerianElements(
        2.0e7, 1.0e-3, 0.60, 0.0, 2.2, 1.0, Time(epoch).gps
    )

    cases = build_cases(epoch, ground, platform)
    # Body views sit above their curves so the figure stays close to square.
    figure = plt.figure(figsize=(12.5, 11.0))
    # Row 2 is an unused spacer. GridSpec applies one hspace to every row gap,
    # so without it the lower body captions land on the upper x-axis labels.
    grid = figure.add_gridspec(
        5, 2, height_ratios=(0.72, 2.15, 0.42, 0.72, 2.15), hspace=0.12, wspace=0.20
    )
    curve_axes, body_axes = [], []
    for row in (0, 3):
        for column in range(2):
            body_axes.append(figure.add_subplot(grid[row, column], projection="3d"))
            curve_axes.append(figure.add_subplot(grid[row + 1, column]))

    for axis, body_axis, (title, facets, orbit, spin_period, offsets, times) in zip(
        curve_axes, body_axes, cases
    ):
        draw_body(body_axis, facets)
        positions = _orbit_positions(orbit, times)
        quaternions = spin_quaternions(offsets, spin_period)
        platform_positions = _orbit_positions(platform, times)

        ground_curves = faceted_light_curve(
            positions, quaternions, facets, times, bands=BANDS, observer=ground
        )
        space_curves = faceted_light_curve(
            positions, quaternions, facets, times, bands=BANDS, observer=platform_positions
        )

        seconds = offsets
        for band in BANDS:
            color = BAND_COLORS[band]
            axis.plot(
                seconds, _detectable(ground_curves[band]["ab_mag_observed"]),
                color=color, linewidth=1.3,
            )
            axis.plot(
                seconds, _detectable(space_curves[band]["ab_mag_observed"]),
                color=color, linewidth=1.0, linestyle="--", alpha=0.85,
            )
        axis.set_xlim(seconds[0], seconds[-1])
        axis.invert_yaxis()
        # The caption belongs to the body view, so it reads title, geometry,
        # curve from top to bottom without crossing either axes.
        body_axis.set_title(
            f"{title}\n{len(facets)} facets, "
            f"{sum(facet.area for facet in facets):.2f} m$^2$, "
            f"{spin_period:g} s spin, {len(times)} samples",
            fontsize=9, pad=2.0,
        )
        axis.grid(alpha=0.25)
        axis.set_ylabel("AB magnitude")

    for axis in curve_axes:
        axis.set_xlabel("seconds from window start")

    # Colour carries the band and dash carries the observer, so six proxy
    # entries describe all eight curves.
    handles = [Line2D([], [], color=BAND_COLORS[band], lw=1.6, label=band) for band in BANDS]
    handles += [
        Line2D([], [], color="0.3", lw=1.6, linestyle="-", label="ground"),
        Line2D([], [], color="0.3", lw=1.3, linestyle="--", label="space"),
    ]
    curve_axes[0].legend(
        handles=handles, loc="lower left", ncol=3, fontsize=8,
        framealpha=0.85, borderpad=0.4, columnspacing=1.1, handlelength=1.8,
    )
    figure.suptitle(
        "Faceted multi-band light curves\n"
        "solid: ground observer with extinction   |   dashed: space-based observer"
        f"   |   detection floor {LIMITING_MAGNITUDE:g} mag",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.055)
    figsave(figure, "demo_faceted_light_curves.jpg")
    print("Saved via figsave: figures/demo_faceted_light_curves.jpg")


if __name__ == "__main__":
    main()
