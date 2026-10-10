from __future__ import annotations


import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.time import Time


def test_equatorial_ecliptic_round_trips_radians_and_degrees():
    from ssatk.coordinates import equatorial_ecliptic as eqecl

    ra_rad = np.deg2rad(132.5)
    dec_rad = np.deg2rad(-18.25)
    lon_rad, lat_rad = eqecl.equatorial_to_ecliptic(ra_rad, dec_rad)
    round_ra_rad, round_dec_rad = eqecl.ecliptic_to_equatorial(lon_rad, lat_rad)

    assert round_ra_rad == pytest.approx(ra_rad)
    assert round_dec_rad == pytest.approx(dec_rad)

    lon_deg, lat_deg = eqecl.equatorial_to_ecliptic(132.5, -18.25, degrees=True)
    round_ra_deg, round_dec_deg = eqecl.ecliptic_to_equatorial(lon_deg, lat_deg, degrees=True)

    assert round_ra_deg == pytest.approx(132.5)
    assert round_dec_deg == pytest.approx(-18.25)

    ecl_x, ecl_y, ecl_z = eqecl.equatorial_xyz_to_ecliptic_xyz(1.0, 2.0, 3.0)
    ra_from_xyz, dec_from_xyz = eqecl.ecliptic_xyz_to_equatorial(ecl_x, ecl_y, ecl_z)
    ra_direct, dec_direct = eqecl.xyz_to_equatorial(1.0, 2.0, 3.0)
    assert ra_from_xyz == pytest.approx(ra_direct)
    assert dec_from_xyz == pytest.approx(dec_direct)


def test_gcrf_to_itrf_astropy_is_geocentric_and_norm_preserving():
    from ssatk.coordinates.earth_fixed import gcrf_to_itrf_astropy

    times = Time(["2025-01-01T00:00:00", "2025-01-01T00:10:00"], scale="utc")
    positions = np.array([[0.0, 0.0, 0.0], [6_378_137.0, 0.0, 0.0]])

    transformed = gcrf_to_itrf_astropy(positions, times)

    assert transformed.shape == (2, 3)
    np.testing.assert_allclose(transformed[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.linalg.norm(transformed[1]) == pytest.approx(np.linalg.norm(positions[1]), rel=0, abs=1e-3)
    with pytest.raises(ValueError, match="shape"):
        gcrf_to_itrf_astropy(np.ones(3), times[0])


def test_frame_transform_conventions_match_ssapy_ntw_order():
    from ssatk.coordinates.satellite_frames import ntw_to_gcrf_matrix
    from ssatk.coordinates.frames import (
        Frame,
        FrameTransform,
        eci_to_ecf_matrix,
        eci_to_lon_lat,
        lvlh_axes,
        lvlh_matrix,
        ntw_axes,
        ntw_matrix,
    )

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])

    assert "velocity" in Frame.NTW.label
    rotation = eci_to_ecf_matrix(0.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    lvlh = lvlh_matrix(r, v)
    ntw = ntw_matrix(r, v)
    np.testing.assert_allclose(lvlh @ lvlh.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw @ ntw.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw, ntw_to_gcrf_matrix(r, v).T)

    T_hat, N_hat, W_hat = ntw_axes(r, v)
    R_hat, S_hat, W_lvlh = lvlh_axes(r, v)
    np.testing.assert_allclose(T_hat, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(N_hat, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(W_hat, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(R_hat, N_hat)
    np.testing.assert_allclose(S_hat, T_hat)
    np.testing.assert_allclose(W_lvlh, W_hat)

    ntw_tf = FrameTransform(Frame.NTW)
    np.testing.assert_allclose(ntw_tf.transform_point(r, v), [7000.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ntw_tf.transform_vector(v, r, v), [0.0, 7.5, 0.0], atol=1e-12)

    r_series = np.array([r, [0.0, 7000.0, 0.0]])
    v_series = np.array([v, [-7.5, 0.0, 0.0]])
    transformed = ntw_tf.transform_trajectory(r_series, v_series)
    np.testing.assert_allclose(transformed[:, 0], [7000.0, 7000.0], atol=1e-12)
    relative = ntw_tf.relative_trajectory(r_series, v_series)
    np.testing.assert_allclose(relative[0], [0.0, 0.0, 0.0], atol=1e-12)

    with pytest.raises(ValueError, match="t_gps required"):
        FrameTransform(Frame.ECF).transform_trajectory(r_series, v_series)
    lon, lat = eci_to_lon_lat(np.array([[7000.0, 0.0, 0.0]]), np.array([0.0]))
    assert lon.shape == lat.shape == (1,)
    assert np.isfinite(lon[0]) and np.isfinite(lat[0])


def test_satellite_burns_use_canonical_ntw_components(tmp_path):
    from ssatk.plots.orbit_state import OrbitalState, Trajectory
    from ssatk.plots.satellite import BurnEvent, Satellite3D

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    burn = BurnEvent(epoch_offset_s=10.0, dv_ntw_km_s=[0.0, 0.02, 0.0])

    np.testing.assert_allclose(burn.dv_eci(r, v), [0.0, 0.02, 0.0], atol=1e-12)
    assert burn.dv_mag_m_s == pytest.approx(20.0)
    assert burn.dv_mag_km_s == pytest.approx(0.02)
    assert burn.burn_duration_s() == 0.0

    finite = BurnEvent(
        epoch_offset_s=20.0,
        dv_ntw_km_s=[0.0, 0.01, 0.0],
        mode="finite",
        thrust_N=10.0,
        isp_s=300.0,
        mass_kg=100.0,
    )
    assert finite.burn_duration_s() > 0.0

    sat = Satellite3D(mass_kg=100.0)
    sat.add_burn(finite)
    sat.add_burn(burn)
    assert [item.epoch_offset_s for item in sat.burns] == [10.0, 20.0]
    assert sat.total_delta_v_m_s() == pytest.approx(30.0)

    T, N, W = sat.ntw_vectors(r, v, scale=2.0)
    R, S, W_lvlh = sat.lvlh_vectors(r, v, scale=3.0)
    np.testing.assert_allclose(T, [0.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(N, [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(W, [0.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(R, [3.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(S, [0.0, 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(W_lvlh, [0.0, 0.0, 3.0], atol=1e-12)
    assert np.linalg.norm(sat.burn_vector_eci(burn, r, v)) == pytest.approx(sat.ntw_scale * 0.2)
    sat.remove_burn(0)
    assert sat.burns == [finite]
    sat.add_burn(burn)

    obj_path = tmp_path / "cube.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    obj_sat = Satellite3D(model_path=obj_path)
    assert obj_sat.load_obj()
    assert obj_sat.faces == [[0, 1, 2]]
    vertices = obj_sat.model_vertices_eci(r, v, scale_km=1.0)
    assert vertices.shape == (3, 3)

    empty_sat = Satellite3D()
    assert empty_sat.model_vertices_eci(r, v, scale_km=1.0).shape == (8, 3)
    assert len(empty_sat.faces) == 6
    assert not Satellite3D(model_path=tmp_path / "missing.obj").load_obj()

    trajectory = Trajectory(
        r=np.array([r, r + np.array([0.0, 75.0, 0.0])]),
        v=np.array([v, v]),
        t=np.array([0.0, 10.0]),
    )
    state = OrbitalState.from_rv(r, v, epoch="2025-01-01T00:00:00+00:00", name="audit")
    results = sat.apply_burns_to_trajectory(trajectory, state)
    assert len(results) == 2
    assert results[0][1] == 1
    assert results[0][2] is burn


def test_orbital_state_public_quantities_have_physical_oracles():
    from ssatk.plots.orbit_state import MU, OrbitalState, PropagatorConfig

    state = OrbitalState(
        a_km=7000.0,
        e=0.001,
        inc_deg=45.0,
        raan_deg=10.0,
        argp_deg=20.0,
        nu_deg=30.0,
        epoch="2025-01-01T00:00:00+00:00",
        name="audit",
    )

    assert PropagatorConfig(propagator="rk4", gravity="j2", third_body="moon", non_grav="drag").label() == "RK4 + j2 + moon + drag"
    assert state.period_s == pytest.approx(2.0 * np.pi * np.sqrt(state.a_km**3 / MU))
    assert state.v_p > state.v_a
    assert state.v_circ == pytest.approx(np.sqrt(MU / state.a_km))
    assert state.specific_angular_momentum == pytest.approx(np.sqrt(MU * state.a_km * (1.0 - state.e**2)))
    assert state.regime == "LEO"
    assert state.j2_raan_drift_deg_day < 0.0
    assert np.isfinite(state.j2_argp_drift_deg_day)
    assert state.warnings == []

    r, v = state.to_rv()
    roundtrip = OrbitalState.from_rv(r, v, epoch=state.epoch)
    assert roundtrip.a_km == pytest.approx(state.a_km)
    assert roundtrip.e == pytest.approx(state.e)
    assert roundtrip.inc_deg == pytest.approx(state.inc_deg)
    assert state.osculating_ellipse(n_pts=12).shape == (12, 3)

    clone = state.clone(e=0.01, name="clone")
    assert clone.e == pytest.approx(0.01)
    assert clone.name == "clone"

    state.set_elements(a_km=7100.0)
    assert state.a_km == pytest.approx(7100.0)

    bad = OrbitalState(a_km=6000.0, e=-0.1, inc_deg=190.0)
    assert any("Negative eccentricity" in warning for warning in bad.warnings)
    assert any("Inclination" in warning for warning in bad.warnings)
    assert not bad.propagate(n_orbits=0.01, dt_s=10.0).ok

    callback_results = []
    traj = clone.propagate(n_orbits=0.01, dt_s=10.0, callback=callback_results.append)
    assert traj.ok
    assert callback_results == [traj]
    done = []
    thread, stop = clone.propagate_async(n_orbits=0.001, dt_s=10.0, on_done=done.append)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert not stop.is_set()
    assert done and done[0].ok

    with pytest.warns(UserWarning, match="two-body Keplerian"):
        cislunar = OrbitalState.from_preset("Cislunar Test Orbit")
    assert cislunar.name == "Orbit"
    with pytest.raises(KeyError):
        OrbitalState.from_preset("not a preset")


def test_orbital_state_tle_and_ssapy_roundtrip():
    pytest.importorskip("ssapy")
    from ssatk.plots.orbit_state import OrbitalState

    tle = """ISS (ZARYA)
1 25544U 98067A   25001.00000000  .00016717  00000+0  10270-3 0  9000
2 25544  51.6400 120.0000 0007000  90.0000  10.0000 15.50000000 00001
"""
    state = OrbitalState.from_tle(tle, name="ISS audit")
    assert state.name == "ISS audit"
    assert 6500.0 < state.a_km < 7000.0
    assert state.e == pytest.approx(0.0007)
    assert state.inc_deg == pytest.approx(51.64)

    ssapy_orbit = state.to_ssapy()
    converted = OrbitalState.from_ssapy(ssapy_orbit, name="converted")
    assert converted.name == "converted"
    assert converted.a_km == pytest.approx(state.a_km, rel=1e-6)
    assert converted.e == pytest.approx(state.e, abs=1e-8)


def test_sun_geometry_helpers_have_expected_directions_and_scaling():
    from ssatk.constants import SUN_EARTH_AVERAGE_DISTANCE_KM, SUN_RADIUS_KM
    from ssatk.plots import sun_mpl, sun_render, sun_view

    PHI, THETA = np.meshgrid(np.linspace(0.0, np.pi, 5), np.linspace(0.0, 2.0 * np.pi, 7))
    lit = sun_mpl.shade_texture(PHI, THETA, [0.0, 0.0, 1.0], ambient=0.2, diffuse=0.6)
    assert lit.min() >= 0.2
    assert lit.max() <= 0.8
    assert lit[0, 0] == pytest.approx(0.8)

    image = np.ones((*PHI.shape, 3))
    rows, cols = np.indices(PHI.shape)
    shaded = sun_mpl.apply_shading(image, rows, cols, PHI, THETA, [0.0, 0.0, 1.0])
    assert shaded.shape == image.shape
    assert np.all((0.0 <= shaded) & (shaded <= 1.0))

    assert sun_mpl.auto_sun_distance(100.0) == pytest.approx(42.0)
    assert sun_mpl.auto_sun_radius(100.0) == pytest.approx(4.5)
    np.testing.assert_allclose(
        sun_render.light_direction_from_positions([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        sun_render.light_direction_from_positions([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
    )
    sun_pos = sun_render.background_sun_position([0.0, 1.0, 0.0], 100.0, distance_factor=3.0)
    np.testing.assert_allclose(sun_pos, [0.0, 300.0, 0.0])
    assert sun_render.background_sun_radius(100.0, distance_factor=3.0) == pytest.approx(
        max(100.0 * 3.0 * SUN_RADIUS_KM / SUN_EARTH_AVERAGE_DISTANCE_KM, 1.0)
    )
    assert sun_render.background_sun_radius(100.0, size_factor=0.2) == pytest.approx(20.0)
    assert sun_view.auto_sun_position(np.array([1.0, 0.0, 0.0]), 100.0).shape == (3,)
    assert sun_view.auto_sun_radius(100.0) >= 1.0

    import datetime

    assert sun_view.jd_from_datetime(datetime.datetime(2000, 1, 1, 12, 0, 0)) == pytest.approx(2_451_545.0)
    with pytest.raises(TypeError, match="datetime"):
        sun_view.jd_from_datetime("2000-01-01")

    sun_position = sun_mpl.get_sun_position(Time([0.0], format="gps"))
    assert np.atleast_2d(sun_position).shape[-1] == 3
    sun_hat = sun_mpl.sun_direction_in_frame(
        Time([0.0, 60.0], format="gps"),
        transform_func=lambda pos, t: pos,
    )
    assert np.linalg.norm(sun_hat) == pytest.approx(1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sun_mpl.draw_sun(ax, [10.0, 0.0, 0.0], radius=1.0, n=8)
    sun_render.render_sun(ax, [0.0, 10.0, 0.0], radius=1.0, n=8, corona_layers_n=2, label="")
    assert ax.collections
    plt.close(fig)


def test_satellite_viewer_rotates_gcrf_state_to_teme(monkeypatch):
    from ssapy import utils
    from ssatk.plots import build_satellite_viewer as builder

    captured = {}
    monkeypatch.setattr(builder, "build", lambda **kwargs: captured.update(kwargs) or "viewer.html")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(
        utils,
        "gcrf_to_teme",
        lambda times: np.repeat(rotation[None, ...], len(np.atleast_1d(times)), axis=0),
    )
    builder.satellite_viewer(
        r=np.array([[7_000_000.0, 0.0, 0.0]]),
        v=np.array([[0.0, 7_500.0, 0.0]]),
        t=np.array([0.0]),
        verbose=False,
    )
    track = captured["state_vectors"][0]
    assert "q" not in track
    assert track["r"][0] == pytest.approx([0.0, 7_000.0, 0.0])
    assert track["v"][0] == pytest.approx([-7.5, 0.0, 0.0])


def test_sun_ra_dec_matches_astropy_solar_position():
    # R2: astropy's apparent geocentric Sun; the toolkit returns SSAPy's geometric
    # position, so the two differ by aberration (~20.5 arcsec) at most.
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.time import Time

    from ssatk.coordinates.sky import sun_ra_dec

    times = Time(["2025-03-20T09:01:00", "2025-06-21T02:42:00", "2026-10-08T00:00:00"], scale="utc")
    tolerance_rad = np.radians(30.0 / 3600.0)
    for t in times:
        sun = get_sun(t)
        for given in (t, t.isot, t.mjd):
            ra, dec = sun_ra_dec(given)
            assert abs(np.angle(np.exp(1j * (ra - sun.ra.to_value(u.rad))))) < tolerance_rad
            assert abs(dec - sun.dec.to_value(u.rad)) < tolerance_rad
    ra, dec = sun_ra_dec(times.mjd)
    assert np.shape(ra) == np.shape(dec) == (3,)


def test_satellite_viewer_rotates_body_to_gcrf_quaternions_into_teme():
    # R2: SSAPy's gcrf_to_teme applied to a known body-to-GCRF rotation. The
    # returned quaternion must rotate body vectors exactly as R_teme @ R_body.
    import importlib

    from astropy.time import Time
    from ssapy.utils import gcrf_to_teme

    from ssatk.coordinates.attitude import quaternion_from_matrix

    module = importlib.import_module("ssatk.plots.build_satellite_viewer")

    def rotate(q, x):
        w, qv = q[0], np.asarray(q[1:])
        return x + 2.0 * w * np.cross(qv, x) + 2.0 * np.cross(qv, np.cross(qv, x))

    rng = np.random.default_rng(7)
    body_to_gcrf, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(body_to_gcrf) < 0.0:
        body_to_gcrf[:, 0] *= -1.0
    epoch = Time("2026-10-07T00:00:00", scale="utc")
    _, _, _, q_out = module._gcrf_state_vectors_to_teme(
        np.array([[7.0e6, 0.0, 0.0]]),
        np.array([[0.0, 7.5e3, 0.0]]),
        np.array([epoch.gps]),
        q=quaternion_from_matrix(body_to_gcrf)[None, :],
    )

    body_vector = rng.normal(size=3)
    expected = np.asarray(gcrf_to_teme(epoch), dtype=float) @ (body_to_gcrf @ body_vector)
    np.testing.assert_allclose(rotate(q_out[0][0], body_vector), expected, atol=1e-12)


def test_gcrf_to_itrf_transforms_a_supplied_velocity_like_astropy():
    # R2: astropy GCRS -> ITRS with a CartesianDifferential. SSAPy's IAU 1976/80
    # rotation and astropy's IAU 2006/2000A agree to ~10 m in position here, so
    # the Earth-fixed velocities are held to 0.01 m/s.
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
    from astropy.time import Time

    from ssatk.coordinates import gcrf_to_itrf

    times = Time(["2026-10-08T00:00:00", "2026-10-08T06:00:00", "2026-03-20T12:00:00"], scale="utc")
    r = np.array([[7000e3, 1000e3, -500e3], [-4000e3, 5000e3, 3000e3], [42164e3, 0.0, 0.0]])
    v = np.array([[-1000.0, 7300.0, 1500.0], [-3000.0, -4000.0, 5000.0], [0.0, 3074.7, 0.0]])

    representation = CartesianRepresentation(r.T * u.m, differentials=CartesianDifferential(v.T * u.m / u.s))
    itrs = GCRS(representation, obstime=times).transform_to(ITRS(obstime=times))
    expected_v = itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T

    _r_itrf, v_itrf = gcrf_to_itrf(r, times.gps, v=v)
    np.testing.assert_allclose(v_itrf, expected_v, rtol=0, atol=0.01)
