"""Numerical references for solar scene coordinates and angular projection.

These check geometry, not rendered pixels. Figure/export contracts live in
scripts/tests. SSAPy supplies a separate DE ephemeris; ERFA supplies an
independent explicit IAU 2006 ecliptic rotation matrix.
"""

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import pytest
from ssapy import get_body

from ssatk.plots._solar_scene import body_position_au, ecliptic_vectors, project_sky


def test_ecliptic_axes_match_erfa_iau_2006_rotation():
    """R2: arbitrary ICRS vectors vs ERFA ecm06 at J2000, 4e-15 AU roundoff."""
    xyz = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1.2, -2., .7]])
    reference = xyz @ erfa.ecm06(2451545., 0.).T
    np.testing.assert_allclose(ecliptic_vectors(xyz), reference, rtol=0, atol=4e-15)


@pytest.mark.parametrize('origin,name', [('sun','earth'), ('earth','moon')])
def test_solar_positions_match_ssapy_de_ephemeris(origin, name):
    """R2: built-in ERFA/ELP vs SSAPy's shipped DE ephemeris at four 2025 epochs.

    The built-in ephemeris is approximate: allow 25 km Earth-Sun and 15 km
    Earth-Moon, covering ERFA epv00's km-scale and moon98's ~10 km errors.
    This is an accuracy bound against a different ephemeris, not exact parity.
    """
    times = Time(['2025-01-01','2025-04-01','2025-07-01','2025-10-01'])
    if name == 'earth':
        xyz = -get_body('sun').position(times).T
        bound_km = 25.
    else:
        xyz = get_body('moon').position(times).T
        bound_km = 15.
    reference = xyz @ erfa.ecm06(2451545., 0.).T / 1000
    actual = body_position_au(name, times, origin=origin) * u.au.to(u.km)
    error = np.linalg.norm(actual-reference, axis=1)
    assert np.max(error) < bound_km, error


def test_numeric_jd_and_solar_distances_use_simultaneous_geometry():
    """R2: AU separation against Astropy barycentric vectors, 2e-15 AU.

    Numeric JD denotes UTC midnight, with no former twelve-hour date offset.
    """
    epoch = Time('2025-06-01', scale='utc')
    reference = get_body_barycentric('earth', epoch) - get_body_barycentric('sun', epoch)
    actual = body_position_au('earth', epoch.jd)
    np.testing.assert_allclose(np.linalg.norm(actual), reference.norm().to_value(u.au), rtol=0, atol=2e-15)


@pytest.mark.parametrize('eye,center', [([0,0,1],[0,0,0]), ([0,0,100],[0,0,0]), ([5,9,11],[5,9,10])])
def test_infinite_sky_matches_pinhole_rays_without_parallax(eye, center):
    """R1: 45-degree vertical FOV, center and +/-11.25 degree rays; 1e-10 px.

    Moving the eye and target together, or zooming, cannot move infinite stars.
    """
    camera = dict(eye=dict(zip('xyz',eye)), center=dict(zip('xyz',center)), up=dict(x=0,y=1,z=0))
    tangent = np.tan(np.pi/16)
    rays = [[0,0,-1], [tangent,0,-1], [0,tangent,-1], [0,0,1]]
    xy, visible = project_sky(rays,camera,1000,600)
    offset = 300 * tangent / np.tan(np.pi/8)
    np.testing.assert_allclose(xy[:3], [[500,300],[500+offset,300],[500,300-offset]], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(visible, [True,True,True,False])


def test_infinite_sky_accepts_both_opposite_camera_directions():
    """R1: rotating 180 degrees exposes the opposite hemisphere, no missing sky."""
    rays = [[1,0,0],[-1,0,0]]
    for sign in [-1,1]:
        camera = dict(eye=dict(x=sign,y=0,z=0), up=dict(x=0,y=0,z=1))
        xy, visible = project_sky(rays,camera,800,600)
        np.testing.assert_array_equal(visible, [sign == -1,sign == 1])
        np.testing.assert_allclose(xy[visible], [[400,300]], rtol=0, atol=1e-12)
