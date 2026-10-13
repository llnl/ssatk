"""R1/C: the optical-navigation demo against known simulated truth."""

import numpy as np
import pytest
from ssatk.navigation.star_tracker import load_star_catalogue
from demos.navigation.demo_star_finder import simulate_star_finder, build_star_finder_figure


def test_star_finder_recovers_the_simulated_orbit_without_star_identity_inputs():
    # R1: SSATK DOP853 is compared with a separate closed-form circular orbit
    # over one revolution. Navigation fits are scored against prescribed truth,
    # not against themselves. The coarse ephemeris is offset by 100 km/50 m/s;
    # its small aberration residual remains even with zero centroid noise.
    result = simulate_star_finder(samples=9, noise_arcsec=0)
    assert result['propagation_error_m'] < .002
    frames = result['frames']
    assert all(f['correct'] == f['matched'] == 24 for f in frames)
    assert max(f['attitude_error_arcsec'] for f in frames) < .1
    assert max(f['position_error_m'] for f in frames) < 2
    assert all(sum(not used for used in f['fit']) == 6 for f in frames)
    assert len(result['observations']) == 24*9
    np.testing.assert_allclose(result['truth_r'][-1], result['truth_r'][0], atol=.002, rtol=0)
    # R1: filtered velocity is checked against the prescribed analytic circle,
    # after the coarse 50-m/s prior has been constrained by repeated fixes.
    # Nine widely spaced updates and a prescribed 100-m observation sigma do
    # not imply the same precision as the raw noise-free geometric fixes.
    final = result['filtered_states'][-1]
    np.testing.assert_allclose(final.x[:3], result['analytic_r'][-1], atol=20., rtol=0)
    np.testing.assert_allclose(final.x[3:], result['truth_v'][0], atol=.02, rtol=0)
    assert final.time == result['fixes'][-1].time
    # C: the public plot includes every supplied epoch, including both endpoints.
    figure = build_star_finder_figure(result)
    assert len(figure.frames) == 9
    assert frames[-1]['utc'] in figure.frames[-1].layout.title.text



def test_catalogue_loader_rejects_bad_units_and_missing_explicit_sources(tmp_path):
    # C: unit ranges and source selection cannot silently fabricate a sky.
    with pytest.raises(FileNotFoundError):
        load_star_catalogue(catalog_path=tmp_path/'missing.csv')
    path = tmp_path/'stars.csv'
    path.write_text('ra,dec,pmra,pmdec,mag\n0,0,0,0,1\n6,45,0,0,2\n25,0,0,0,3\n')
    with pytest.raises(ValueError, match='RA in'):
        load_star_catalogue(catalog_path=path)
    path.write_text('ra,dec,pmra,pmdec,mag\n0,0,0,0,1\n6,45,0,0,2\n12,0,nan,0,3\n')
    with pytest.raises(ValueError, match='nonfinite'):
        load_star_catalogue(catalog_path=path)
