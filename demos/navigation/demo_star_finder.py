#!/usr/bin/env python3
"""Star matching, attitude determination, and landmark-only position fixes.

All measurements are simulated. The star table is real; mapped Earth
landmarks are an ideal WGS84 latitude/longitude grid with known identities.
Two rigidly calibrated cameras are assumed: a sky camera and a nadir camera.
No true attitude, star IDs, or positions enter the angle-fit solvers.
A coarse attitude and orbit prior are supplied for matching/aberration.
"""

from dataclasses import replace
from pathlib import Path
import json
import os
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.spatial.transform import Rotation

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.coordinates.frames import eci_to_ecf_matrix
from ssatk.navigation.star_tracker import (
    catalogue_directions, fit_star_attitude, load_star_catalogue,
    match_star_directions, position_from_landmark_bearings,
)
from ssatk.plots.figpath import figpath
from ssatk.propagators_orbit import propagate_orbit_state

GALLERY_CATEGORY = "navigation"
TITLE = "Star Finder — Angles-Only Attitude and Position"
DESCRIPTION = (
    "One LEO orbit of simulated camera measurements against a real star catalogue. "
    "Stars determine attitude; bearings to ideal mapped Earth landmarks determine "
    "position. Includes withheld-star checks, pixel/angle residuals, and CSV data."
)
ARCSEC = np.pi / (180*3600)


def _unit(v):
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def _radec(v):
    v = _unit(v)
    return np.column_stack([np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360,
                            np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))])


def simulate_star_finder(*, start="2025-06-01T00:00:00", duration_s=None,
                         samples=181, noise_arcsec=3., seed=20261012):
    """Simulate calibrated centroid/bearing measurements over a chosen span.

    Default span is one 700-km circular orbit, propagated by SSATK DOP853.
    The known coarse orbit has a 100-km position offset and 50-m/s velocity
    offset; only it supplies the observer correction to predicted star rays.
    Truth states are used to generate observations and score the estimates.
    Parallax/RV, optical distortion, pixel-image detection, clouds, landmark
    misidentification, and calibration drift are outside this simulation.
    """
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples < 3:
        raise ValueError("samples must be an integer >= 3")
    if not np.isfinite(noise_arcsec) or noise_arcsec < 0:
        raise ValueError("noise_arcsec must be finite and nonnegative")
    start = start if isinstance(start, Time) else Time(start, scale="utc")
    radius = EARTH_RADIUS+700e3
    period = 2*np.pi*np.sqrt(radius**3/EARTH_MU)
    duration = period if duration_s is None else float(duration_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be positive and finite")
    elapsed = np.linspace(0, duration, samples)
    epochs = start + elapsed*u.s
    ra, dec = np.radians([300., 45.])
    bore = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    x = _unit(np.cross([0, 0, 1], bore)); y = np.cross(bore, x)
    base = np.column_stack([x, y, bore])
    r0, v0 = radius*x, np.sqrt(EARTH_MU/radius)*y
    orbit = propagate_orbit_state(times=epochs.gps, r0=r0, v0=v0, t0=float(start.gps),
                                  rtol=1e-11, atol=1e-8)
    # Independent analytic circle checks the numerical propagation used here.
    phase = np.sqrt(EARTH_MU/radius**3)*elapsed
    analytic = radius*(np.cos(phase)[:, None]*x + np.sin(phase)[:, None]*y)
    propagation_error_m = float(np.linalg.norm(orbit.r-analytic, axis=1).max())
    catalogue = load_star_catalogue(mag_limit=6.5)
    all_count = len(catalogue.row_ids)
    initial = catalogue_directions(catalogue, start)
    select = initial @ bore > np.cos(np.radians(30))
    catalogue = replace(catalogue, **{name: getattr(catalogue, name)[select] for name in (
        "row_ids", "names", "ra_hours", "dec_deg", "pmra_mas_yr", "pmdec_mas_yr", "magnitude")})
    lat, lon = np.meshgrid(np.arange(-88., 89., 4.), np.arange(-180., 180., 4.))
    mapped = EarthLocation.from_geodetic(lon.ravel()*u.deg, lat.ravel()*u.deg)
    ecef = np.array([part.to_value(u.m) for part in mapped.geocentric]).T
    normals = _unit(ecef / np.array([6378137.**2, 6378137.**2, 6356752.314245**2]))
    focal = 1024/np.tan(np.radians(15))  # 2048 x 2048, 30-degree square FOV
    rng = np.random.default_rng(seed)
    frames, observations, nav = [], [], []
    prior_offset = np.array([100e3, 0., 0.])
    velocity_offset = np.array([0., 50., 0.])
    for i, t in enumerate(epochs):
        angles = [1.5*np.sin(phase[i]), 1.0*np.cos(phase[i]), 18*np.sin(phase[i]/2)]
        attitude = base @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        prior = attitude @ Rotation.from_rotvec(np.radians([.06, -.04, .08])).as_matrix()
        truth_dirs = catalogue_directions(catalogue, t, observer_position_m=orbit.r[i], observer_velocity_m_s=orbit.v[i])
        predicted = catalogue_directions(catalogue, t, observer_position_m=orbit.r[i]+prior_offset,
                                         observer_velocity_m_s=orbit.v[i]+velocity_offset)
        body = truth_dirs @ attitude
        xy = body[:, :2]/body[:, 2, None]
        visible = (body[:, 2] > 0) & (np.max(np.abs(xy), axis=1) < np.tan(np.radians(14.8)))
        ids = np.flatnonzero(visible)
        ids = ids[np.argsort(catalogue.magnitude[ids])[:24]]
        if len(ids) < 8:
            raise RuntimeError("fewer than eight usable stars in the chosen field")
        # A calibrated centroid model, not fake raw-image detection. Shuffle
        # detections so catalogue correspondence is inferred by the matcher.
        measured_xy = xy[ids] + rng.normal(0, noise_arcsec*ARCSEC, (len(ids), 2))
        order = rng.permutation(len(ids))
        measured_xy, truth_ids = measured_xy[order], ids[order]
        body_measured = _unit(np.column_stack([measured_xy, np.ones(len(ids))]))
        detected, found = match_star_directions(body_measured, predicted, prior, gate_rad=np.radians(.25))
        fit = np.arange(len(detected)) % 4 != 0  # one quarter withheld from fitting
        if fit.sum() < 5:
            raise RuntimeError("insufficient unambiguous matches for attitude fitting")
        estimate = fit_star_attitude(body_measured[detected[fit]], predicted[found[fit]])
        fitted_xy3 = predicted[found] @ estimate
        fitted_xy = fitted_xy3[:, :2]/fitted_xy3[:, 2, None]
        residual = (measured_xy[detected]-fitted_xy)/ARCSEC
        recovered = body_measured[detected] @ estimate.T
        angle_error = np.arctan2(np.linalg.norm(np.cross(recovered, predicted[found]), axis=1),
                                 np.sum(recovered*predicted[found], axis=1))/ARCSEC
        attitude_error = np.linalg.norm(Rotation.from_matrix(estimate @ attitude.T).as_rotvec())/ARCSEC
        # Finite landmarks provide translation information; the sky does not.
        # Nadir camera has an ideal known rigid calibration to the star camera.
        rotation = eci_to_ecf_matrix(float(t.gps))
        landmarks = ecef @ rotation
        normal = normals @ rotation
        line = landmarks-orbit.r[i]
        keep = np.einsum("ij,ij->i", normal, -line) > 0
        nadir = -_unit(orbit.r[i])
        keep &= _unit(line) @ nadir > np.cos(np.radians(40))
        local = landmarks[keep]
        bearings_body = _unit(local-orbit.r[i]) @ attitude
        # Tangent-plane angular Gaussian noise, no range observations.
        tangent = rng.normal(size=bearings_body.shape)
        tangent -= (tangent*bearings_body).sum(axis=1)[:, None]*bearings_body
        bearings_body = _unit(bearings_body + noise_arcsec*ARCSEC*tangent)
        position = position_from_landmark_bearings(local, bearings_body @ estimate.T)
        position_error = float(np.linalg.norm(position-orbit.r[i]))
        astrometric = _radec(catalogue_directions(catalogue, t)[found])
        apparent, measured_ra_dec = _radec(predicted[found]), _radec(recovered)
        names = catalogue.names[found].tolist()
        correct = catalogue.row_ids[found] == catalogue.row_ids[truth_ids[detected]]
        for j in range(len(found)):
            observations.append(dict(utc=t.utc.isot, elapsed_s=float(elapsed[i]),
                catalogue_row=int(catalogue.row_ids[found[j]]), name=names[j],
                magnitude=float(catalogue.magnitude[found[j]]), used_for_fit=bool(fit[j]),
                correctly_identified=bool(correct[j]),
                astrometric_ra_deg=astrometric[j, 0], astrometric_dec_deg=astrometric[j, 1],
                predicted_apparent_ra_deg=apparent[j, 0], predicted_apparent_dec_deg=apparent[j, 1],
                measured_ra_deg=measured_ra_dec[j, 0], measured_dec_deg=measured_ra_dec[j, 1],
                measured_x_px=1024+focal*measured_xy[detected[j], 0],
                measured_y_px=1024+focal*measured_xy[detected[j], 1],
                predicted_x_px=1024+focal*fitted_xy[j, 0], predicted_y_px=1024+focal*fitted_xy[j, 1],
                residual_x_arcsec=residual[j, 0], residual_y_arcsec=residual[j, 1],
                angular_residual_arcsec=angle_error[j]))
        frames.append(dict(utc=t.utc.isot, names=names, fit=fit.tolist(),
            observed=(1024+focal*measured_xy[detected]).tolist(), predicted=(1024+focal*fitted_xy).tolist(),
            residual=residual.tolist(), recovered_radec=measured_ra_dec.tolist(), catalogue_radec=apparent.tolist(),
            position=position.tolist(), attitude_error_arcsec=float(attitude_error),
            position_error_m=position_error, heldout_rms_arcsec=float(np.sqrt(np.mean(angle_error[~fit]**2))),
            matched=len(found), correct=int(correct.sum()), landmarks=len(local)))
        nav.append(dict(utc=t.utc.isot, elapsed_s=elapsed[i], attitude_error_arcsec=attitude_error,
            position_error_m=position_error, heldout_rms_arcsec=frames[-1]['heldout_rms_arcsec'],
            **{f'true_{axis}_m': orbit.r[i, k] for k, axis in enumerate('xyz')},
            **{f'estimated_{axis}_m': position[k] for k, axis in enumerate('xyz')}))
    return dict(frames=frames, observations=observations, navigation=nav, times_s=elapsed,
        truth_r=orbit.r, analytic_r=analytic, source=catalogue.source, source_sha256=catalogue.sha256,
        catalogue_selected=all_count, period_s=float(period), duration_s=duration,
        propagation_error_m=propagation_error_m, noise_arcsec=noise_arcsec,
        prior_position_error_m=float(np.linalg.norm(prior_offset)), prior_velocity_error_m_s=50.,
        seed=seed)


def build_star_finder_figure(result):
    """Interactive camera, residual, orbit and error-history panels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2,
        specs=[[{}, {}], [{"type": "scene"}, {"secondary_y": True}]],
        subplot_titles=("Camera: catalogue prediction and measured centroids", "Star residuals (withheld stars were not fitted)",
                        "True orbit and angles-only position fixes", "Error against simulation truth"),
        vertical_spacing=.16, horizontal_spacing=.12)
    frames = result['frames']; f = frames[0]
    def star_traces(frame):
        measured, predicted = np.array(frame['observed']), np.array(frame['predicted'])
        residual, fit = np.array(frame['residual']), np.array(frame['fit'])
        details = np.column_stack([frame['names'], np.round(frame['catalogue_radec'], 6), np.round(frame['recovered_radec'], 6)])
        return [
            go.Scatter(x=predicted[:, 0], y=predicted[:, 1], mode='markers', name='Catalogue prediction',
                marker=dict(symbol='circle-open', color='#69e5cc', size=13), text=frame['names'], hovertemplate='%{text}<extra>Catalogue prediction</extra>'),
            go.Scatter(x=measured[:, 0], y=measured[:, 1], mode='markers', name='Measured stars',
                marker=dict(color='#ffb85c', size=5), customdata=details,
                hovertemplate='%{customdata[0]}<br>Predicted RA/Dec: %{customdata[1]}°, %{customdata[2]}°<br>Measured RA/Dec: %{customdata[3]}°, %{customdata[4]}°<extra></extra>'),
            go.Scatter(x=residual[fit, 0], y=residual[fit, 1], mode='markers', name='Fitted stars',
                marker=dict(color='#73bfff', size=7), text=np.array(frame['names'])[fit]),
            go.Scatter(x=residual[~fit, 0], y=residual[~fit, 1], mode='markers', name='Withheld stars',
                marker=dict(color='#ffb85c', symbol='diamond', size=9), text=np.array(frame['names'])[~fit])]
    traces = star_traces(f)
    for trace in traces[:2]: fig.add_trace(trace, row=1, col=1)
    for trace in traces[2:]: fig.add_trace(trace, row=1, col=2)
    truth = result['truth_r']/1000
    estimates = np.array([frame['position'] for frame in frames])/1000
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 48), np.linspace(0, np.pi, 24))
    rad = EARTH_RADIUS/1000
    fig.add_trace(go.Surface(x=rad*np.sin(phi)*np.cos(theta), y=rad*np.sin(phi)*np.sin(theta), z=rad*np.cos(phi),
        colorscale=[[0, '#132d49'], [1, '#2c6890']], showscale=False, opacity=.85, hoverinfo='skip', name='Earth'), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2], mode='lines', line=dict(color='#69e5cc', width=3), name='True orbit'), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=estimates[:, 0], y=estimates[:, 1], z=estimates[:, 2], mode='markers', marker=dict(size=2, color='#ffb85c'), name='Landmark fixes'), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=[truth[0, 0]], y=[truth[0, 1]], z=[truth[0, 2]], mode='markers', marker=dict(size=6, color='white'), name='Spacecraft'), row=2, col=1)
    mins = result['times_s']/60
    fig.add_trace(go.Scatter(x=mins, y=[q['attitude_error_arcsec'] for q in frames], mode='lines', line=dict(color='#73bfff'), name='Attitude error (arcsec)'), row=2, col=2)
    fig.add_trace(go.Scatter(x=mins, y=[q['position_error_m'] for q in frames], mode='lines', line=dict(color='#ffb85c'), name='Position error (m)'), row=2, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=mins, y=[q['heldout_rms_arcsec'] for q in frames], mode='lines', line=dict(color='#69e5cc', dash='dot'), name='Withheld-star RMS (arcsec)'), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, max(q['attitude_error_arcsec'] for q in frames)*1.2], mode='lines', line=dict(color='white', dash='dot'), showlegend=False, hoverinfo='skip'), row=2, col=2)
    def title(frame):
        return (f"Star Finder · {frame['utc']} UTC · {frame['matched']} stars / {frame['landmarks']} landmarks"
                "<br><sup>Simulated measurements · real star catalogue · stars → attitude; mapped landmark bearings → position</sup>")
    animation = []
    for i, frame in enumerate(frames):
        data = star_traces(frame)
        data += [go.Scatter3d(x=[truth[i, 0]], y=[truth[i, 1]], z=[truth[i, 2]]),
                 go.Scatter(x=[mins[i], mins[i]])]
        animation.append(go.Frame(name=str(i), data=data, traces=[0, 1, 2, 3, 7, 11], layout=dict(title=dict(text=title(frame)))))
    fig.frames = animation
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0a1220', plot_bgcolor='#0a1220',
        height=1050, width=1500, title=dict(text=title(f), x=.5, font=dict(size=19)),
        margin=dict(t=130, b=140, l=80, r=90), legend=dict(orientation='h', y=-.13, x=0),
        font=dict(family='Arial', color='#dbe9f7'),
        scene=dict(aspectmode='data', xaxis_title='GCRF x (km)', yaxis_title='y (km)', zaxis_title='z (km)',
                   camera=dict(eye=dict(x=1.4, y=1.4, z=.8))),
        updatemenus=[dict(type='buttons', direction='left', x=0, y=-.25, buttons=[
            dict(label='▶ Play sequence', method='animate', args=[None, dict(frame=dict(duration=60000/len(frames), redraw=True), transition=dict(duration=0), fromcurrent=True)]),
            dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(x=.22, len=.76, y=-.23, currentvalue=dict(prefix='Elapsed minutes: '), steps=[dict(label=f'{mins[i]:.1f}', method='animate', args=[[str(i)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for i in range(len(frames))])])
    fig.update_xaxes(title_text='Sensor x (px)', range=[0, 2048], row=1, col=1)
    fig.update_yaxes(title_text='Sensor y (px)', range=[0, 2048], row=1, col=1)
    residual_limit = max(12., max(np.max(np.abs(q['residual'])) for q in frames)*1.15)
    fig.update_xaxes(title_text='Centroid residual x (arcsec)', range=[-residual_limit, residual_limit], row=1, col=2)
    fig.update_yaxes(title_text='Centroid residual y (arcsec)', range=[-residual_limit, residual_limit], row=1, col=2)
    fig.update_xaxes(title_text='Elapsed time (minutes)', row=2, col=2)
    fig.update_yaxes(title_text='Attitude / withheld residual (arcsec)', row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text='Position error (m)', row=2, col=2, secondary_y=True)
    return fig


def main(make_figures=None, fast=None, **simulation_kwargs):
    under_pytest = 'pytest' in sys.modules or os.environ.get('PYTEST_CURRENT_TEST') is not None
    if make_figures is None: make_figures = not under_pytest
    if fast is None: fast = under_pytest
    simulation_kwargs.setdefault('samples', 181 if fast else 361)
    result = simulate_star_finder(**simulation_kwargs)
    frames = result['frames']
    summary = {key: result[key] for key in ('source', 'source_sha256', 'catalogue_selected', 'period_s', 'duration_s',
        'propagation_error_m', 'noise_arcsec', 'prior_position_error_m', 'prior_velocity_error_m_s', 'seed')}
    summary.update(samples=len(frames), matched=sum(f['matched'] for f in frames), correctly_identified=sum(f['correct'] for f in frames),
        attitude_rms_arcsec=float(np.sqrt(np.mean([f['attitude_error_arcsec']**2 for f in frames]))),
        position_rms_m=float(np.sqrt(np.mean([f['position_error_m']**2 for f in frames]))),
        heldout_rms_arcsec=float(np.sqrt(np.mean([f['heldout_rms_arcsec']**2 for f in frames]))),
        measurement_origin='SIMULATION, not observed spacecraft telemetry',
        limits='Distant stars: no parallax/RV. Coarse attitude/orbit prior. Ideal calibrated cameras and identified WGS84 grid landmarks; no raw-image detection, clouds, terrain, illumination, finite-landmark light-time/aberration, or landmark misidentification model.')
    outputs = dict(result=result, summary=summary)
    if make_figures:
        import pandas as pd
        fig = build_star_finder_figure(result)
        html = Path(figpath('demo_star_finder.html'))
        plot_html = fig.to_html(full_html=False, auto_play=False, include_plotlyjs=True)
        html.write_text("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>SSATK Star Finder</title>
<style>body{margin:0;background:#0a1220;color:#dbe9f7;font:16px system-ui}header,footer{max-width:1400px;margin:24px auto;padding:0 24px}a{color:#69e5cc}h1{margin-bottom:8px}p{line-height:1.55}.note{color:#a9bbce}</style></head><body>
<header><h1>Star Finder</h1><p>Real catalogue stars. Simulated camera measurements over one 98.8-minute orbit.
Stars determine the satellite's attitude; a second calibrated camera observes known Earth landmarks to estimate position.</p>
<p><a href="demo_star_finder_star_measurements.csv">Star measurements and catalogue comparisons (CSV)</a> ·
<a href="demo_star_finder_navigation.csv">Position and attitude errors (CSV)</a> ·
<a href="demo_star_finder_summary.json">Run settings and results</a></p></header>""".replace('one 98.8-minute orbit', f"{result['duration_s']/60:.1f} minutes") + plot_html + """
<footer><p>Hover over measured stars for predicted and measured right ascension/declination. Scrub the time slider or play the sequence.
Diamonds in the residual panel are stars withheld from the attitude fit. At orbit scale, the position fixes nearly overlap the true trajectory; the error plot resolves their differences.</p>
<p class="note">The reference is the packaged bright-star catalogue at epoch J2000, propagated to the observation time. Star rays include aberration and solar light deflection, using a coarse orbit prior.
The catalogue extract lacks distance, radial velocity and uncertainty: parallax and perspective acceleration are omitted. Row labels refer to this CSV, not Hipparcos identifiers.</p>
<p class="note">This is a calibrated-centroid simulation, not camera telemetry or raw-image star detection. It assumes a coarse attitude prior, two perfectly calibrated cameras, and correctly identified WGS84 grid landmarks.
Landmark bearings use instantaneous geometry without light-time or aberration corrections. Clouds, terrain, illumination, optical distortion and calibration drift are not modeled. The 3-arcsecond input noise is a tangent-plane assumption, not a claimed sensor specification.</p>
<p>Background: <a href="https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/">NASA: star trackers determine attitude</a> ·
<a href="https://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html">Astropy: apparent geocentric directions</a></p></footer></body></html>""".replace("3-arcsecond input noise", f"{result['noise_arcsec']:g}-arcsecond input noise"))
        fig.write_image(figpath('demo_star_finder.png'), width=1500, height=1050)
        pd.DataFrame(result['observations']).to_csv(figpath('demo_star_finder_star_measurements.csv'), index=False)
        pd.DataFrame(result['navigation']).to_csv(figpath('demo_star_finder_navigation.csv'), index=False)
        Path(figpath('demo_star_finder_summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
        outputs.update(html=str(html), figure=fig)
    print(json.dumps(summary, indent=2))
    return outputs


if __name__ == '__main__':
    main(make_figures=True, fast=False)
