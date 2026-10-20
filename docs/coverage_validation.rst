Coverage maps: definitions and validation
=========================================

The coverage demo computes **geometric ground access**, using archived HST
two-line elements (TLEs) at 2026-07-06T10:33:18.829 UTC. It is not a current
tracking prediction. The satellite's simplified general perturbations model
(SGP4) states come from ``ssapy.compute.rv(..., propagator=SGP4Propagator())``;
SSAPy's default two-body propagator is appropriate only for the separately
specified Keplerian example. An explicit propagator can override either choice.

Every coverage cell represents a sea-level site on the World Geodetic System
1984 (WGS84) ellipsoid. The selected site's specified ellipsoid height is used
for its own statistics. Geometric elevation is measured relative to the local
geodetic normal. Neither path models terrain, atmospheric refraction, weather,
a radio link budget or optical detectability.

Map definitions
----------------

The default maps cover three days with **15-second** samples on a
**90 × 180** grid of **2° × 2°** cells. Latitude and longitude are the cell
centers; neither the antimeridian nor the poles are duplicated. The displayed
values are the calculated cell values, without Gaussian smoothing or contour
interpolation. Cells do not resolve geographic variation within their bounds.
The surface-area-weighted global mean uses the exact WGS84 latitude-band areas.

Access is elevation **strictly greater than 5°**. Rise and set times interpolate
between adjacent elevation samples. Both time-window endpoints are included,
but no duration outside the window is counted. The four panels show:

* Mean contact duration observed within the window. Gray denotes no contact.
* Longest no-contact interval within the window, including edge gaps. No
  contact anywhere in the window therefore produces a **4320-minute** gap.
* Contacts intersecting the window divided by its duration in days. A contact
  already in progress at the beginning counts once; this is not a long-run
  contact-arrival rate.
* Total observed contact time normalized to minutes per day, bounded by **1440**.

Boundary contacts are clipped to the window, and the statistics explicitly
return ``start_clipped`` and ``end_clipped``. Any contact wholly between samples
can be missed. Three days is a chosen analysis window, not a guarantee of
longitude-uniform coverage: geostationary and resonant orbits retain strong
longitude dependence.

The separate site plot uses a **10°** elevation mask. Its optical-candidate
screen requires full solar illumination and geometric Sun elevation below
**−6°**, using the same WGS84 local normal. The phase angle is the
Sun–satellite–observer angle, with its vertex at the satellite. These conditions
identify geometric candidates rather than confirmed optical detections.

Verified defects and corrections
---------------------------------

The review reproduced the following defects in the preceding gallery build:

* **TLE propagation:** implicit two-body propagation displaced HST by
  **4675.521 km** relative to direct SGP4 at the end of three days. At Cape
  Canaveral, **322** of the sampled above-5° classifications disagreed. Both
  site and map propagation now explicitly select SGP4 for TLE-derived input.
* **Inconsistent site geometry:** grid points used a sphere, while the selected
  site used WGS84. The two positions differed by **23.905 km** at 45° latitude.
  An analytic 1000-km-range target at true elevation **5°** was classified at
  **4.485581°** by the old grid. Grid and site geometry now use WGS84.
* **Duration normalization:** 8640 samples spaced 30 seconds apart were treated
  as 8640 intervals but normalized by only 8639 intervals. Continuous contact
  therefore yielded **1440.166686 min/day**. Endpoint-aware integration fixes
  this, and interpolated crossings reduce sample-quantization error.
* **Global averaging:** equal weighting of latitude/longitude samples
  overweighted polar area. For an analytic spherical cap of angular radius
  30°, the correct surface fraction is **0.0669873**; the old uniform-angle
  grid average was **0.0439256**. The production grid now uses ellipsoid cell
  areas and unique cell centers.
* **Displayed metrics:** Gaussian smoothing altered pass lengths, counts and
  worst gaps. Masking zero-contact cells also hid full-window outages. Raw
  metrics now appear directly, including those outages.
* **Site illumination geometry:** Sun elevation used the geocentric radial
  direction, and the phase curve used an Earth-centered angle rather than the
  observer's phase angle at the satellite. Both now use the stated geometry.

References and measured agreement
-----------------------------------

``tests/test_coverage_references.py`` contains five test functions with seven
cases. References are literal piecewise-linear crossing times; an analytic
circular-overhead-pass horizon angle; explicit WGS84 coordinates and ellipsoid
normals; numerical quadrature of the ellipsoid surface-area element; and
native ``sgp4.api.Satrec`` positions transformed directly from TEME to ITRS
with Astropy. Here TEME means True Equator, Mean Equinox, and ITRS is the
International Terrestrial Reference System. The end-to-end position comparison
allows **0.1 m**; the three-day sampled audit measured **0.022239 m** maximum
between the different frame-conversion paths.

``scripts/validate_coverage_maps.py`` independently recomputes all **16,200**
published cells using native SGP4, Astropy, explicit WGS84/local-horizon
geometry, and a connected-run crossing implementation. At the same 15-second
sample times, contact counts match in every cell. Maximum absolute differences
are **1.954 × 10⁻⁹** in coverage fraction and **4.625 × 10⁻⁶ minutes** in the
longest gap. These are implementation comparisons at matched sampling, not
bounds on unresolved contacts or on actual satellite tracking accuracy.

A separate native-SGP4 event search brackets crossings at **1 second** and
refines roots with Brent's method to **10 microseconds**. Across eight selected
locations, including near-tangent northern and southern passes and a no-access
site, the 15-second results preserve the contact counts. The largest rise/set
error is **0.711753 seconds**, reduced to **0.078722 seconds** at 5-second
sampling. Cape Canaveral's maximum is **0.128230 seconds** at 15-second sampling.
These measured errors apply to these cases; a sub-step grazing contact can
still be missed elsewhere.

The direct SGP4 comparison shares the underlying dynamical model with SSAPy.
It validates model selection, original TLE drag terms, units, frame handling,
geometry and integration. It does not validate the archived TLE against flight
tracking. The analytic tests provide separate geometric and timing references.

Reproduction and exports
-------------------------

Generate the demo, then run the reference audit against its output directory:

.. code-block:: bash

   python demos/sensor_coverage/demo_coverage_analysis.py
   python -m scripts.validate_coverage_maps \
     --data-dir /path/to/cape_canaveral_p28.39_m80.60 \
     --output /path/to/coverage-validation.json
   python -m pytest tests/test_coverage_references.py \
     tests/test_coverage_site_geometry.py tests/test_eclipse_partial_shadow.py -q

The demo exports both PNGs, ``coverage_grid_hst.csv`` with the raw cell metrics,
and ``coverage_metadata_hst.json`` with the exact TLE, epoch, masks, sampling,
grid, selected-site contact intervals and model identifiers.
``make_figures=False`` computes results without writing figures or files.
The workflow suite separately checks this behavior and the site-illumination
wiring with prescribed collinear geometry.

For the SGP4/TEME conventions, see the
`python-sgp4 documentation <https://pypi.org/project/sgp4/>`_ and
`Astropy satellite-coordinate guide <https://docs.astropy.org/en/stable/coordinates/satellites.html>`_.
