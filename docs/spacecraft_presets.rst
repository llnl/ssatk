NASA astronomy viewing mesh catalog
===================================

Scope and fidelity
------------------

This is an offline catalog of procedural **viewing templates**, not NASA CAD.
It covers NASA astronomy observatories, historical Explorers, NASA-funded
small missions and technology demonstrations, international observatories with
NASA participation, and Shuttle/ISS astronomy payloads. Failed launches and
selected future missions are included. Reused spacecraft names such as K2 and
NEOWISE are aliases, not extra vehicles. Hosted models contain the instrument,
not its entire carrier. The HWO entry is explicitly a concept placeholder.

The inventory was reconciled on 2026-10-12 against NASA's Astrophysics fleet,
Pioneers and SmallSat lists, the HEASARC mission inventory, MAST mission list,
and JPL astronomy missions. These inventories have different scopes and are
not a single exhaustive NASA register. The table below states the implemented
coverage rather than claiming an unqualified census of every historical
NASA-funded experiment. Pure heliophysics, Earth observation, planetary probes
with incidental astronomy sensors, sounding rockets, balloons and airborne
observatories are outside the selection. The OSO series and SMM are included
because HEASARC lists their non-solar astronomy measurements; EPOCh is included
as a dedicated astronomy program on a repurposed planetary spacecraft.

Some entries have distinct mission geometry (e.g. Webb, Hubble, Roman, Chandra,
TESS, SPHEREx and NICER); others use explicitly identified telescope, CubeSat,
reflector or detector family models. A source establishes mission identity or
selected features; it does not certify every modeled surface. Only features
listed under "Referenced features" are tied to published numbers/counts.
All other lengths, colors, appendage placement, optical surfaces, antennas,
rails and clearances are schematic. A family model is a selectable stand-in,
not a recovered engineering model. No textures, mesh binaries or downloads
are added for these presets.

All coordinates are metres in right-handed **visual** body axes. The origin is
fixed by construction, not a measured center of mass. These axes must be aligned
with the user's attitude convention before combining a preset with flight
quaternions; the catalog does not supply mission attitude frame definitions.
No mass, inertia, articulation, automatic Sun tracking, optical ray tracing,
or validated radiation-pressure properties are inferred from the presets.

Selection
---------

.. code-block:: python

   from ssatk.plots.spacecraft_view import SpacecraftMesh, spacecraft_view

   print(SpacecraftMesh.presets())
   print(SpacecraftMesh.presets(group="small"))
   print(SpacecraftMesh.preset_info("webb"))  # sources, axes, approximation notes
   mesh = SpacecraftMesh.from_preset("hubble")
   view = spacecraft_view(r, t, q, mesh="jwst")
   view.save("webb.png")
   view.save("webb.gif", fps=15, duration_s=6)

Names ignore case, spaces, hyphens and underscores. The canonical names and
additional aliases are below. ``scale=2`` doubles all lengths; it does not
change any dynamics. ``mesh.face_parts`` labels each original convex face,
and ``mesh.triangle_faces`` maps render triangles back to those faces. These
labels support inspection, not engineering part identifiers. Each factory call
owns its arrays, and returned metadata can be edited without changing the catalog.

Inventory sources
-----------------

* `NASA Astrophysics <https://science.nasa.gov/astrophysics/>`__
* `NASA Pioneers <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__
* `NASA 2025 fleet <https://www.nasa.gov/wp-content/uploads/2025/03/emerging-technologies-presentation-final-compressed.pdf>`__
* `NASA 2021 SmallSat inventory <https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086>`__
* `HEASARC <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/allmissions.html>`__
* `MAST <https://archive.stsci.edu/missions-and-data>`__
* `JPL <https://www.jpl.nasa.gov/go/astronomy-and-physics/missions/>`__

HEASARC and MAST also archive non-NASA missions; archive presence alone is not
used to relabel a foreign observatory as NASA-owned. The "partner" group
includes NASA instrument/science participation or launch support, not ownership.
Mission status and future launch dates are intentionally not encoded because
those change independently of the viewing geometry.

Available presets
-----------------

There are **107 canonical presets**, including 104 named mission/payload/configuration entries and three generic templates.

starlink — Starlink-like (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: generic. Fidelity: schematic family model.
* Axes: Arrays extend along +/-Y; +Z is the array normal. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Flat bus and dual-wing visual template; no manufacturer dimensions claimed.
* `Source 1 <https://www.starlink.com/technology>`__

cubesat — Generic 3U CubeSat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: generic. Fidelity: schematic; selected referenced features only.
* Aliases: ``3u``.
* Axes: Long axis +Z; no deployed wings. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Four corner rails, two end decks, recessed body cells. Rail/cell details are schematic.
* Referenced features: 3U envelope: 0.100 x 0.100 x 0.3405 m (NASA Table 6-1).
* `Source 1 <https://www.nasa.gov/smallsat-institute/sst-soa/structures-materials-and-mechanisms/>`__

boxwing — Generic box with two wings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: generic. Fidelity: schematic family model.
* Axes: Arrays extend along +/-X; +Z is the array normal. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Chosen bus 1.2 x 1.0 x 0.8 m and two 2.0 x 1.4 m panels.

hubble — Hubble Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``hst``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Outer shrouds, open pupil and two arrays. Boom lengths and antenna are schematic.
* Referenced features: Body length 13.2 m; aft diameter 4.2 m; primary diameter 2.4 m. Two current-generation array envelopes, each 7.1 x 2.6 m.
* `Source 1 <https://science.nasa.gov/mission/hubble/observatory/design/>`__
* `Source 2 <https://science.nasa.gov/mission/hubble/observatory/design/electrical-power/>`__

jwst — James Webb Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``webb``, ``james-webb``.
* Axes: Primary faces -Y; sunshield lies in XY; telescope is on +Z side. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Planar hexagonal mosaic and kite shields; gaps, curvature, layer separation and supports simplified.
* Referenced features: 18 primary hexagons, each 1.32 m across flats; five shield layers. Largest shield envelope 21.2 x 14.2 m.
* `Source 1 <https://science.nasa.gov/mission/webb/science-overview/science-explainers/telescope-overview/>`__
* `Source 2 <https://science.nasa.gov/mission/webb/webbs-sunshield/>`__

roman — Nancy Grace Roman Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``wfirst``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Barrel, six-panel shield and aperture cover; construction proportions are schematic.
* Referenced features: 2.4 m primary diameter; six solar-array sunshield panels.
* `Source 1 <https://science.nasa.gov/mission/roman-space-telescope/telescope/>`__
* `Source 2 <https://science.nasa.gov/missions/roman-space-telescope/building-roman/>`__

chandra — Chandra X-ray Observatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``axaf``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mirror/bench/bus and wings; internal nested mirror surfaces are omitted.
* Referenced features: 13.8 m body length and 19.5 m array span (deployed specification).
* `Source 1 <https://www.chandra.harvard.edu/about/specs.html>`__

spitzer — Spitzer Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``sirtf``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* Referenced features: 4 m body length and 0.85 m primary diameter.
* `Source 1 <https://www.jpl.nasa.gov/news/press_kits/spitzer/quick-facts/>`__

compton — Compton Gamma Ray Observatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``cgro``, ``gro``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: BATSE corner placement retained; other instrument volumes are schematic, not complete internal payloads.
* Referenced features: Eight BATSE modules, one at each bus corner.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/cgro/cgro/batse.html>`__

kepler — Kepler / K2
~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``k2``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* Referenced features: 0.95 m entrance aperture (not the larger primary mirror).
* `Source 1 <https://science.nasa.gov/wp-content/uploads/2023/10/Kepler_Exhibit-1.pdf>`__

tess — Transiting Exoplanet Survey Satellite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Four camera hoods fan in the YZ plane. Bus, hood positions and panel sizes schematic.
* Referenced features: Four cameras, 0.105 m entrance pupils; 0.7 m high-gain antenna.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/tess/telescope_information.html>`__

fermi — Fermi Gamma-ray Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Aliases: ``glast``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shows the LAT segmentation above a bus; external blankets and GBM detectors are omitted.
* Referenced features: LAT comprises a 4 x 4 array of 16 modules.
* `Source 1 <https://fermi.gsfc.nasa.gov/science/instruments/lat.html>`__

swift — Neil Gehrels Swift Observatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic family model.
* Aliases: ``gehrels-swift``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic BAT coded mask and adjacent XRT/UVOT barrels; no mask pattern or calibrated apertures.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/swift.html>`__

nustar — Nuclear Spectroscopic Telescope Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Open truss with two optical modules. The reference planes, not the outer tips, set focal separation.
* Referenced features: Two optics modules; 10.14 m optics-reference to detector-reference separation.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_obsguide.pdf>`__

ixpe — Imaging X-ray Polarimetry Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic open boom; the flight thermal sock is omitted to expose the separation.
* Referenced features: Three optical modules; 4 m optics-reference to detector-reference separation.
* `Source 1 <https://ixpe.msfc.nasa.gov/for_scientists/documentation/IXPE-SOC-DOC-011A_UG-Observatory.pdf>`__

wise — WISE / NEOWISE
~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic family model.
* Aliases: ``neowise``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/mission/wise/>`__

spherex — SPHEREx
~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Conical shields and internal telescope. Bus and bottom array are schematic.
* Referenced features: Three photon shields; outer shield 3.2 m diameter and 1.7 m height; 0.20 m aperture.
* `Source 1 <https://www.jpl.nasa.gov/press-kits/spherex/mission-overview/spacecraft/>`__

galex — Galaxy Evolution Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: observatory. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* Referenced features: 0.5 m telescope diameter.
* `Source 1 <https://science.nasa.gov/mission/galex/>`__
* `Source 2 <https://science.nasa.gov/mission/ULTRASAT/>`__

iras — Infrared Astronomical Satellite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic telescope family; apertures and proportions are visual approximations.
* `Source 1 <https://www.jpl.nasa.gov/missions/infrared-astronomical-satellite-iras/>`__

iue — International Ultraviolet Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic telescope family; apertures and proportions are visual approximations.
* `Source 1 <https://science.nasa.gov/mission/iue/>`__

fuse — Far Ultraviolet Spectroscopic Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic telescope family; apertures and proportions are visual approximations.
* `Source 1 <https://science.nasa.gov/mission/fuse/>`__

euve — Extreme Ultraviolet Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic telescope family; apertures and proportions are visual approximations.
* `Source 1 <https://science.nasa.gov/mission/euve/>`__

wire — Wide-field Infrared Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic telescope family; apertures and proportions are visual approximations.
* `Source 1 <https://science.nasa.gov/mission/wire/>`__

chipsat — Cosmic Hot Interstellar Plasma Spectrometer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``chips``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/mission/chips/>`__

cobe — Cosmic Background Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Simplified instrument/shield volumes; reflector geometry is not an optical surface.
* `Source 1 <https://science.nasa.gov/mission/cobe/>`__

wmap — Wilkinson Microwave Anisotropy Probe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Simplified instrument/shield volumes; reflector geometry is not an optical surface.
* `Source 1 <https://science.nasa.gov/mission/wmap/>`__

swas — Submillimeter Wave Astronomy Satellite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Simplified instrument/shield volumes; reflector geometry is not an optical surface.
* `Source 1 <https://science.nasa.gov/mission/swas/>`__

explorer11 — Explorer 11
~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``explorer-11``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/explorer11.html>`__

rae1 — Radio Astronomy Explorer 38
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``explorer38``, ``explorer-38``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Very long V antennas; schematic ~458 m tip separation. Wire thickness exaggerated for visibility.
* `Source 1 <https://ntrs.nasa.gov/citations/19900020097>`__

rae2 — Radio Astronomy Explorer 49
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``explorer49``, ``explorer-49``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Very long V antennas; schematic ~458 m tip separation. Wire thickness exaggerated for visibility.
* `Source 1 <https://science.nasa.gov/mission/explorer-49/>`__

oao1 — OAO-1
~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared octagonal OAO family model; instruments differ between missions and are simplified.
* `Source 1 <https://www.nasa.gov/solar-system/nasas-first-stellar-observatory-oao-2-turns-50/>`__

oao2 — OAO-2 / Stargazer
~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared octagonal OAO family model; instruments differ between missions and are simplified.
* `Source 1 <https://www.nasa.gov/solar-system/nasas-first-stellar-observatory-oao-2-turns-50/>`__

oaob — OAO-B (launch failure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared octagonal OAO family model; instruments differ between missions and are simplified.
* `Source 1 <https://www.nasa.gov/solar-system/nasas-first-stellar-observatory-oao-2-turns-50/>`__

copernicus — Copernicus / OAO-3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``oao3``, ``oao-3``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared octagonal OAO family model; instruments differ between missions and are simplified.
* `Source 1 <https://www.nasa.gov/solar-system/nasas-first-stellar-observatory-oao-2-turns-50/>`__

uhuru — Uhuru / SAS-1
~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``sas1``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/uhuru.html>`__

sas2 — Small Astronomy Satellite 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/sas2.html>`__

sas3 — Small Astronomy Satellite 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/sas3.html>`__

heao1 — High Energy Astronomy Observatory 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/heao1.html>`__

einstein — Einstein Observatory / HEAO-2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``heao2``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/heao2.html>`__

heao3 — High Energy Astronomy Observatory 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/heao3.html>`__

rxte — Rossi X-ray Timing Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``xte``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/rxte.html>`__

hete1 — High Energy Transient Explorer 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared HETE family schematic. HETE-1 did not separate from its launch vehicle.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/hete2.html>`__

hete2 — High Energy Transient Explorer 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared HETE family schematic. HETE-1 did not separate from its launch vehicle.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/hete2.html>`__

asteria — ASTERIA
~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://www.jpl.nasa.gov/topics/cubesats/>`__
* `Source 2 <https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086>`__

halosat — HaloSat
~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/halosat.html>`__

burstcube — BurstCube
~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/burstcube.html>`__

cute — Colorado Ultraviolet Transit Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086>`__

sparcs — Star-Planet Activity Research CubeSat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086>`__

sprite — Supernova Remnants and Proxies for ReIonization Testbed Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086>`__

blackcat — BlackCAT
~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/blackcat.html>`__

aspera — Aspera
~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/aspera.html>`__
* `Source 2 <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__

mantis — Monitoring Activity from Nearby Stars with UV Imaging and Spectroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2025/03/emerging-technologies-presentation-final-compressed.pdf>`__

landolt — Landolt artificial-star calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.
* `Source 1 <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__

pandora — Pandora
~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__

starburst — StarBurst
~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/starburst.html>`__
* `Source 2 <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__

gosox — Globe Orbiting Soft X-ray Polarimeter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: small. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Conceptual telescope-family stand-in; final external mechanical design is not represented.
* `Source 1 <https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/>`__

rosat — ROSAT
~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/rosat/rosat.html>`__

asca — ASCA
~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/asca.html>`__

astroe — Astro-E (launch failure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/astroe.html>`__

suzaku — Suzaku / Astro-E2
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/suzaku.html>`__

hitomi — Hitomi / Astro-H
~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/hitomi.html>`__

xrism — XRISM
~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/xrism.html>`__

xmm-newton — XMM-Newton
~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Aliases: ``xmm``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/xmm.html>`__

exosat — EXOSAT
~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/exosat.html>`__

integral — INTEGRAL
~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/integral.html>`__

herschel — Herschel Space Observatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic reflector/shroud family, not an off-axis optical model or measured flight envelope.
* `Source 1 <https://science.nasa.gov/mission/herschel-space-observatory/>`__

planck — Planck
~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic reflector/shroud family, not an off-axis optical model or measured flight envelope.
* `Source 1 <https://science.nasa.gov/mission/planck/>`__

euclid — Euclid
~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/mission/euclid/>`__

iso — Infrared Space Observatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://irsa.ipac.caltech.edu/Missions/iso.html>`__

ans — Astronomical Netherlands Satellite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/ans.html>`__

ariel5 — Ariel V
~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/ariel5.html>`__

ariel6 — Ariel VI
~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/ariel6.html>`__

cosi — Compton Spectrometer and Imager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: development. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/cosi.html>`__

uvex — Ultraviolet Explorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: development. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/astrophysics/programs/physics-of-the-cosmos/community/uvex-selected-as-next-astrophysics-medium-class-explorer/>`__
* `Source 2 <https://science.nasa.gov/mission/ULTRASAT/>`__

ultrasat — Ultraviolet Transient Astronomy Satellite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://science.nasa.gov/mission/ULTRASAT/>`__

ariel — Ariel / NASA CASE
~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Aliases: ``case``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2025/03/emerging-technologies-presentation-final-compressed.pdf>`__
* `Source 2 <https://www.jpl.nasa.gov/go/astronomy-and-physics/missions/>`__

athena — NewAthena
~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Aliases: ``newathena``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Conceptual open optical bench; final structure, optics and mission design may differ.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/athena.html>`__

lisa — LISA (one spacecraft)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: partner. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: One disc-shaped spacecraft with two laser telescope directions; the three-spacecraft constellation is not generated.
* `Source 1 <https://science.nasa.gov/mission/lisa/>`__

hwo — Habitable Worlds Observatory concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: concept. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Generic 6 m-aperture concept stand-in. NASA has not fixed the flight design; this is not a prediction of it.
* `Source 1 <https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/>`__

nicer — NICER (ISS payload)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic; selected referenced features only.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: 56 apertures and schematic mounting struts; ISS, gimbal articulation and internal mirrors omitted.
* Referenced features: 56 X-ray concentrator assemblies.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/nicer/mission_guide/>`__

astro1 — Astro-1 UV payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

astro2 — Astro-2 UV payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

bbxrt — Broad Band X-ray Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/bbxrt.html>`__

dxs — Diffuse X-ray Spectrometer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/dxs.html>`__

orfeus1 — ORFEUS-SPAS I
~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

orfeus2 — ORFEUS-SPAS II
~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

imaps — Interstellar Medium Absorption Profile Spectrograph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

spartan1 — Spartan 1
~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/spartan1.html>`__

spacelab1 — Spacelab 1 X-ray payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/spacelab1.html>`__

spacelab2 — Spacelab 2 X-ray payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/spacelab2.html>`__

tigeriss — TIGERISS
~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://www.nasa.gov/wp-content/uploads/2025/03/emerging-technologies-presentation-final-compressed.pdf>`__

calet — CALorimetric Electron Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/calet.html>`__

ams02 — Alpha Magnetic Spectrometer 02
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/ams.html>`__

glowbug — Glowbug gamma-ray telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/glowbug.html>`__

iss-cream — Cosmic Ray Energetics and Mass (ISS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://svs.gsfc.nasa.gov/12679/>`__

faust — Far Ultraviolet Space Telescope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://www.nasa.gov/mission/sts-45/>`__

uvstar — Ultraviolet Spectrograph Telescope for Astronomical Research
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://ntrs.nasa.gov/citations/19940014705>`__

starlite — Spectrograph/Telescope for Astronomical Research
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: hosted. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.
* `Source 1 <https://science.nasa.gov/mission/ieh-3/>`__

epoch — EPOCh / EPOXI (Deep Impact spacecraft)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``epoxi``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Schematic dual-imager spacecraft; EPOCh was the stellar/exoplanet observing program on the repurposed Deep Impact spacecraft.
* `Source 1 <https://archive.stsci.edu/missions-and-data>`__

oso1 — Orbiting Solar Observatory 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso1.html>`__

oso3 — Orbiting Solar Observatory 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso3.html>`__

oso4 — Orbiting Solar Observatory 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso4.html>`__

oso5 — Orbiting Solar Observatory 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso5.html>`__

oso6 — Orbiting Solar Observatory 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso6.html>`__

oso7 — Orbiting Solar Observatory 7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso7.html>`__

oso8 — Orbiting Solar Observatory 8
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/oso8.html>`__

smm — Solar Maximum Mission
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Group: legacy. Fidelity: schematic family model.
* Aliases: ``solarmax``.
* Axes: +Z is the schematic telescope/payload direction. Right-handed visual axes, not a mission flight attitude convention.
* Model notes: Component sizes, separation, appendages and colors are illustrative.
* `Source 1 <https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/solarmax.html>`__

