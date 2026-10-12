"""Offline procedural spacecraft catalog with explicit provenance and fidelity.

Sources identify missions and visible components; they are NOT CAD licenses or
certificates of geometric accuracy. Only ``reference_features`` are tied to
published numbers. Everything else is schematic, including visual body axes.
"""
from __future__ import annotations

from copy import deepcopy

from . import _spacecraft_shapes as shapes

HEASARC = "https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/"
SCIENCE = "https://science.nasa.gov/mission/"
PIONEERS = "https://science.nasa.gov/astrophysics/programs/astrophysics-pioneers/"
FLEET = "https://www.nasa.gov/wp-content/uploads/2025/03/emerging-technologies-presentation-final-compressed.pdf"
SMALLSATS = "https://www.nasa.gov/wp-content/uploads/2021/08/3.tan_nasa_town_hall_2021_smallsat-sscg_0.pdf?emrc=9bf086"
OAO = "https://www.nasa.gov/solar-system/nasas-first-stellar-observatory-oao-2-turns-50/"
MAST = "https://archive.stsci.edu/missions-and-data"

_CATALOG = {}


def _add(name, label, recipe, *, group="observatory", sources=(), aliases=(),
         features=(), notes="", axes="+Z is the schematic telescope/payload direction.",
         **parameters):
    _CATALOG[name] = dict(
        name=name, label=label, group=group, recipe=recipe, parameters=parameters,
        sources=tuple(sources), aliases=tuple(aliases), reference_features=tuple(features),
        fidelity="schematic; selected referenced features only" if features else "schematic family model",
        notes=notes or "Component sizes, separation, appendages and colors are illustrative.",
        axes=axes + " Right-handed visual axes, not a mission flight attitude convention.",
        origin="Fixed construction origin near the bus; not a measured center of mass.",
    )


_add("starlink", "Starlink-like (default)", "legacy", group="generic",
     sources=("https://www.starlink.com/technology",),
     notes="Flat bus and dual-wing visual template; no manufacturer dimensions claimed.",
     axes="Arrays extend along +/-Y; +Z is the array normal.")
_add("cubesat", "Generic 3U CubeSat", "cubesat", group="generic", aliases=("3u",),
     sources=("https://www.nasa.gov/smallsat-institute/sst-soa/structures-materials-and-mechanisms/",),
     features=("3U envelope: 0.100 x 0.100 x 0.3405 m (NASA Table 6-1).",),
     notes="Four corner rails, two end decks, recessed body cells. Rail/cell details are schematic.",
     axes="Long axis +Z; no deployed wings.")
_add("boxwing", "Generic box with two wings", "legacy", group="generic",
     notes="Chosen bus 1.2 x 1.0 x 0.8 m and two 2.0 x 1.4 m panels.",
     axes="Arrays extend along +/-X; +Z is the array normal.")

_add("hubble", "Hubble Space Telescope", "hubble", aliases=("hst",),
     sources=(SCIENCE+"hubble/observatory/design/", SCIENCE+"hubble/observatory/design/electrical-power/"),
     features=("Body length 13.2 m; aft diameter 4.2 m; primary diameter 2.4 m.",
               "Two current-generation array envelopes, each 7.1 x 2.6 m."),
     notes="Outer shrouds, open pupil and two arrays. Boom lengths and antenna are schematic.")
_add("jwst", "James Webb Space Telescope", "webb", aliases=("webb", "james-webb"),
     sources=(SCIENCE+"webb/science-overview/science-explainers/telescope-overview/", SCIENCE+"webb/webbs-sunshield/"),
     features=("18 primary hexagons, each 1.32 m across flats; five shield layers.",
               "Largest shield envelope 21.2 x 14.2 m."),
     notes="Planar hexagonal mosaic and kite shields; gaps, curvature, layer separation and supports simplified.",
     axes="Primary faces -Y; sunshield lies in XY; telescope is on +Z side.")
_add("roman", "Nancy Grace Roman Space Telescope", "roman", aliases=("wfirst",),
     sources=(SCIENCE+"roman-space-telescope/telescope/", "https://science.nasa.gov/missions/roman-space-telescope/building-roman/"),
     features=("2.4 m primary diameter; six solar-array sunshield panels.",),
     notes="Barrel, six-panel shield and aperture cover; construction proportions are schematic.")
_add("chandra", "Chandra X-ray Observatory", "chandra", aliases=("axaf",),
     sources=("https://www.chandra.harvard.edu/about/specs.html",),
     features=("13.8 m body length and 19.5 m array span (deployed specification).",),
     notes="Mirror/bench/bus and wings; internal nested mirror surfaces are omitted.")
_add("spitzer", "Spitzer Space Telescope", "barrel", aliases=("sirtf",),
     sources=("https://www.jpl.nasa.gov/news/press_kits/spitzer/quick-facts/",),
     features=("4 m body length and 0.85 m primary diameter.",),
     radius=1.0, length=4.0, aperture=.425, panel="side")
_add("compton", "Compton Gamma Ray Observatory", "detector_bus", aliases=("cgro", "gro"),
     sources=("https://heasarc.gsfc.nasa.gov/docs/cgro/cgro/batse.html",),
     features=("Eight BATSE modules, one at each bus corner.",), size=(4, 3, 2), layout="compton",
     notes="BATSE corner placement retained; other instrument volumes are schematic, not complete internal payloads.")
_add("kepler", "Kepler / K2", "barrel", aliases=("k2",),
     sources=("https://science.nasa.gov/wp-content/uploads/2023/10/Kepler_Exhibit-1.pdf",),
     features=("0.95 m entrance aperture (not the larger primary mirror).",),
     radius=1.2, length=4.7, aperture=.475, panel="side")
_add("tess", "Transiting Exoplanet Survey Satellite", "tess",
     sources=("https://heasarc.gsfc.nasa.gov/docs/tess/telescope_information.html",),
     features=("Four cameras, 0.105 m entrance pupils; 0.7 m high-gain antenna.",),
     notes="Four camera hoods fan in the YZ plane. Bus, hood positions and panel sizes schematic.")
_add("fermi", "Fermi Gamma-ray Space Telescope", "detector_bus", aliases=("glast",),
     sources=("https://fermi.gsfc.nasa.gov/science/instruments/lat.html",),
     features=("LAT comprises a 4 x 4 array of 16 modules.",),
     notes="Shows the LAT segmentation above a bus; external blankets and GBM detectors are omitted.")
_add("swift", "Neil Gehrels Swift Observatory", "detector_bus", aliases=("gehrels-swift",),
     sources=(HEASARC+"swift.html",), size=(1.8, 1.5, 1.6), layout="swift",
     notes="Schematic BAT coded mask and adjacent XRT/UVOT barrels; no mask pattern or calibrated apertures.")
_add("nustar", "Nuclear Spectroscopic Telescope Array", "mast",
     sources=("https://heasarc.gsfc.nasa.gov/docs/nustar/nustar_obsguide.pdf",),
     features=("Two optics modules; 10.14 m optics-reference to detector-reference separation.",),
     notes="Open truss with two optical modules. The reference planes, not the outer tips, set focal separation.")
_add("ixpe", "Imaging X-ray Polarimetry Explorer", "mast", length=4, count=3, optics_radius=.16,
     sources=("https://ixpe.msfc.nasa.gov/for_scientists/documentation/IXPE-SOC-DOC-011A_UG-Observatory.pdf",),
     features=("Three optical modules; 4 m optics-reference to detector-reference separation.",),
     notes="Schematic open boom; the flight thermal sock is omitted to expose the separation.")
_add("wise", "WISE / NEOWISE", "barrel", aliases=("neowise",),
     sources=(SCIENCE+"wise/",), radius=.75, length=2.85, aperture=.20, panel="side")
_add("spherex", "SPHEREx", "spherex", sources=("https://www.jpl.nasa.gov/press-kits/spherex/mission-overview/spacecraft/",),
     features=("Three photon shields; outer shield 3.2 m diameter and 1.7 m height; 0.20 m aperture.",),
     notes="Conical shields and internal telescope. Bus and bottom array are schematic.")
_add("galex", "Galaxy Evolution Explorer", "barrel", radius=.5, length=2.0, aperture=.25,
     sources=(SCIENCE+"galex/", SCIENCE+"ULTRASAT/"), features=("0.5 m telescope diameter.",))

# Historical UV, IR, microwave and Explorer missions. Shared families are
# deliberately labeled: recognition is not an assertion of engineering fidelity.
for name, label, radius, length, aperture, panel in (
    ("iras", "Infrared Astronomical Satellite", 1.0, 3.6, .285, "side"),
    ("iue", "International Ultraviolet Explorer", .85, 4.3, .225, "wings"),
    ("fuse", "Far Ultraviolet Spectroscopic Explorer", .8, 3.8, .6, "wings"),
    ("euve", "Extreme Ultraviolet Explorer", .85, 2.0, .55, "wings"),
    ("wire", "Wide-field Infrared Explorer", .5, 1.8, .15, "wings"),
):
    _add(name, label, "barrel", group="legacy", sources=(("https://www.jpl.nasa.gov/missions/infrared-astronomical-satellite-iras/" if name == "iras" else SCIENCE+name+"/"),),
         radius=radius, length=length, aperture=aperture, panel=panel,
         count=4 if name in ("fuse", "euve") else 1,
         notes="Schematic telescope family; apertures and proportions are visual approximations.")
_add("chipsat", "Cosmic Hot Interstellar Plasma Spectrometer", "detector_bus", group="legacy",
     aliases=("chips",), sources=("https://science.nasa.gov/mission/chips/",),
     size=(.7,.7,.4), layout="generic", count=6)
for name, label, diameter, length, dual, shield in (
    ("cobe", "Cosmic Background Explorer", 2.0, 4, False, True),
    ("wmap", "Wilkinson Microwave Anisotropy Probe", 1.8, 3.2, True, True),
    ("swas", "Submillimeter Wave Astronomy Satellite", .7, 1.5, False, False),
):
    _add(name, label, "dish", group="legacy", sources=(SCIENCE+name+"/",),
         diameter=diameter, length=length, dual=dual, shield=shield,
         notes="Simplified instrument/shield volumes; reflector geometry is not an optical surface.")
_add("explorer11", "Explorer 11", "early", group="legacy", aliases=("explorer-11",),
     sources=(HEASARC+"explorer11.html",), style="explorer", radius=.15, length=.8)
for name, number in (("rae1", 38), ("rae2", 49)):
    _add(name, f"Radio Astronomy Explorer {number}", "early", group="legacy",
         aliases=(f"explorer{number}", f"explorer-{number}"), style="rae",
         sources=(("https://ntrs.nasa.gov/citations/19900020097" if number == 38 else SCIENCE+"explorer-49/"),),
         notes="Very long V antennas; schematic ~458 m tip separation. Wire thickness exaggerated for visibility.")
for name, label in (("oao1", "OAO-1"), ("oao2", "OAO-2 / Stargazer"),
                    ("oaob", "OAO-B (launch failure)"), ("copernicus", "Copernicus / OAO-3")):
    _add(name, label, "early", group="legacy", sources=(OAO,),
         aliases=("oao3", "oao-3") if name == "copernicus" else (),
         length=3.0, radius=1.0,
         notes="Shared octagonal OAO family model; instruments differ between missions and are simplified.")
for name, label, source, radius, length in (
    ("uhuru", "Uhuru / SAS-1", "uhuru", .28, .9),
    ("sas2", "Small Astronomy Satellite 2", "sas2", .30, 1.0),
    ("sas3", "Small Astronomy Satellite 3", "sas3", .38, 1.4),
    ("heao1", "High Energy Astronomy Observatory 1", "heao1", 1.25, 5.7),
    ("einstein", "Einstein Observatory / HEAO-2", "heao2", 1.1, 5.7),
    ("heao3", "High Energy Astronomy Observatory 3", "heao3", 1.25, 5.7),
):
    _add(name, label, "early", group="legacy", sources=(HEASARC+source+".html",),
         aliases=("sas1",) if name == "uhuru" else (("heao2",) if name == "einstein" else ()),
         radius=radius, length=length, style="explorer" if name.startswith("sas") or name == "uhuru" else "oao")
_add("rxte", "Rossi X-ray Timing Explorer", "detector_bus", group="legacy", aliases=("xte",),
     sources=(HEASARC+"rxte.html",), size=(1.8, 1.8, 2.4), layout="generic", count=5)
for number in (1, 2):
    _add(f"hete{number}", f"High Energy Transient Explorer {number}", "detector_bus", group="legacy",
         sources=(HEASARC+"hete2.html",), size=(.65, .65, .85), layout="generic", count=4,
         notes="Shared HETE family schematic. HETE-1 did not separate from its launch vehicle.")

# NASA-funded small missions, including technology demonstrations and selected
# Pioneers. A generic chassis is used when a detailed public shape is unavailable.
for name, label, size, instrument, wings, count, sources in (
    ("asteria", "ASTERIA", (.2263,.100,.366), "aperture", True, 1, ("https://www.jpl.nasa.gov/topics/cubesats/", SMALLSATS)),
    ("halosat", "HaloSat", (.2263,.100,.366), "detector", False, 3, (HEASARC+"halosat.html",)),
    ("burstcube", "BurstCube", (.2263,.100,.366), "detector", True, 4, (HEASARC+"burstcube.html",)),
    ("cute", "Colorado Ultraviolet Transit Experiment", (.2263,.100,.366), "aperture", True, 1, (SMALLSATS,)),
    ("sparcs", "Star-Planet Activity Research CubeSat", (.2263,.100,.366), "aperture", True, 1, (SMALLSATS,)),
    ("sprite", "Supernova Remnants and Proxies for ReIonization Testbed Experiment", (.2263,.2263,.366), "aperture", True, 1, (SMALLSATS,)),
    ("blackcat", "BlackCAT", (.2263,.100,.366), "detector", True, 1, (HEASARC+"blackcat.html",)),
    ("aspera", "Aspera", (.2263,.2263,.366), "aperture", True, 2, (HEASARC+"aspera.html", PIONEERS)),
    ("mantis", "Monitoring Activity from Nearby Stars with UV Imaging and Spectroscopy", (.2263,.2263,.366), "aperture", True, 2, (FLEET,)),
    ("landolt", "Landolt artificial-star calibrator", (.2263,.2263,.366), "aperture", True, 1, (PIONEERS,)),
):
    _add(name, label, "cubesat", group="small", sources=sources,
         size=size, instrument=instrument, wings=wings, count=count,
         notes="Rail-framed family model with symbolic payload. Chassis dimensions, deployed wings and payload placement are schematic, not flight CAD.")
_add("pandora", "Pandora", "barrel", group="small", sources=(PIONEERS,),
     radius=.4, length=1.6, aperture=.225)
_add("starburst", "StarBurst", "detector_bus", group="small", sources=(HEASARC+"starburst.html", PIONEERS),
     size=(.6,.6,.5), layout="generic", count=4)
_add("gosox", "Globe Orbiting Soft X-ray Polarimeter", "barrel", group="small", sources=(PIONEERS,),
     radius=.5, length=2.5, aperture=.3,
     notes="Conceptual telescope-family stand-in; final external mechanical design is not represented.")

# International observatories with NASA contributions, launch support or guest
# observer participation. Do not interpret this group as NASA-owned spacecraft.
for name, label, slug, radius, length, aperture, count in (
    ("rosat", "ROSAT", "rosat", 1.1, 4.5, .6, 1),
    ("asca", "ASCA", "asca", .7, 4.5, .5, 4),
    ("astroe", "Astro-E (launch failure)", "astroe", 1.0, 6.0, .8, 5),
    ("suzaku", "Suzaku / Astro-E2", "suzaku", 1.0, 6.0, .8, 5),
    ("hitomi", "Hitomi / Astro-H", "hitomi", 1.1, 8.0, .8, 4),
    ("xrism", "XRISM", "xrism", .9, 5.0, .65, 2),
    ("xmm-newton", "XMM-Newton", "xmm", 1.9, 10.0, 1.2, 3),
    ("exosat", "EXOSAT", "exosat", .85, 2.6, .6, 2),
):
    _add(name, label, "barrel", group="partner", sources=(("https://heasarc.gsfc.nasa.gov/docs/rosat/rosat.html" if name == "rosat" else HEASARC+slug+".html"),),
         aliases=("xmm",) if name == "xmm-newton" else (),
         radius=radius, length=length, aperture=aperture, count=count,
         notes="Mission-family shroud, optical openings and solar wings. Actual off-axis instruments, deployable benches and dimensions are simplified.")
_add("integral", "INTEGRAL", "detector_bus", group="partner", sources=(HEASARC+"integral.html",),
     size=(2.5, 2.2, 3.5), layout="generic", count=4)
for name, label, diameter, length, shield in (
    ("herschel", "Herschel Space Observatory", 3.5, 7.5, False),
    ("planck", "Planck", 1.8, 4.0, True),
):
    _add(name, label, "dish", group="partner", sources=(SCIENCE+("herschel-space-observatory" if name == "herschel" else name)+"/",),
         diameter=diameter, length=length, shield=shield,
         notes="Schematic reflector/shroud family, not an off-axis optical model or measured flight envelope.")
for name, label, radius, length in (("euclid", "Euclid", 1.5, 4.5), ("iso", "Infrared Space Observatory", 1.0, 4.0)):
    _add(name, label, "barrel", group="partner", sources=(("https://irsa.ipac.caltech.edu/Missions/iso.html" if name == "iso" else SCIENCE+name+"/"),),
         radius=radius, length=length, aperture=.6 if name == "euclid" else .3, panel="side")
for name, label, source in (("ans", "Astronomical Netherlands Satellite", "ans"),
                            ("ariel5", "Ariel V", "ariel5"), ("ariel6", "Ariel VI", "ariel6")):
    _add(name, label, "early", group="partner", sources=(HEASARC+source+".html",),
         radius=.45, length=1.0, style="explorer")

_add("cosi", "Compton Spectrometer and Imager", "detector_bus", group="development",
     sources=(HEASARC+"cosi.html",), size=(1.4,1.4,1.1), layout="generic", count=4)
_add("uvex", "Ultraviolet Explorer", "barrel", group="development",
     sources=("https://science.nasa.gov/astrophysics/programs/physics-of-the-cosmos/community/uvex-selected-as-next-astrophysics-medium-class-explorer/", SCIENCE+"ULTRASAT/"), radius=.7, length=2.7, aperture=.375)
_add("ultrasat", "Ultraviolet Transient Astronomy Satellite", "barrel", group="partner",
     sources=(SCIENCE+"ULTRASAT/",), radius=.6, length=2.2, aperture=.165)
_add("ariel", "Ariel / NASA CASE", "barrel", group="partner", aliases=("case",),
     sources=(FLEET, "https://www.jpl.nasa.gov/go/astronomy-and-physics/missions/"), radius=.8, length=3.0, aperture=.5, panel="side")
_add("athena", "NewAthena", "mast", group="partner", aliases=("newathena",),
     sources=(HEASARC+"athena.html",), length=12, count=1, optics_radius=1.3,
     notes="Conceptual open optical bench; final structure, optics and mission design may differ.")
_add("lisa", "LISA (one spacecraft)", "lisa", group="partner", sources=(SCIENCE+"lisa/",),
     notes="One disc-shaped spacecraft with two laser telescope directions; the three-spacecraft constellation is not generated.")
_add("hwo", "Habitable Worlds Observatory concept", "barrel", group="concept",
     sources=("https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/",), radius=3.5, length=12, aperture=3.0, panel="side",
     notes="Generic 6 m-aperture concept stand-in. NASA has not fixed the flight design; this is not a prediction of it.")

# Hosted payloads: only the astronomy instrument is drawn, never an entire ISS,
# Shuttle or carrier. Separate names distinguish distinct flights/configurations.
_add("nicer", "NICER (ISS payload)", "hosted", group="hosted",
     sources=("https://heasarc.gsfc.nasa.gov/docs/nicer/mission_guide/",),
     features=("56 X-ray concentrator assemblies.",),
     notes="56 apertures and schematic mounting struts; ISS, gimbal articulation and internal mirrors omitted.")
for name, label, source, layout, count in (
    ("astro1", "Astro-1 UV payload", MAST, "astro", 3),
    ("astro2", "Astro-2 UV payload", MAST, "astro", 3),
    ("bbxrt", "Broad Band X-ray Telescope", HEASARC+"bbxrt.html", "generic", 2),
    ("dxs", "Diffuse X-ray Spectrometer", HEASARC+"dxs.html", "generic", 2),
    ("orfeus1", "ORFEUS-SPAS I", MAST, "generic", 1),
    ("orfeus2", "ORFEUS-SPAS II", MAST, "generic", 1),
    ("imaps", "Interstellar Medium Absorption Profile Spectrograph", MAST, "generic", 1),
    ("spartan1", "Spartan 1", HEASARC+"spartan1.html", "generic", 2),
    ("spacelab1", "Spacelab 1 X-ray payload", HEASARC+"spacelab1.html", "generic", 2),
    ("spacelab2", "Spacelab 2 X-ray payload", HEASARC+"spacelab2.html", "generic", 2),
    ("tigeriss", "TIGERISS", FLEET, "generic", 4),
    ("calet", "CALorimetric Electron Telescope", HEASARC+"calet.html", "generic", 1),
    ("ams02", "Alpha Magnetic Spectrometer 02", HEASARC+"ams.html", "generic", 1),
    ("glowbug", "Glowbug gamma-ray telescope", HEASARC+"glowbug.html", "generic", 4),
    ("iss-cream", "Cosmic Ray Energetics and Mass (ISS)", "https://svs.gsfc.nasa.gov/12679/", "generic", 1),
    ("faust", "Far Ultraviolet Space Telescope", "https://www.nasa.gov/mission/sts-45/", "generic", 1),
    ("uvstar", "Ultraviolet Spectrograph Telescope for Astronomical Research", "https://ntrs.nasa.gov/citations/19940014705", "generic", 2),
    ("starlite", "Spectrograph/Telescope for Astronomical Research", SCIENCE+"ieh-3/", "generic", 1),
):
    _add(name, label, "hosted", group="hosted", sources=(source,), layout=layout, count=count,
         notes="Symbolic instrument assembly; host vehicle omitted. Detector shapes and placement are schematic.")

_add("epoch", "EPOCh / EPOXI (Deep Impact spacecraft)", "detector_bus", group="legacy",
     aliases=("epoxi",), sources=(MAST,), size=(1.0,1.0,1.3), layout="swift",
     notes="Schematic dual-imager spacecraft; EPOCh was the stellar/exoplanet observing program on the repurposed Deep Impact spacecraft.")

# Solar observatories that also carried non-solar astronomy experiments are in
# the HEASARC astronomy inventory; pure heliophysics is outside this catalog.
for number in (1, 3, 4, 5, 6, 7, 8):
    _add(f"oso{number}", f"Orbiting Solar Observatory {number}", "early", group="legacy",
         sources=(HEASARC+f"oso{number}.html",), style="oso", radius=.6, length=1.7,
         notes="Shared wheel-and-sail family, including the non-solar astronomy payload context; instrument differences omitted.")
_add("smm", "Solar Maximum Mission", "detector_bus", group="legacy", aliases=("solarmax",),
     sources=(HEASARC+"solarmax.html",), size=(1.6,1.6,2.0), layout="generic", count=3)


def _key(name):
    if not isinstance(name, str):
        return ""
    return "".join(c for c in name.casefold() if c.isalnum())


_LOOKUP = {}
for _name, _info in _CATALOG.items():
    for _alias in (_name, *_info["aliases"]):
        _normalized = _key(_alias)
        if _normalized in _LOOKUP and _LOOKUP[_normalized] != _name:
            raise RuntimeError(f"Conflicting spacecraft alias: {_alias}")
        _LOOKUP[_normalized] = _name


def preset_names(group=None):
    """Canonical names, optionally filtered by the metadata's group."""
    groups = {info["group"] for info in _CATALOG.values()}
    if group is not None and group not in groups:
        raise ValueError(f"unknown preset group {group!r}; choose from {', '.join(sorted(groups))}")
    return tuple(name for name, info in _CATALOG.items() if group is None or info["group"] == group)


def preset_info(name):
    """Independent metadata copy; never fetch sources or external mesh assets."""
    try:
        canonical = _LOOKUP[_key(name)]
    except KeyError:
        raise ValueError(f"unknown mesh preset {name!r}; choose from {', '.join(preset_names())}") from None
    info = deepcopy(_CATALOG[canonical])
    del info["recipe"], info["parameters"]
    return info


def build_preset(name):
    """Build named convex parts; the two original templates live in the view."""
    canonical = preset_info(name)["name"]
    info = _CATALOG[canonical]
    builder = shapes.Builder()
    getattr(shapes, info["recipe"])(builder, **info["parameters"])
    return builder
