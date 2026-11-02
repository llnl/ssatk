"""Mission overview and 3D demo: hypothetical Artemis III transfer and IRENE dose.

Run with ``python -m demos.space_environment.demo_artemis_radiation --movie``.
See docs/artemis_radiation_demo.rst for runtime setup and scientific assumptions.
"""

import argparse
import json
from pathlib import Path
import tempfile

import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
import numpy as np

from ssatk import Orbit, rv
from ssatk.coordinates.lunar import get_lunar_rv
from ssatk.coordinates.earth_fixed import gcrf_to_itrf_astropy
from ssatk.constants import EARTH_RADIUS, WGS84_EARTH_OMEGA
from ssatk.data import read_data_text
from ssatk.io.irene import find_irene_runtime, prepare_irene_run, run_irene
from ssatk.orbital_mechanics.transfer_ssapy_function import solve_lambert
from ssatk.plots.build_satellite_viewer import load_star_catalog, load_textures
from ssatk.plots._threejs import classic_scripts_html
from ssatk.plots.figpath import figpath
from ssatk.plots.magfield_plot_3d import _flux_isosurfaces
from ssatk.plots.plotutils import figsave

TITLE = "Artemis III | Earth to Moon | IRENE"
DESCRIPTION = "Hypothetical four-day lunar transfer with mean trapped-particle absorbed tissue dose."
PROMPT = ("Visualize in 3D Artemis 3 going from Earth to the Moon, overlaid on the "
          "radiation belts, plotting cumulative trapped-particle absorbed dose from IRENE by mission day.")


def mission():
    """Four-day SSAPy two-body Lambert transfer to a moving Moon ephemeris."""
    start = Time("2027-06-01T00:00:00", scale="utc")
    # shortcut: illustrative two-body outbound arc, use an approved trajectory for mission analysis.
    seconds = np.unique(np.r_[np.arange(0, 6 * 3600, 30), np.arange(6 * 3600, 4 * 86400, 600), 4 * 86400])
    times = start + seconds * u.s
    rmoon, _ = get_lunar_rv(times)
    arrival = rmoon[-1] * (1 - 2e6 / np.linalg.norm(rmoon[-1]))
    departure = np.array([EARTH_RADIUS + 200e3, 0, 0])
    velocity, _ = solve_lambert(departure, arrival, seconds[-1])
    positions, _ = rv(Orbit(departure, velocity, t=start.gps), times)
    assert np.linalg.norm(positions[-1] - arrival) < 1, "Lambert endpoint did not converge"
    assert np.min(np.linalg.norm(positions, axis=1)) > EARTH_RADIUS
    return times, positions, rmoon


_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artemis III | Earth to Moon</title><style>
*{box-sizing:border-box}body{margin:0;background:#050b14;color:#eaf1fb;font:15px system-ui,sans-serif;overflow:hidden}
header{height:78px;padding:20px 34px;border-bottom:1px solid #263347;display:flex;align-items:center;justify-content:space-between}
.brand{font-size:18px;font-weight:650;letter-spacing:2px}.brand span{color:#6fdfeb}.eyebrow{font-size:11px;letter-spacing:2.3px;color:#889bb8;text-transform:uppercase}
.badge{color:#f6ca80;background:#322616;border:1px solid #69512d;padding:8px 12px;border-radius:20px;font-size:11px;letter-spacing:1px}
main{height:calc(100vh - 144px);display:grid;grid-template-columns:1fr 360px}
#viewport{position:relative;min-width:0;overflow:hidden}#scene{width:100%;height:100%;display:block}
.title{position:absolute;top:29px;left:34px;pointer-events:none}.title h1{font-size:37px;font-weight:400;letter-spacing:-1px;margin:8px 0}.title p{color:#8fa6c4;font-size:13px;margin:0}
.legend{position:absolute;bottom:24px;left:34px;display:flex;gap:22px;font-size:12px;color:#bbcada}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}
aside{padding:29px 26px;border-left:1px solid #263347;background:linear-gradient(145deg,#101d2d,#07101d);overflow:auto}
.day{font-size:56px;letter-spacing:-3px;margin:8px 0 0}.phase{color:#6fdfeb;font-size:13px;margin:5px 0 24px}
.metric{border-top:1px solid #28374a;padding:20px 0}.metric b{font-size:26px;font-weight:400;display:block;margin:8px 0}.metric p{color:#91a3bb;font-size:12px;line-height:1.7;margin:6px 0}
#chart{width:100%;height:175px;margin-top:10px}.pending{color:#f6ca80!important}.status{font-size:12px;line-height:1.8;color:#99abc3;margin-top:15px}
footer{height:66px;border-top:1px solid #263347;padding:13px 34px;display:flex;align-items:center;gap:18px}button{background:#173044;color:#b5f4f7;border:1px solid #326077;border-radius:7px;padding:9px 18px;cursor:pointer}button:focus-visible{outline:2px solid #b5f4f7}input{flex:1;accent-color:#6fdfeb}.clock{font-variant-numeric:tabular-nums;color:#8fa6c4;font-size:12px}
.overview main,.overview footer{visibility:hidden}
#intro{position:absolute;inset:78px 0 0;display:grid;place-items:center;overflow:auto;padding:40px 24px;background:radial-gradient(ellipse at 70% 35%,#183448,#050b14 72%);z-index:3}
.missionbox{width:min(800px,100%)}.missionbox h2{font-weight:400;font-size:clamp(38px,6vw,76px);line-height:1.08;letter-spacing:-2px;margin:20px 0 24px}.missionbox>p{max-width:640px;color:#aabed5;line-height:1.8}.mission-facts{display:flex;flex-wrap:wrap;gap:12px;margin:30px 0}.mission-facts span{border:1px solid #294255;border-radius:7px;padding:12px 16px;color:#c4d7e8;font-size:13px}#start{background:#6fdfeb;color:#07101d;border-color:#6fdfeb;padding:13px 28px;font-size:16px;font-weight:650}#start:disabled{opacity:.5;cursor:wait}.missionbox .intro-note{font-size:12px;margin-top:24px;color:#8299b2}
.label{position:absolute;font-size:12px;letter-spacing:2px;text-shadow:0 2px 5px #000;pointer-events:none}.toast{position:absolute;top:150px;left:34px;color:#f6ca80;font-size:12px;line-height:1.7;max-width:400px}
@media(max-width:1100px){main{grid-template-columns:1fr 300px}aside{padding:24px 18px}.title h1{font-size:28px}.legend{gap:12px}}
@media(max-width:700px){header{height:64px;padding:14px 16px}header>.eyebrow{display:none}.brand{font-size:14px}.badge{font-size:9px;padding:6px 8px}main{height:calc(100dvh - 130px);overflow:auto;grid-template-columns:1fr;grid-template-rows:minmax(230px,1fr) 220px}aside{border-left:0;border-top:1px solid #263347;padding:16px}.title{top:16px;left:16px}.title h1{font-size:23px}.title p{max-width:280px;font-size:11px}.legend{left:16px;bottom:12px;gap:8px;flex-wrap:wrap;font-size:10px}.toast{top:125px;left:16px;max-width:280px;font-size:11px}footer{padding:10px 12px;gap:8px}footer button{padding:9px 10px;font-size:11px}input{min-width:0}#intro{inset:64px 0 0}}
</style></head><body class="overview">
<header><div class="brand">SSAPy <span>TOOLKIT</span></div><div class="eyebrow">Earth to Moon</div><div class="badge">HYPOTHETICAL ARTEMIS III</div></header>
<main inert><section id="viewport"><canvas id="scene" aria-label="3D Earth, radiation belts, catalog stars and lunar transfer" tabindex="-1"></canvas>
<div class="title"><div class="eyebrow">Earth → Moon / outbound transfer</div><h1>A journey through the belts.</h1><p>SSAPy trajectory · Moon ephemeris · catalog stars · synchronized mission timeline</p></div>
<div class="toast" id="notice"></div><div class="label" id="earthlabel">EARTH</div><div class="label" id="moonlabel">MOON</div>
<div class="legend"><span><i class="dot" style="background:#ffc775"></i>AP9 protons &gt;10 MeV</span><span><i class="dot" style="background:#639efe"></i>AE9 electrons &gt;1 MeV</span><span><i class="dot" style="background:#6fdfeb"></i>Outbound trajectory</span></div></section>
<aside><div class="eyebrow">Mission day</div><div class="day" id="day">0.00</div><div class="phase" id="phase">Translunar injection</div>
<div class="metric"><div class="eyebrow">Distance from Earth center</div><b id="distance">—</b><p>Earth and Moon at physical scale.<br>Spacecraft marker enlarged for visibility.</p></div>
<div class="metric"><div class="eyebrow">Cumulative trapped-particle dose</div><b id="dose" class="pending">Pending IRENE</b><p id="dosecaption">AE9 electrons + AP9 protons<br>3 mm aluminum · 4π sphere · tissue</p>
<canvas id="chart" aria-label="Cumulative absorbed dose in mGy versus mission day"></canvas>
<p id="chartnote" class="pending">No model output available.</p></div>
<div class="status" id="status"></div></aside></main>
<footer inert><button id="play" type="button">Play</button><input id="scrub" type="range" min="10" max="42" step="0.01" value="10" aria-label="Movie time"><span class="clock" id="clock">00:00 / 00:42</span><button id="restart" type="button">Mission overview</button></footer>
<section id="intro" aria-labelledby="mission-heading"><div class="missionbox"><div class="eyebrow">Explore the mission</div><h2 id="mission-heading">From Earth to Moon.<br>Through the radiation belts.</h2><p>Follow an illustrative four-day lunar transfer. Watch the spacecraft leave Earth, cross the radiation belts, and approach the Moon alongside its cumulative trapped-particle dose.</p><div class="mission-facts"><span>4-day outbound transfer</span><span>AE9 / AP9 radiation belts</span><span>3 mm aluminum shielding</span></div><button id="start" type="button" disabled>Start demo</button><p id="intro-status" role="status">Preparing the scene…</p><p class="intro-note">Hypothetical lunar scenario; NASA's current Artemis III plan is an Earth-orbit test.<br>Dose appears when valid official IRENE output is available. Solar particle events and galactic cosmic rays are excluded.</p></div></section>
__SCRIPTS__</body></html>"""


def main(make_figures=True, fast=False, *, directory=None, executable=None, model_data=None, movie=False):
    """Build a self-contained viewer, IRENE inputs, and optionally an MP4."""
    if not make_figures:
        return {"figures": [], "skipped": True, "reason": "figures_disabled"}
    directory = Path(directory or figpath("artemis_radiation")).expanduser().resolve()
    runtime = find_irene_runtime(executable=executable, model_data=model_data)
    directory.mkdir(parents=True, exist_ok=True)
    with iers.conf.set_temp("auto_download", False):
        times, positions, moon = mission()
        stars = load_star_catalog(when=times[0].utc.to_datetime(), frame="gcrf")
        run_directory = Path(tempfile.mkdtemp(prefix="irene_", dir=directory)) / "run"
        prepare_irene_run(positions, times, run_directory)
        rotation = gcrf_to_itrf_astropy(np.eye(3), times[0])
    result = None
    if runtime is not None:
        result = run_irene(run_directory, executable=runtime[0], model_data=runtime[1])
    belts = []
    if runtime is not None:
        surfaces = _flux_isosurfaces(times[0].utc.to_datetime(), None, mesh=True,
                                    grid_n=36 if fast else 48,
                                    executable=runtime[0], model_data=runtime[1])
        if not surfaces:
            raise ValueError("IRENE returned no positive belt flux; no substitute geometry rendered")
        for surface in surfaces:
            belts.append({"positions": (np.column_stack([surface.x, surface.y, surface.z])
                                         / (EARTH_RADIUS / 1000)).tolist(),
                          "indices": np.column_stack([surface.i, surface.j, surface.k]).ravel().tolist(),
                          "color": surface.color, "metadata": surface.meta})
    payload = {
        "prompt": PROMPT, "seconds": ((times.gps - times[0].gps)).tolist(),
        "r": (positions / EARTH_RADIUS).tolist(), "moon": (moon / EARTH_RADIUS).tolist(),
        "rotation": rotation.tolist(), "belts": belts, "earth_radius_km": EARTH_RADIUS / 1000,
        "earth_omega": WGS84_EARTH_OMEGA,
        "stars": stars,
        "dose": result, "earth_texture": "data:image/jpeg;base64," + load_textures()["day"],
    }
    scene_source = read_data_text("artemis_radiation_scene.js", package="ssatk", data_root="plots")
    data = "const DATA=" + json.dumps(payload, allow_nan=False).replace("<", "\\u003c") + ";"
    html = _HTML.replace("__SCRIPTS__", classic_scripts_html(data, scene_source))
    output = directory / "artemis_radiation.html"
    figsave(html, output)
    (directory / "demo.json").write_text(json.dumps({"title": TITLE, "prompt": PROMPT,
        "scenario": "hypothetical", "epoch": times[0].utc.isot, "duration_days": 4,
        "trajectory": "SSAPy two-body Lambert", "radiation": "calculated" if result else "pending official IRENE",
        "belt_geometry": "IRENE mean integral flux contours" if belts else "pending official IRENE",
        "belt_contours": [belt["metadata"] for belt in belts],
        "starfield": {"frame": stars["frame"], "epoch": stars["epoch"], "count": len(stars["s"])},
        "irene_run": run_directory.relative_to(directory).as_posix()}, indent=2), encoding="utf-8")
    if movie:
        record_movie(output, directory / "artemis_radiation.mp4", fps=15 if fast else 24)
    print(f"Viewer: {output}\nIRENE inputs: {run_directory}\nRadiation: {'calculated' if result else 'pending official runtime'}")
    return {"figures": [str(output)], "irene_run": str(run_directory), "dose": result, "skipped": False}


def record_movie(html, output, *, fps=24):
    """Capture actual browser rendering at deterministic movie times."""
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from ssatk.plots.write_videos import write_video

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-first-run")
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    with webdriver.Edge(options=options) as browser:
        browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride",
                               {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False})
        browser.get(Path(html).as_uri())
        WebDriverWait(browser, 30).until(lambda driver: driver.execute_script("return window.demoReady === true"))
        browser.execute_script("window.demoRecording = true")
        with tempfile.TemporaryDirectory(prefix="frames_", dir=Path(output).parent) as frames:
            for frame in range(42 * fps + 1):
                browser.execute_async_script("const done=arguments[arguments.length-1]; window.setMovieTime(arguments[0]); requestAnimationFrame(()=>requestAnimationFrame(done));", frame / fps)
                if not browser.save_screenshot(str(Path(frames) / f"frame_{frame:05d}.png")):
                    raise RuntimeError(f"Movie screenshot failed at frame {frame}")
                if frame in (0, 8 * fps, 16 * fps, 30 * fps, 42 * fps):
                    if not browser.save_screenshot(str(Path(output).parent / f"preview_{frame // fps:02d}.png")):
                        raise RuntimeError(f"Preview screenshot failed at frame {frame}")
                if frame % (5 * fps) == 0:
                    print(f"Recording: {frame / fps:.0f}/42 seconds", flush=True)
            write_video(str(output), frames, fps=fps, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--irene-executable", type=Path)
    parser.add_argument("--irene-data", type=Path)
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    # gallery-call: artemis_radiation.html
    main(directory=args.output, executable=args.irene_executable, model_data=args.irene_data,
         movie=args.movie, fast=args.fast)
    # gallery-end
