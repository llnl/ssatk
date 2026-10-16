#!/usr/bin/env python3
"""
Demo: self-contained Three.js satellite viewer with an 8K NASA Earth map.

Run scripts.prepare_gallery_data first for the 8192 x 4096 day texture;
otherwise the installed 2K texture is used. The page accepts satellite
catalogs from JSON, CSV, or HDF5. Its PNG is a capture of the actual viewer,
requiring the browser extra and Chrome; set SSATK_CHROME_BINARY to point at a
specific browser, or the capture is skipped and only the HTML viewer is written.
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import shutil
import sys
from pathlib import Path

from ssatk.plots.build_satellite_viewer import build
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def _browser_unavailable_errors():
    """Selenium exceptions that mean no browser or driver could be started.

    Selenium raises NoSuchDriverException when no driver matches the
    configured binary, and SessionNotCreatedException when the launched
    process is not a working Chrome. The import is lazy so the module loads
    without selenium; an ``except`` clause's expression is only evaluated
    when reached, after the ImportError clause has already been tried.
    """
    from selenium.common.exceptions import NoSuchDriverException, SessionNotCreatedException
    return (NoSuchDriverException, SessionNotCreatedException)


def _save_preview(html_path, png_path):
    """Render a fixed-epoch, sunlit preview using the same WebGL scene."""
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait

    options = webdriver.ChromeOptions()
    # SSATK_CHROME_BINARY matches ssatk/io/html_to_gif.py so both tools point
    # at the same browser when a caller sets it.
    chrome = (os.environ.get("SSATK_CHROME_BINARY")
              or shutil.which("google-chrome")
              or shutil.which("chromium"))
    if chrome:
        options.binary_location = chrome
    for flag in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                 "--enable-unsafe-swiftshader", "--window-size=1600,1100"):
        options.add_argument(flag)
    with webdriver.Chrome(options=options) as driver:
        driver.get(html_path.resolve().as_uri())
        WebDriverWait(driver, 60).until(lambda d: d.execute_script("""
            return typeof earthMesh !== 'undefined' && earthMesh &&
                ['dayTexture', 'nightTexture', 'specularTexture'].every(
                    k => earthMesh.material.uniforms[k].value.image?.complete) &&
                cloudMesh?.material.map.image?.complete;
        """))
        driver.execute_async_script("""
            const done = arguments[0];
            setSimTime(Date.UTC(2026, 0, 1, 12));
            if (!isPaused) togglePause();
            camTheta = -satellite.gstime(new Date(getCurrentSimMs())) - 20 * Math.PI / 180;
            camPhi = 65 * Math.PI / 180;
            camDist = 18500;
            requestAnimationFrame(() => requestAnimationFrame(done));
        """)
        if not driver.save_screenshot(str(png_path)):
            raise RuntimeError("Satellite viewer PNG could not be saved")


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    if not make_figures:
        return {"html": None, "png": None, "skipped": True,
                "reason": "figures_disabled"}

    html_path = Path(figpath("demo_satellite_viewer.html"))
    # gallery-call: *.html
    written = build(out_path=str(html_path), verbose=True)
    # gallery-end
    png_path = Path(figpath("demo_satellite_viewer.png"))
    try:
        # gallery-call: *.png
        _save_preview(html_path, png_path)
        # gallery-end
    except ImportError:
        return {"html": str(written), "png": None, "skipped": True,
                "reason": "Selenium is not installed; install ssatk[browser] "
                          "to capture the PNG preview."}
    except _browser_unavailable_errors():
        return {"html": str(written), "png": None, "skipped": True,
                "reason": "No usable Chrome/driver could be started; install "
                          "Chrome/Chromium or set SSATK_CHROME_BINARY."}
    return {"html": str(written), "png": str(png_path), "skipped": False}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
