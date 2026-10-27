"""Deterministic camera tours of the magnetic-field demo's fixed model."""

from io import BytesIO
import math
import os
from pathlib import Path
import shutil


def save_field_videos(html_path, *, fast=False):
    """Stream two camera tours to MP4 without retracing or evolving the field."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait

    html_path = Path(html_path).resolve()
    options = webdriver.ChromeOptions()
    chrome = (os.environ.get("SSATK_CHROME_BINARY")
              or shutil.which("google-chrome") or shutil.which("chromium"))
    if chrome:
        options.binary_location = chrome
    for flag in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                 "--enable-unsafe-swiftshader", "--hide-scrollbars"):
        options.add_argument(flag)
    fps, duration = (12, 12) if fast else (24, 15)
    width, height = 1280, 960
    outputs = []
    with webdriver.Chrome(options=options) as driver:
        driver.set_script_timeout(60)
        driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", dict(
            width=width, height=height, deviceScaleFactor=1, mobile=False))
        driver.get(html_path.as_uri())
        WebDriverWait(driver, 60).until(lambda d: d.execute_script("""
            return typeof Plotly !== 'undefined' &&
                !!document.querySelector('.js-plotly-plot')?._fullLayout?.scene;
        """))
        driver.execute_async_script("""
            const done = arguments[0];
            document.body.style.margin = '0';
            window.fieldPlot = document.querySelector('.js-plotly-plot');
            window.fieldEye = {...fieldPlot.layout.scene.camera.eye};
            window.beltIndices = fieldPlot.data.flatMap((trace, i) =>
                /^(Inner|Outer) belt/.test(trace.name || '') ? [i] : []);
            Plotly.relayout(fieldPlot, {width: 1280, height: 960,
                'scene.dragmode': false}).then(() => done());
        """)
        belt_count = driver.execute_script("return beltIndices.length")
        if belt_count != 2:
            raise RuntimeError(f"Expected two illustrative belt surfaces, found {belt_count}")
        for name, belts in (("lines", False), ("belts", True)):
            path = html_path.with_name(f"demo_magfield_{name}.mp4")
            driver.execute_async_script("""
                const belts = arguments[0], done = arguments[1];
                const subject = belts ? 'Internal field + illustrative L-shell belts' : 'Internal magnetic field lines';
                Plotly.restyle(fieldPlot, {visible: belts}, beltIndices)
                    .then(() => Plotly.relayout(fieldPlot, {
                        'title.text': subject + '<br><span style="font-size:14px">IGRF 2025 · camera tour · fixed field, no time evolution</span>'
                    })).then(() => done());
            """, belts)
            # Stream frames so export memory is independent of video duration.
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1, pixelformat="yuv420p") as writer:
                for index in range(fps * duration):
                    phase = 2 * math.pi * index / (fps * duration)
                    driver.execute_async_script("""
                        const phase = arguments[0], done = arguments[1];
                        const radius = Math.hypot(fieldEye.x, fieldEye.y, fieldEye.z);
                        const start = Math.atan2(fieldEye.y, fieldEye.x);
                        const elevation = (25 + 20 * Math.sin(phase)) * Math.PI / 180;
                        const eye = {
                            x: radius * Math.cos(elevation) * Math.cos(start + phase),
                            y: radius * Math.cos(elevation) * Math.sin(start + phase),
                            z: radius * Math.sin(elevation)
                        };
                        Plotly.relayout(fieldPlot, {'scene.camera.eye': eye})
                            .then(() => requestAnimationFrame(() => requestAnimationFrame(done)));
                    """, phase)
                    with Image.open(BytesIO(driver.get_screenshot_as_png())) as frame:
                        writer.append_data(np.asarray(frame.convert("RGB")))
                    if index % (fps * 3) == 0:
                        print(f"  {path.name}: {index}/{fps * duration} frames", flush=True)
            outputs.append(str(path))
            print(f"  Video: {path}", flush=True)
    return outputs
