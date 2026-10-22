"""C/R1: opt-in rendered camera target test, using literal Cartesian points."""

from io import BytesIO
import os
import shutil

import numpy as np
from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SSATK_BROWSER_TESTS") != "1",
    reason="Set SSATK_BROWSER_TESTS=1 for headless Chrome checks",
)


def test_plotly_follow_keeps_moving_target_at_image_center(tmp_path):
    """R1: two distinct positions project to the same pixel, within 1 raster pixel.

    Autoplay is disabled so a second animation cannot race the frame under test.
    This tests the renderer's camera/data normalization rather than copying its
    conversion equation into a numerical assertion.
    """
    from plotly import graph_objects as go
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from ssatk.plots.camera import AnimationCamera, _plotly_camera

    positions = np.array([[2.0, 4, 1], [-2, -4, -1]])

    def marker(p):
        return go.Scatter3d(
            x=[p[0]],
            y=[p[1]],
            z=[p[2]],
            mode="markers",
            marker=dict(size=10, color="red"),
        )

    fig = go.Figure(
        data=[marker(positions[0])],
        frames=[
            go.Frame(name=str(i), data=[marker(p)], traces=[0])
            for i, p in enumerate(positions)
        ],
    )
    axis = dict(range=[-10, 10], visible=False)
    fig.update_layout(
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, bgcolor="black"),
    )
    _plotly_camera(fig, AnimationCamera("follow", offset=(3, 4, 5)), positions)
    path = tmp_path / "camera.html"
    fig.write_html(path, auto_play=False)
    options = webdriver.ChromeOptions()
    chrome = os.environ.get("SSATK_CHROME_BINARY") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        options.binary_location = chrome
    for arg in (
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--enable-unsafe-swiftshader",
        "--window-size=1100,800",
    ):
        options.add_argument(arg)
    with webdriver.Chrome(options=options) as driver:
        driver.get(path.as_uri())
        wait = WebDriverWait(driver, 45)
        wait.until(
            lambda d: d.execute_script(
                "return !!document.querySelector('.js-plotly-plot')?._fullLayout?.scene?._scene"
            )
        )
        centroids = []
        for i in range(2):
            driver.execute_async_script(
                """const done=arguments[arguments.length-1];
                Plotly.animate(document.querySelector('.js-plotly-plot'),[String(arguments[0])],
                    {mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}})
                    .then(()=>setTimeout(done,200));""",
                i,
            )
            pixels = np.array(
                Image.open(
                    BytesIO(
                        driver.find_element(
                            "css selector", ".js-plotly-plot"
                        ).screenshot_as_png
                    )
                ).convert("RGB")
            )
            mask = (
                (pixels[:, :, 0] > 180)
                & (pixels[:, :, 1] < 70)
                & (pixels[:, :, 2] < 70)
            )
            y, x = np.nonzero(mask)
            assert len(x) > 20
            centre = [float(x.mean()), float(y.mean())]
            np.testing.assert_allclose(centre, [399.5, 299.5], rtol=0, atol=1)
            centroids.append(centre)
        np.testing.assert_allclose(centroids[0], centroids[1], rtol=0, atol=1)


def test_browser_replaced_and_reordered_targets_do_not_leave_a_stale_subject(tmp_path):
    """C: object identity, not its display name, controls tracking and tangent reuse.

    Runs the shipped controller and Three.js in Chrome without a network or
    ephemeris fixture. Replacing a paused target must clear the old chase
    direction; reordering still-visible objects must preserve the selection.
    """
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from ssatk.plots._threejs import runtime_loader_html
    from ssatk.plots.build_satellite_viewer import text

    options = webdriver.ChromeOptions()
    chrome = os.environ.get("SSATK_CHROME_BINARY") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        options.binary_location = chrome
    for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage"):
        options.add_argument(arg)
    with webdriver.Chrome(options=options) as driver:
        page = tmp_path / "three.html"
        page.write_text(f"<html><body>{runtime_loader_html()}</body></html>", encoding="utf-8")
        driver.get(page.as_uri())
        WebDriverWait(driver, 20).until(lambda d: d.execute_script("return !!window.THREE"))
        result = driver.execute_script(text("animation_camera.js") + """
            const shot = new SSATKAnimationCamera(new THREE.PerspectiveCamera(),
                {mode:'chase',target:0,elevation:25,azimuth:35,turns:1},[0,0,1],10);
            const a = {name:'Satellite'}, b = {name:'Satellite'};
            const label = s => s.name;
            shot.selectTarget([a],label);
            shot.lastTangent = new THREE.Vector3(1,0,0);
            const replacement = shot.selectTarget([b],label);
            const replaced = replacement === b && shot.lastTangent === null;
            shot.selectTarget([a,b],label);
            shot.targetSelect.value = '1'; shot.targetSelect.onchange();
            shot.lastTangent = new THREE.Vector3(0,1,0);
            shot.selectTarget([b,a],label);
            const reordered = shot.subject === b && shot.targetSelect.value === '0'
                && shot.lastTangent.y === 1;
            shot.selectTarget([],label);
            const cleared = !shot.subject && shot.lastTangent === null;
            // Names containing the old signature delimiter must remain distinct.
            const c = {name:'A|B'}, d = {name:'C'};
            shot.selectTarget([c,d],label);
            c.name = 'A'; d.name = 'B|C';
            shot.selectTarget([c,d],label);
            const labels = Array.from(shot.targetSelect.options,o=>o.textContent);
            shot.panel.remove();
            return {replaced,reordered,cleared,labels};
        """)
    assert result == dict(
        replaced=True, reordered=True, cleared=True, labels=["A", "B|C"]
    )
