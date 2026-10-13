"""C/R1: mesh catalogue export and browser contracts, not flight-model validation."""

import base64
from html.parser import HTMLParser
from io import BytesIO
import os
import shutil

import numpy as np
from PIL import Image
import pytest

from demos.six_dof import demo_spacecraft_mesh_catalog as demo
from ssatk.plots.spacecraft_view import SpacecraftMesh


class CatalogParser(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.names, self.images, self.scripts = [], [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.names.append(attrs["data-name"])
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "script":
            self.scripts.append(attrs)


@pytest.fixture(scope="module")
def entries():
    return demo.catalog_entries(width=192, height=120)


@pytest.mark.parametrize("size", [(300, 200), (200, 300)])
def test_thumbnail_fits_camera_plane_rectangle_without_stretching(size):
    """R1: 4x2-m camera-plane rectangle must stay 2:1 and occupy 90% of width.

    Edges allow one pixel for supersampling. A translated, uniformly scaled
    copy has the same silhouette; this constrains framing, not flight geometry.
    """
    normal = np.array([1.1, -1.8, 1.4])
    normal /= np.linalg.norm(normal)
    horizontal = np.array([1.8, 1.1, 0])
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    points = np.array([x * horizontal + y * vertical
                       for x, y in [(-2, -1), (2, -1), (2, 1), (-2, 1)]])
    width, height = size
    for vertices in (points, points * 7 + [19, -31, 5]):
        image = np.asarray(demo.render_thumbnail(SpacecraftMesh(vertices, [[0, 1, 2, 3]]), *size))
        y, x = np.where(image[:, :, 3] >= 128)
        # The rectangle is wider than either viewport's aspect, so width limits the fit.
        np.testing.assert_allclose([x.min(), x.max() + 1, y.min(), y.max() + 1],
                                   [0.05 * width, 0.95 * width,
                                    height / 2 - 0.225 * width, height / 2 + 0.225 * width],
                                   rtol=0, atol=1)


def test_catalog_covers_registry_with_decodable_embedded_geometry(entries, tmp_path):
    """C: every selectable canonical ID has an offline, nonempty actual-mesh preview."""
    path = demo.write_catalog(entries, tmp_path / "catalog.html")
    parsed = CatalogParser(path.read_text())
    assert len(parsed.names) == len(set(parsed.names))
    assert set(parsed.names) == set(SpacecraftMesh.presets())
    assert len(parsed.images) == len(parsed.names)
    for source in parsed.images:
        assert source.startswith("data:image/png;base64,")
        image = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
        assert image.size == (192, 120)
        assert np.any(np.asarray(image)[:, :, 3] > 128)
    assert parsed.scripts == [{}]  # Inline logic; no CDN script dependencies.
    # Metadata is text, including literal markup, rather than executable page content.
    info, image = entries[0]
    altered = dict(info, label='<script src="unexpected.js">& mission</script>')
    path = demo.write_catalog([(altered, image)], path)
    assert CatalogParser(path.read_text()).scripts == [{}]
    assert '&lt;script src=&quot;unexpected.js&quot;&gt;' in path.read_text()


@pytest.mark.skipif(os.environ.get("SSATK_BROWSER_TESTS") != "1",
                    reason="Set SSATK_BROWSER_TESTS=1 for headless Chrome checks")
def test_catalog_search_selection_and_downloads_work_offline(entries, tmp_path):
    """C: real browser checks aliases, combined filters, selection and local exports."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait

    path = demo.write_catalog(entries, tmp_path / "spacecraft_mesh_catalog.html")
    png, jpg = demo.write_poster(entries, tmp_path / "spacecraft_mesh_catalog.png", cell_width=180)
    for image_path, format_name in [(png, "PNG"), (jpg, "JPEG")]:
        with Image.open(image_path) as image:
            assert image.format == format_name
            image.verify()
    options = webdriver.ChromeOptions()
    chrome = os.environ.get("SSATK_CHROME_BINARY") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        options.binary_location = chrome
    for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage"):
        options.add_argument(arg)
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    with webdriver.Chrome(options=options) as browser:
        browser.execute_cdp_cmd("Network.enable", {})
        browser.execute_cdp_cmd("Network.emulateNetworkConditions", dict(
            offline=True, latency=0, downloadThroughput=0, uploadThroughput=0))
        browser.get(path.as_uri())
        WebDriverWait(browser, 20).until(lambda d: d.execute_script(
            "return [...document.images].every(i => i.complete && i.naturalWidth > 0)"))
        search = browser.find_element(By.ID, "search")
        search.send_keys("HST")
        visible = browser.find_elements(By.CSS_SELECTOR, ".mesh-card:not([hidden])")
        assert [card.get_attribute("data-name") for card in visible] == ["hubble"]
        visible[0].find_element(By.CSS_SELECTOR, ".choose").click()
        code = browser.find_element(By.ID, "selection-code").text
        assert 'SpacecraftMesh.from_preset("hubble")' in code
        assert 'view.save("hubble.png")' in code
        assert visible[0].find_element(By.CSS_SELECTOR, ".card-code").text == code
        assert visible[0].find_element(By.CSS_SELECTOR, ".choose").get_attribute("aria-pressed") == "true"
        assert not browser.find_element(By.ID, "empty").is_displayed()
        Select(browser.find_element(By.ID, "group")).select_by_value("generic")
        assert browser.find_element(By.ID, "empty").is_displayed()
        search.clear()
        search.send_keys("3u")
        visible = browser.find_elements(By.CSS_SELECTOR, ".mesh-card:not([hidden])")
        assert [card.get_attribute("data-name") for card in visible] == ["cubesat"]
        summary = visible[0].find_element(By.TAG_NAME, "summary")
        browser.execute_script('arguments[0].scrollIntoView({block:"center"})', summary)
        summary.click()
        assert "0.100 x 0.100 x 0.3405 m" in visible[0].text
        for anchor in browser.find_elements(By.CSS_SELECTOR, ".downloads a"):
            assert (tmp_path / anchor.get_attribute("href").split("/")[-1]).is_file()
        assert not [entry for entry in browser.get_log("browser") if entry["level"] == "SEVERE"]
