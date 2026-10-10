#!/usr/bin/env python3

import os
import argparse
import time
import math
import pathlib
import warnings
from io import BytesIO

try:
    from PIL import Image, ImageChops
except ImportError as exc:
    Image = None
    ImageChops = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except ImportError as exc:
    webdriver = None
    Options = None
    _SELENIUM_IMPORT_ERROR = exc
else:
    _SELENIUM_IMPORT_ERROR = None


def _require_html_to_gif_dependencies():
    missing = []
    if _PIL_IMPORT_ERROR is not None:
        missing.append("Pillow (PIL)")
    if _SELENIUM_IMPORT_ERROR is not None:
        missing.append("Selenium")
    if missing:
        raise ImportError(
            "html_to_gif requires optional dependencies: "
            + ", ".join(missing)
            + ". Install ssatk[browser]."
        )


def _auto_crop_bbox(img, bg_color=None, fuzz=0):
    """
    Return bounding box of non-background pixels.

    bg_color: (R, G, B) or None to auto-detect from top-left pixel.
    fuzz: tolerance for color differences, 0 = exact match.
    """
    _require_html_to_gif_dependencies()
    img = img.convert("RGB")
    w, h = img.size

    if bg_color is None:
        bg_color = img.getpixel((0, 0))  # assume top-left is background

    # Create solid background image
    bg = Image.new("RGB", (w, h), bg_color)

    # Difference image
    diff = ImageChops.difference(img, bg)
    if fuzz > 0:
        # Expand difference to account for near-bg colors
        diff = diff.point(lambda v: 0 if v <= fuzz else 255)

    bbox = diff.getbbox()
    return bbox  # (left, upper, right, lower) or None if no difference


def html_to_gif(
    *,
    html_path: str,
    out_gif: str,
    duration_s: float = 18.0,
    fps: int = 20,
    viewport=(1280, 720),
    wait_after_load_s: float = 0.8,
    forced_height: int = 2000,   # big enough to capture entire animation
    bg_color=None,               # None = auto-detect from top-left
    fuzz: int = 0,               # increase if background not perfectly uniform
    chrome_binary: str | None = None,
    verbose: bool = False,
):
    """Render a local HTML page to an animated GIF.

    ``chrome_binary`` overrides the browser executable used by Selenium. If it
    is not provided, the ``SSATK_CHROME_BINARY`` environment variable is used
    when set; otherwise Selenium chooses its default browser.

    Set ``verbose=True`` to print crop and output progress messages.
    """
    _require_html_to_gif_dependencies()
    html_path = pathlib.Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(str(html_path))

    url = html_path.as_uri()
    nframes = int(math.ceil(duration_s * fps))
    raw_frames = []

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument(f"--window-size={viewport[0]},{viewport[1]}")
    chrome_binary = chrome_binary or os.environ.get("SSATK_CHROME_BINARY")
    if chrome_binary:
        chrome_options.binary_location = chrome_binary

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(wait_after_load_s)

        # Force large window; we’ll crop later
        driver.set_window_size(viewport[0], forced_height)
        time.sleep(0.3)

        start = time.time()
        for i in range(nframes):
            target_t = start + (i / fps)
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)

            png_bytes = driver.get_screenshot_as_png()
            img = Image.open(BytesIO(png_bytes)).convert("RGB")
            raw_frames.append(img)

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    # --- Compute global crop box across all frames ---
    global_bbox = None
    for img in raw_frames:
        bbox = _auto_crop_bbox(img, bg_color=bg_color, fuzz=fuzz)
        if bbox is None:
            continue
        if global_bbox is None:
            global_bbox = bbox
        else:
            l1, t1, r1, b1 = global_bbox
            l2, t2, r2, b2 = bbox
            global_bbox = (min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2))

    if global_bbox is None:
        # No difference from background; just use original frames
        frames = raw_frames
        warnings.warn("No non-background pixels detected; skipping auto-crop.", UserWarning, stacklevel=2)
    else:
        frames = [img.crop(global_bbox) for img in raw_frames]
        if verbose:
            print("Cropped to bbox:", global_bbox, "-> size", frames[0].size)

    # --- Save GIF ---
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    if verbose:
        print(f"Wrote GIF: {out_gif}  ({len(frames)} frames @ {fps} fps)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a local HTML animation to GIF using Selenium.")
    parser.add_argument("html_path", help="Path to the input HTML file.")
    parser.add_argument("out_gif", help="Path for the output GIF.")
    parser.add_argument("--duration-s", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--viewport", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(1280, 720))
    parser.add_argument("--wait-after-load-s", type=float, default=0.8)
    parser.add_argument("--forced-height", type=int, default=2000)
    parser.add_argument("--fuzz", type=int, default=5)
    parser.add_argument("--chrome-binary", default=None, help="Optional Chrome/Chromium binary path.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    html_to_gif(
        html_path=args.html_path,
        out_gif=args.out_gif,
        duration_s=args.duration_s,
        fps=args.fps,
        viewport=tuple(args.viewport),
        wait_after_load_s=args.wait_after_load_s,
        forced_height=args.forced_height,
        bg_color=None,  # or (255, 255, 255) if you know background is white
        fuzz=args.fuzz,
        chrome_binary=args.chrome_binary,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
