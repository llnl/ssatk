"""Cross-browser WebGL acceptance runner for V22.2 eclipse products.

Each product/mode pair runs in its own Python and Chromium process.  Large
self-contained eclipse pages allocate sizeable JavaScript and WebGL resources;
process isolation prevents a prior product's retained renderer state from
poisoning later checks and makes failures diagnosable rather than aborting the
entire matrix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

RAF_PATCH = r'''<script>
(() => {
 const nativeRAF=window.requestAnimationFrame.bind(window); let budget=6,pending=null,scheduled=false,serial=1;
 function schedule(){ if(scheduled||budget<=0||!pending)return; const item=pending; pending=null; scheduled=true; budget--;
 nativeRAF((ts)=>{scheduled=false; try{item.cb(ts)}finally{schedule()}}); }
 window.__addRafBudget=(n=1)=>{budget+=Number(n)||0;schedule();return budget};
 window.__rafBudget=()=>budget;
 window.requestAnimationFrame=(cb)=>{const id=serial++;pending={id,cb};schedule();return id};
 window.cancelAnimationFrame=()=>{};
})();
</script>'''

MODES = {
    "chromium-default": [
        "--no-sandbox", "--disable-dev-shm-usage", "--enable-webgl",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
    ],
    "chromium-swiftshader": [
        "--no-sandbox", "--disable-dev-shm-usage", "--ignore-gpu-blocklist",
        "--enable-unsafe-swiftshader", "--use-gl=swiftshader", "--enable-webgl",
        "--disable-gpu-sandbox", "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
    ],
}


def _load_large_html(page, text: str, chunk: int = 500_000) -> None:
    """Execute the sealed offline HTML through a browser-native Blob URL.

    File and localhost navigation are administratively blocked in the build
    image.  Blob navigation still creates a real document and, unlike
    ``document.write``, does not stall while a large inline ES module and its
    data-URI assets initialize.
    """
    page.set_content("<!doctype html><html><body>bootstrap</body></html>")
    page.evaluate("window.__eclipseChunks=[]")
    for index in range(0, len(text), chunk):
        page.evaluate("c=>window.__eclipseChunks.push(c)", text[index:index+chunk])
    page.evaluate(
        "() => {const blob=new Blob(window.__eclipseChunks,{type:'text/html'});"
        "window.__eclipseChunks=null;location.href=URL.createObjectURL(blob);}"
    )
    page.wait_for_selector("#status", timeout=60_000)


def _frames(page, count: int = 2, wait_ms: int = 100) -> None:
    try:
        page.evaluate("n=>window.__addRafBudget?window.__addRafBudget(n):null", count)
    except Exception:
        pass
    page.wait_for_timeout(wait_ms)


def _portable_path(path: Path) -> str:
    """Return a release-portable path for machine-readable browser evidence."""
    resolved = Path(path).expanduser().resolve()
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[1])
    for base in candidates:
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


def _rect(box):
    return None if box is None else {key: float(box[key]) for key in ("x", "y", "width", "height")}


def _overlap(a, b) -> bool:
    if not a or not b:
        return False
    return not (a["x"]+a["width"] <= b["x"] or b["x"]+b["width"] <= a["x"]
                or a["y"]+a["height"] <= b["y"] or b["y"]+b["height"] <= a["y"])


def _run_product(browser, html_path: Path, output_dir: Path, label: str) -> dict[str, object]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    page = browser.new_page(viewport={"width": 1100, "height": 720}, device_scale_factor=1)
    result: dict[str, object] = {
        "label": label, "html": _portable_path(html_path), "html_bytes": html_path.stat().st_size,
        "console_errors": [], "page_errors": [], "cameras": {}, "toggles": {}, "timeline": {},
    }
    page.on("console", lambda msg: result["console_errors"].append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: result["page_errors"].append(str(exc)))
    try:
        text = html_path.read_text(encoding="utf-8").replace(
            '<script type="module">', RAF_PATCH+'<script type="module">', 1
        )
        _load_large_html(page, text)
        try:
            page.wait_for_function(
                "() => {const s=document.querySelector('#status'); const t=s?.textContent||''; "
                "return t.includes('Ready')||t.includes('Basic rendering active')||t.includes('HDR postprocessing was disabled');}",
                timeout=180_000,
            )
        except PlaywrightTimeoutError:
            result["load_timeout"] = True
        _frames(page, 4, 350)
        result["status"] = page.locator("#status").inner_text() if page.locator("#status").count() else ""
        result["webgl"] = page.evaluate("""() => {
          const c=document.querySelector('#gl'); const g2=c&&c.getContext('webgl2'); const g=g2||(c&&c.getContext('webgl'));
          if(!g)return {available:false}; const e=g.getExtension('WEBGL_debug_renderer_info');
          return {available:true,webgl2:!!g2,version:g.getParameter(g.VERSION),
                  vendor:e?g.getParameter(e.UNMASKED_VENDOR_WEBGL):g.getParameter(g.VENDOR),
                  renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)};
        }""")
        title = _rect(page.locator(".title").bounding_box())
        panel = _rect(page.locator(".panel").bounding_box())
        result["layout"] = {
            "title_panel_overlap": _overlap(title, panel),
            "top_align_items": page.evaluate("getComputedStyle(document.querySelector('#top')).alignItems"),
        }
        page.screenshot(path=str(output_dir / f"{label}.png"))
        slider_probe = page.locator("#timeline")
        animated_page = bool(slider_probe.count() and float(slider_probe.get_attribute("max") or 0) > float(slider_probe.get_attribute("min") or 0))
        camera_set = ("observer", "system") if animated_page and page.locator('[data-camera="observer"]').count() else (("system", "earth") if animated_page else ("system", "earth", "moon", "optics", "true", "sun", "observer"))
        for camera in camera_set:
            button = page.locator(f'[data-camera="{camera}"]')
            if not button.count():
                result["cameras"][camera] = "not-present"
                continue
            button.click(timeout=10_000); _frames(page, 1, 45); result["cameras"][camera] = "passed"
        for camera in ("system", "earth", "moon", "optics", "true", "sun", "observer"):
            result["cameras"].setdefault(camera, "skipped-animated" if animated_page else "not-present")
        toggle_ids = ("raysToggle", "boundaryToggle") if animated_page else ("raysToggle", "boundaryToggle", "atmosphereToggle", "starToggle", "labelsToggle", "scientificToggle")
        for ident in toggle_ids:
            item = page.locator("#"+ident)
            if not item.count():
                result["toggles"][ident] = "not-present"
                continue
            initial = item.is_checked(); item.click(); _frames(page, 1, 40); changed = item.is_checked()
            item.click(); _frames(page, 1, 40); restored = item.is_checked()
            result["toggles"][ident] = "passed" if changed != initial and restored == initial else "failed"
        photometry = page.locator("#photometrySelect")
        if photometry.count():
            initial_photometry = photometry.input_value()
            photometry.select_option("uniform"); _frames(page, 2, 80)
            uniform_value = photometry.input_value()
            uniform_metric = page.locator("#metricPhotometry").inner_text() if page.locator("#metricPhotometry").count() else ""
            photometry.select_option("limb"); _frames(page, 2, 80)
            limb_value = photometry.input_value()
            limb_metric = page.locator("#metricPhotometry").inner_text() if page.locator("#metricPhotometry").count() else ""
            result["photometry_toggle"] = {
                "initial": initial_photometry,
                "uniform": uniform_value,
                "limb": limb_value,
                "uniform_metric": uniform_metric,
                "limb_metric": limb_metric,
                "passed": uniform_value == "uniform" and limb_value == "limb" and uniform_metric != limb_metric,
            }
        else:
            result["photometry_toggle"] = {"passed": False, "reason": "control not present"}
        slider = page.locator("#timeline")
        if slider.count():
            lo, hi = float(slider.get_attribute("min") or 0), float(slider.get_attribute("max") or 0)
            before = page.locator("#timeLabel").inner_text(); midpoint = 0.5*(lo+hi)
            slider.evaluate(
                "(el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));"
                "el.dispatchEvent(new Event('change',{bubbles:true}));}", midpoint
            )
            _frames(page, 3, 150); after = page.locator("#timeLabel").inner_text()
            result["timeline"] = {"min": lo, "max": hi, "changed": before != after, "before": before, "after": after}
        result["final_status"] = page.locator("#status").inner_text() if page.locator("#status").count() else ""
        toggle_pass = all(value != "failed" for value in result["toggles"].values())
        timeline_pass = result["timeline"].get("max", 0) <= result["timeline"].get("min", 0) or result["timeline"].get("changed", False)
        result["passed"] = bool(
            result.get("webgl", {}).get("available") and not result["console_errors"] and not result["page_errors"]
            and not result["layout"]["title_panel_overlap"] and result["layout"]["top_align_items"] == "flex-start"
            and toggle_pass and timeline_pass and result.get("photometry_toggle", {}).get("passed", False)
            and not result.get("load_timeout", False)
        )
    except Exception as exc:
        result["worker_exception"] = f"{type(exc).__name__}: {exc}"
        result.setdefault("layout", {})
        result.setdefault("webgl", {"available": False})
        result["passed"] = False
    finally:
        try:
            page.close()
        except Exception:
            pass
    return result


def _worker(html: Path, output_dir: Path, label: str, mode: str, result_path: Path) -> int:
    executable = next(filter(None, (
        shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "chrome", "msedge")
    )), None)
    payload: dict[str, object]
    if not executable:
        payload = {"label": label, "available": False, "passed": None,
                   "reason": "Chromium-compatible executable not installed"}
    else:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                probe_browser = playwright.chromium.launch(
                    executable_path=executable, headless=False, args=MODES[mode]
                )
                preflight = probe_browser.new_page()
                preflight.set_content('<canvas id="c"></canvas>')
                capability = preflight.evaluate("""() => {
                  const c=document.querySelector('#c'); const g2=c.getContext('webgl2'); const g=g2||c.getContext('webgl');
                  if(!g)return {available:false}; const e=g.getExtension('WEBGL_debug_renderer_info');
                  const renderer=e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER);
                  return {available:true,webgl2:!!g2,renderer,
                          hardware_accelerated:!String(renderer).toLowerCase().includes('swiftshader')};
                }""")
                preflight.close()
                try:
                    probe_browser.close()
                except Exception:
                    pass
                if capability.get("available"):
                    # The product runs in a fresh browser: probing a WebGL
                    # context in the same process can destabilize ANGLE/
                    # SwiftShader before a large self-contained scene loads.
                    product_browser = playwright.chromium.launch(
                        executable_path=executable, headless=False, args=MODES[mode]
                    )
                    result = _run_product(product_browser, html.resolve(), output_dir, label)
                    payload = {"available": True, "capability": capability, "result": result, "passed": result["passed"]}
                    try:
                        product_browser.close()
                    except Exception:
                        pass
                else:
                    payload = {"available": False, "capability": capability, "passed": None,
                               "reason": "No WebGL context is available"}
        except Exception as exc:
            payload = {"available": True, "passed": False,
                       "worker_exception": f"{type(exc).__name__}: {exc}"}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if payload.get("passed") in {True, None} else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=tuple(MODES), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        if len(args.html) != 1 or not args.mode or not args.label or not args.result:
            raise SystemExit("worker mode requires one HTML, --mode, --label, and --result")
        return _worker(args.html[0], args.output_dir, args.label, args.mode, args.result)
    if not args.html:
        raise SystemExit("at least one HTML product is required")

    matrix: dict[str, object] = {
        "schema": "ssapy-toolkit.eclipse.browser-matrix/2.2",
        "browsers": {}, "products": [_portable_path(path) for path in args.html],
        "process_isolation": "one Python and Chromium process per product and launch mode",
    }
    for mode in MODES:
        results = []
        for path in args.html:
            label = f"{mode}_{path.stem}"
            result_path = args.output_dir / f"{label}.json"
            worker_command = [
                sys.executable, "-m", "ssapy_toolkit.plots.eclipse_browser_matrix", str(path.resolve()),
                "--output-dir", str(args.output_dir.resolve()), "--worker", "--mode", mode,
                "--label", label, "--result", str(result_path.resolve()),
            ]
            xvfb = shutil.which("xvfb-run")
            command = [xvfb, "-a", *worker_command] if xvfb else worker_command
            try:
                completed = subprocess.run(
                    command, cwd=str(Path.cwd()), env=os.environ.copy(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                completed = None
                item = {
                    "available": True, "passed": False,
                    "worker_exception": "browser worker exceeded 300 seconds",
                    "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                }
            if result_path.is_file():
                item = json.loads(result_path.read_text(encoding="utf-8"))
            elif completed is not None:
                item = {"available": True, "passed": False,
                        "worker_exception": f"worker exited {completed.returncode} without a result",
                        "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
            results.append(item)
        available = [item for item in results if item.get("available")]
        matrix["browsers"][mode] = {
            "available": bool(available),
            "passed": bool(available and all(item.get("passed") for item in available)),
            "results": results,
        }
    matrix["browsers"]["firefox"] = {
        "available": bool(shutil.which("firefox")), "passed": None,
        "reason": "Firefox executable is not installed in this validation image" if not shutil.which("firefox") else "external runner required",
    }
    matrix["browsers"]["safari-webkit"] = {
        "available": False, "passed": None,
        "reason": "Safari/WebKit hardware validation requires macOS and is external to this Linux build image",
    }
    available_chromium = [
        value for key, value in matrix["browsers"].items()
        if key.startswith("chromium") and value.get("available")
    ]
    matrix["passed"] = bool(available_chromium and all(value.get("passed") for value in available_chromium))
    output = args.output_dir / "browser_matrix_v22_2.json"
    output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    md = [
        "# V22.2 browser/WebGL matrix", "",
        f"Overall available-browser result: **{'passed' if matrix['passed'] else 'failed'}**", "",
        "Each Chromium product/mode pair ran in a fresh browser and Python process.", "",
    ]
    for name, value in matrix["browsers"].items():
        if value.get("passed") is True:
            state = "passed"
        elif value.get("available") is False:
            state = value.get("reason", "unavailable")
        else:
            state = "failed or not exercised"
        md.append(f"- **{name}:** {state}")
    (args.output_dir / "BROWSER_MATRIX_V20.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps(matrix, indent=2))
    return 0 if matrix["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
