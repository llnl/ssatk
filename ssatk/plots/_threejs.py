"""The single vendored three.js runtime shared by every SSATK WebGL viewer.

``three.core.min.js`` and ``three.module.min.js`` are the unmodified r185
builds from the npm ``three`` package, and ``OrbitControls.js`` is the r185
``examples/jsm/controls`` add-on (MIT, see ``three_license.txt``). r185 is the
newest release that publishes minified builds; r186 ships only unminified
files, and its 1.46 MB core exceeds the repository's 1 MB file limit.

Every viewer loads this one copy; upgrade it for all viewers together. Pages
stay self-contained: the files are embedded through an import map of data
URIs, or mapped to served URLs.
"""
from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

THREE_REVISION = "185"
RUNTIME_FILES = ("three.core.min.js", "three.module.min.js", "OrbitControls.js")
_DIR = Path(__file__).resolve().parent
_SPECIFIERS = {
    "three": "three.module.min.js",
    "three/core": "three.core.min.js",
    "three/addons/controls/OrbitControls.js": "OrbitControls.js",
}


def _data_uri(source: str) -> str:
    return "data:text/javascript;base64," + base64.b64encode(source.encode("utf-8")).decode("ascii")


@lru_cache(maxsize=1)
def _embedded_imports() -> dict[str, str]:
    module = (_DIR / "three.module.min.js").read_text(encoding="utf-8")
    relative = 'from"./three.core.min.js"'
    if relative not in module:
        raise RuntimeError("unexpected three.module.min.js layout; core import not found")
    # A data: URL module cannot resolve a relative import, so the core is
    # reached through the import map instead.
    sources = {
        "three": module.replace(relative, 'from"three/core"'),
        "three/core": (_DIR / "three.core.min.js").read_text(encoding="utf-8"),
        "three/addons/controls/OrbitControls.js": (_DIR / "OrbitControls.js").read_text(encoding="utf-8"),
    }
    return {specifier: _data_uri(source) for specifier, source in sources.items()}


def import_map_html(base_url: str | None = None) -> str:
    """``<script type="importmap">`` for ``three`` and its OrbitControls add-on.

    With ``base_url`` the specifiers map to served copies of
    :data:`RUNTIME_FILES` under that URL; otherwise the files are embedded.
    """
    if base_url is None:
        imports = _embedded_imports()
    else:
        root = base_url.rstrip("/")
        imports = {specifier: f"{root}/{name}" for specifier, name in _SPECIFIERS.items()}
    return f'<script type="importmap">{json.dumps({"imports": imports})}</script>'


CLASSIC_SCRIPT_TYPE = "text/x-ssatk-classic"


def runtime_loader_html(base_url: str | None = None) -> str:
    """Import map plus a module that runs the page's classic three.js scripts.

    The viewers were written as classic scripts whose top-level functions are
    page globals (the headless checks call them). three.js now ships only as
    ES modules, so the module publishes ``window.THREE``, with
    ``OrbitControls`` attached, then executes every
    ``<script type="text/x-ssatk-classic">`` block in document order.
    """
    loader = (
        '<script type="module">\n'
        'import * as THREE from "three";\n'
        'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\n'
        "window.THREE = Object.assign({}, THREE, { OrbitControls });\n"
        f'for (const block of document.querySelectorAll(\'script[type="{CLASSIC_SCRIPT_TYPE}"]\')) {{\n'
        '  const script = document.createElement("script");\n'
        "  script.text = block.textContent;\n"
        "  document.body.appendChild(script);\n"
        "}\n"
        "</script>"
    )
    return import_map_html(base_url) + "\n" + loader


def classic_scripts_html(*sources: str, base_url: str | None = None) -> str:
    """Embed ``sources`` as classic three.js scripts followed by the runtime loader."""
    blocks = "".join(
        f'<script type="{CLASSIC_SCRIPT_TYPE}">' + source.replace("</script>", "<\\/script>") + "</script>\n"
        for source in sources
    )
    return blocks + runtime_loader_html(base_url)
