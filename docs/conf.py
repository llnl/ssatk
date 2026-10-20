import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Space Situational Awareness Toolkit"
author = "Travis R. Yeager"
copyright = f"{datetime.now(timezone.utc).year}, {author}"
version_match = re.search(
    r'^version\s*=\s*["\']([^"\']+)["\']\s*$',
    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    re.MULTILINE,
)
if version_match is None:
    raise RuntimeError("project version is missing from pyproject.toml")
release = version_match.group(1)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

autosummary_generate = ["api.rst"]
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

master_doc = "index"

# If some heavy imports break docs, uncomment and expand this list:
# autodoc_mock_imports = [
#     "cv2",
#     "ipyvolume",
#     "matplotlib",
#     "rebound",
#     "selenium",
#     "spacetrack",
# ]
