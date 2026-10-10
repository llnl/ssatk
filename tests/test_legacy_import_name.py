"""``ssapy_toolkit`` (pre-1.1.0 import name) must keep resolving to ``ssatk``.

Contract test for the rename: every object reached through the old path must
be the identical object reached through the new one, so results cannot differ
between the two names, and pickles written by 1.0.x (which record
``ssapy_toolkit.*`` paths) still load.
"""

import pickle
import warnings

import numpy as np
import pytest

pytest.importorskip("ssapy_toolkit", reason="install packaging/ssapy-toolkit-compat to test the alias")


def _import_legacy():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ssapy_toolkit
        import ssapy_toolkit.coordinates.frames as frames_old
        import ssapy_toolkit.plots.globe_plot  # noqa: F401  (must not clobber the re-export)
    return ssapy_toolkit, frames_old, caught


def test_legacy_name_returns_the_same_objects():
    import ssatk
    import ssatk.coordinates.frames as frames_new
    import ssatk.plots

    legacy, frames_old, _caught = _import_legacy()
    assert legacy.__version__ == ssatk.__version__
    assert frames_old.eci_to_ecf_matrix is frames_new.eci_to_ecf_matrix
    # Same object, so the same GCRF->ITRF matrix to the last bit.
    np.testing.assert_array_equal(frames_old.eci_to_ecf_matrix(1.4e9), frames_new.eci_to_ecf_matrix(1.4e9))
    # Importing the submodule through the old name must leave the package
    # re-export (the function) in place on the real package.
    assert callable(ssatk.plots.globe_plot) and not hasattr(ssatk.plots.globe_plot, "__path__")
    assert ssatk.plots.globe_plot.__module__ == "ssatk.plots.globe_plot"


def test_legacy_name_emits_a_future_warning():
    # A fresh interpreter: the module above was already imported at collection.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", "-c", "import ssapy_toolkit"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "renamed to 'ssatk'" in result.stderr


def test_pickles_that_name_the_old_module_path_still_load():
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    _import_legacy()
    # Protocol 0 stores a global as "c<module>\n<name>\n" with no length
    # prefix, so swapping the module path gives exactly what 1.0.x wrote.
    payload = pickle.dumps(eci_to_ecf_matrix, protocol=0)
    assert payload.startswith(b"cssatk.coordinates.frames\n")
    legacy_payload = payload.replace(b"cssatk.", b"cssapy_toolkit.", 1)
    assert pickle.loads(legacy_payload) is eci_to_ecf_matrix
