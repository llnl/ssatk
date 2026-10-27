"""C: portable, output-specific source examples and canonical demo paths."""

import json
from pathlib import Path
import shutil
import textwrap

import pytest

from ssatk._gallery_calls import collect_figure_calls, source_examples
from ssatk.demo_gallery import DemoResult, build_html_report, discover_demo_files, run_demo_script
from ssatk.gallery_validation import validate_gallery


def test_figure_call_matches_source_and_survives_manifest_relocation(tmp_path):
    """C: use an executed drawing call, quoted filename, and HTML-sensitive arguments."""
    demo = tmp_path / "demo_shapes.py"
    source = '''from pathlib import Path
from ssatk.plots.figpath import figpath
def draw(name, label):
    Path(figpath(name)).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
def main():
    # gallery-call: Earth and Moon.svg
    draw("Earth and Moon.svg", label="<Earth & Moon>")
    # gallery-end
    print("finished")
'''
    demo.write_text(source)
    site = tmp_path / "site"
    result = run_demo_script(demo, site)
    assert result.status == "success", result.error
    filename = "figures/Earth%20and%20Moon.svg"
    assert set(result.figure_calls) == {filename}  # logs receive no figure call
    example = result.figure_calls[filename]
    assert example["function"] == "draw"
    assert example["code"] == 'draw("Earth and Moon.svg", label="<Earth & Moon>")'
    assert example["code"] == textwrap.dedent("\n".join(
        source.splitlines()[example["first_line"]-1:example["last_line"]]))
    build_html_report([result], site)
    relocated = tmp_path / "published/ssatk"
    shutil.copytree(site, relocated)
    rows = json.loads((relocated / "manifest.json").read_text())
    build_html_report([DemoResult(**row) for row in rows], relocated)
    page = (relocated / "index.html").read_text()
    assert 'draw(...)</code>' in page
    assert '&lt;Earth &amp; Moon&gt;' in page and '<Earth & Moon>' not in page
    assert (relocated / example["source"]).read_text() == source
    validate_gallery(relocated)


@pytest.mark.parametrize("source", [
    "# gallery-call: x.png\nf()\n",
    "# gallery-call: x.png\n# gallery-call: y.png\nf()\n# gallery-end\n",
    "# gallery-end\n",
])
def test_incomplete_or_nested_examples_cannot_publish_wrong_source(source):
    """C: malformed boundaries must fail instead of silently selecting other calls."""
    with pytest.raises(ValueError):
        source_examples(source)


def test_overlapping_filename_patterns_are_rejected(tmp_path):
    """C: an image must not silently acquire the example for another figure."""
    demo = tmp_path / "demo_shapes.py"
    demo.write_text("# gallery-call: *.png\na()\n# gallery-end\n"
                    "# gallery-call: a.png\nb()\n# gallery-end\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        collect_figure_calls(demo, tmp_path / "site", "demo_shapes", ["figures/a.png"])


def test_first_workflow_writes_one_figure_directory(tmp_path):
    """C: run the real introductory demo through the same publisher path."""
    demos = Path(__file__).resolve().parents[2] / "demos"
    result = run_demo_script(demos / "getting_started/demo_first_user_workflow.py",
                             tmp_path / "site", demos_dir=demos, fast=True)
    assert result.status == "success", result.error
    assert "figures/first_user_workflow.jpg" in result.files
    assert not (tmp_path / "site/figures/figures").exists()
    assert result.figure_calls["figures/first_user_workflow.jpg"]["function"] == "orbit_plot"


def test_bundled_source_examples_are_parseable_and_quickstart_calls_are_distinct():
    """C: annotations stay with real source; static and animated calls are distinct."""
    demos = Path(__file__).resolve().parents[2] / "demos"
    for path in discover_demo_files(demos):
        examples = source_examples(path.read_text())
        # The data-access demo produces console output only.
        if path.name != "demo_data_package_access.py":
            assert examples, path
    examples = source_examples((demos / "getting_started/demo_plotting_quickstart.py").read_text())
    by_output = {pattern: example for example in examples for pattern in example["patterns"]}
    assert by_output["plotting_quickstart_orbit_animation.gif"]["function"] == "orbit_plot"
    assert by_output["plotting_quickstart_orbit_views.jpg"]["function"] == "orbit_plot"
    assert by_output["plotting_quickstart_orbit_animation.gif"]["code"] != by_output["plotting_quickstart_orbit_views.jpg"]["code"]
    examples = source_examples((demos / "space_environment/demo_van_allen.py").read_text())
    assert {example["function"] for example in examples} == {"plot_van_allen_3d"}
    examples = source_examples((demos / "orbital_mechanics/demo_ellipse_fit_against_ssapy.py").read_text())
    assert examples[0]["function"] == "orbit_plot"
    assert examples[1]["function"] == "groundtrack_dashboard"
