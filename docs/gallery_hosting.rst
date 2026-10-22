Hosting and sharing the gallery
===============================

Exploring the physics
---------------------

The gallery's **Explore the physics** link opens ``physics-explorer.html``,
an interactive concept map covering integrators, gravity, shadows, lighting,
thermal emission, attitude, maneuvers and finite burns. Select a topic to see
its connections, model boundaries, Python entry points and runnable examples.
The orbit slider and optional animation illustrate relationships; they do not
run a propagator or produce quantitative predictions. The example cards link
to computed outputs and retain the success, skip or failure status recorded
by the current gallery build. Missing demos show a standalone command and
source link without claiming an available result.

In particular, the thermal topic describes prescribed-temperature emission
and thermal recoil coefficients. It does not claim that SSATK integrates
temperature, heat conduction or a thermal network. Transfer planning is also
distinguished from finite-thrust propagation with mass flow.

The concept map is generated with the gallery report and included in the
portable archive. It has no external script, font or image dependencies of
its own; thumbnails use the gallery's existing outputs. Source links require
Internet access. Animation starts only on request and can be paused; all
topic selection and orbit controls are keyboard accessible.

Building the gallery
--------------------

Build a portable site from a source checkout with the project dependencies
installed. The publishing environment requires a Java 17 JDK, Maven, FFmpeg,
and Chrome in addition to these Python extras:

.. code-block:: bash

   python -m pip install -e ".[dev,all-data,geomagnetics,video,static,browser]"
   plotly_get_chrome -y
   export SSATK_CACHE="/path/to/writable/ssatk-cache"
   export OREKIT_DATA_DIR="$SSATK_CACHE/orekit"
   mvn -B -q -f demos/benchmarks/orekit/pom.xml dependency:build-classpath
   python -m scripts.prepare_gallery_data
   ssatk-demo-gallery --output site --timeout 900 --strict
   python -m ssatk.gallery_validation site
   python -m http.server --directory site 8000

The output directory must be separate from the demo source tree; a build
refuses to clean a source directory or one of its parents. Use a dedicated
output directory: existing contents are removed unless ``--no-clean`` is set.
A rebuild removes the previous adjacent ZIP first and creates a replacement
only if all demos and link checks succeed. With ``--strict``, any skipped demo
also prevents the ZIP and returns a nonzero exit status. The publishing
workflow uses this mode; a local exploratory build may omit it to retain
explicitly labeled skips.

Open ``http://localhost:8000/`` to preview. Upload the entire ``site`` directory,
or extract the adjacent ``site.zip`` onto a static host. The gallery includes
search and category filters, image previews, animation controls, downloadable
artifacts and links to interactive HTML viewers. Images load lazily and videos
start when the visitor requests playback.

Browsing the demos
------------------

The gallery follows a topic sequence rather than alphabetical source-folder
order. Each section has a description and a jump link with its demo count:

#. Getting started
#. Earth, Moon and Solar System views
#. Orbit geometry and propagation
#. Maneuvers and transfers
#. Spacecraft attitude and dynamics
#. Coverage and navigation
#. Illumination and brightness
#. Space environment
#. Dashboards and uncertainty
#. Coordinates and sampling
#. Reference comparisons
#. Animation export

Within a topic, the built-in examples progress from introductory to specialized.
For example, the attitude topic begins with the mesh/vector view before torque,
control, radiation pressure and finite-burn dynamics. Reference comparisons
remain labeled as comparisons, not blanket claims of physical validation.
Empty sections are omitted. Search and the topic selector work together; empty
sections and their jump links disappear, and the visible demo count updates.
Search uses titles, descriptions, demo names and categories, excluding logs.
``Clear filters`` restores all demos.

The presentation catalog is ``ssatk/_gallery_layout.py``. New demos in known
folders inherit the topic and appear after its curated examples, sorted by title
and source name. Custom categories remain separate topics after the built-in
topics; uncategorized examples remain visible in ``Other examples``. The
manifest and page share the same demo and artifact order. Source paths and
manifest ``category`` identifiers are retained, and demo execution order is
unchanged. Authored titles are retained; generated filename titles are made
readable without the redundant ``Demo`` prefix. No demo is duplicated across
topics.

Every demo lists animations and interactive HTML views first, static images
next, other downloads (such as CSV and JSON) after images, and text logs last.
Within each group, paths sort alphabetically without regard to case, with a
stable tie-break for case variants. The manifest uses the same order, including
when the report is rebuilt from existing results. Console output, warnings and
error details follow the file previews.

Data preparation
----------------

If the default user cache is read-only, set ``SSATK_CACHE`` to a writable
directory outside the site before running the demos. The gallery workflow
uses the runner's temporary directory for this cache; cached inputs are not
published unless an exporter explicitly includes them in its output.

``scripts.prepare_gallery_data`` downloads checksummed NASA SVS LOLA elevation
and LROC color products and bakes the Moon viewer's maps. Normals retain the
5760 × 2880 elevation grid; gallery color and horizon maps are 4096 × 2048 and
720 × 360 respectively. The latter is a rendering approximation for terrain
shadows. The texture colors are intended for visualization, not radiometry.
Credit: `NASA's Scientific Visualization Studio <https://svs.gsfc.nasa.gov/4720/>`_.

It also prepares the satellite viewer's **8192 × 4096** Earth day texture from
NASA's **21600 × 10800** Blue Marble Next Generation map. This is 16 times the
pixel count of the packaged 2048 × 1024 texture. The January 2004 composite
includes shaded topography; it does not depict current weather or terrain
geometry. Credit: `NASA Earth Observatory
<https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography/>`_.
The source download is pinned by SHA-256. Both the source and derived JPEG
remain in ``$SSATK_CACHE/earth`` (or ``~/.ssatk/earth``); no large binary is
added to the source repository. To prepare just this texture, run
``python -c "from scripts.prepare_gallery_data import prepare_earth_texture; prepare_earth_texture()"``
from a source checkout.

The viewer embeds the cached day map in its standalone HTML. Without that
cache it keeps the packaged 2K fallback, with no implicit network download.
Night lights, clouds and the ocean specular mask remain at 2048 × 1024.
Earth, clouds and atmosphere use 256 × 128 sphere segments, with up to 8×
anisotropic texture filtering. Three.js reduces oversized textures to the
browser GPU's supported maximum; an 8K texture requires about 171 MiB of GPU
storage including RGBA mipmaps. These are visualization settings, not changes
to Earth's physical radius or orbit calculations.

The same step installs JPL's Linux-format DE430 file and a pinned Orekit
UTC–TAI history covering the benchmark's January 2026 epoch in
``OREKIT_DATA_DIR``. The DE430 bytes are unchanged; its local name is
``lnxp1550p2650.430`` to match Orekit's file-discovery convention.
The dedicated directory must not contain conflicting
ephemerides. It is not published. Orekit 10.3.1 provides the independent
two-body, long-term, and n-body comparisons. No GMAT installation is required
for this gallery: ``demo_gmat_benchmark.py`` is explicitly excluded and remains
available as a standalone benchmark. The multi-tool comparisons run with
Orekit when GMAT is absent.

``--output`` selects both the report directory and the plotting output root.
During each demo, ``SSATK_OUTPUT_DIR`` and the legacy
``SSAPY_DEMO_OUTPUT_DIR`` point there; previous environment values are restored
afterward. Demos must use SSATK's output helpers or the supplied output directory.
Writing to a hardcoded external path does not make that file a published asset.
Files retain their relative layout, including viewer scripts and textures.
Unlike earlier reports, no published links point into the build machine's
``~/ssatk_output`` directory. Names containing spaces or URL punctuation are
percent-encoded in report links.

The gallery workflow runs on pull requests, pushes to ``main`` or ``master``,
and manual dispatch. Pull requests build and validate without deployment. Other
runs publish the complete site using GitHub Pages; repository Pages settings
must use GitHub Actions as the source and permit the deployment environment.
The LLNL project is served at https://software.llnl.gov/ssatk/.
A fork uses its own configured Pages address.

Validation and limits
---------------------

Generation checks local references before writing the ZIP, and CI repeats the
check before uploading the Pages artifact. The validator checks HTML
``href``, ``src`` and ``poster`` attributes, object data, CSS ``url(...)``
references and the manifest. Missing files, paths escaping the site, and
root-relative URLs and symbolic links fail validation. Nested HTML exports are checked as well.
This supports hosting under a project subpath such as ``/ssatk/``.

The validator does not execute JavaScript, check dynamically constructed URLs,
resolve ``srcset`` or CSS imports, or contact external services. External URL
dependencies are listed in its report and can still require Internet access.
It is a packaging check, not validation of rendered pixels or physical results.
The separate workflow suite, ``python -m pytest scripts/tests -q``, also serves an extracted ZIP from a different directory
under ``/ssatk/`` and checks actual HTTP responses against literal asset bytes.

Demos that explicitly report missing optional data or software are labeled
``skipped`` with their reason. Such skips and failures both block publication
under ``--strict``. The HTML report and logs are still written for diagnosis.
Internal exceptions are never
used to retry a demo with different arguments. Demos with a top-level ``def run``
or ``def main`` are imported once and called using recognized parameter names
(``output_dir``/``output_root``, ``make_figures``, ``fast``, ``verbose``).
Other scripts execute once as ``__main__``. Their subprocess deadlines work
across platforms; imported function demos use Unix SIGALRM, including imports,
and must run on the main thread. That deadline is unavailable on Windows.
Output preceding an exception remains in the diagnostic logs. Missing high-resolution Moon
maps or external benchmark tools are not evidence of successful rendering or
physical validation.

The spacecraft mesh view provides PNG/JPG images and GIF/MP4 animations.
Those exports can be hosted like other assets; this change does not connect
the mesh catalog to an interactive browser renderer. Existing Three.js and
Plotly HTML exports remain separate interactive views.
