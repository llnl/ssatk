Hosting and sharing the gallery
===============================

Build a portable site from a source checkout with the project dependencies
installed, including the ``dev`` and ``all-data`` extras:

.. code-block:: bash

   ssatk-demo-gallery --output site
   python -m ssatk.gallery_validation site
   python -m http.server --directory site 8000

Open ``http://localhost:8000/`` to preview. Upload the entire ``site`` directory,
or extract the adjacent ``site.zip`` onto a static host. The gallery includes
search and category filters, image previews, animation controls, downloadable
artifacts and links to interactive HTML viewers. Images load lazily and videos
start when the visitor requests playback.

If the default user cache is read-only, set ``SSATK_CACHE`` to a writable
directory outside the site before running the demos. The gallery workflow
uses the runner's temporary directory for this cache; cached inputs are not
published unless an exporter explicitly includes them in its output.

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
root-relative URLs fail validation. Nested HTML exports are checked as well.
This supports hosting under a project subpath such as ``/ssatk/``.

The validator does not execute JavaScript, check dynamically constructed URLs,
resolve ``srcset`` or CSS imports, or contact external services. External URL
dependencies are listed in its report and can still require Internet access.
It is a packaging check, not validation of rendered pixels or physical results.
The regression suite also serves an extracted ZIP from a different directory
under ``/ssatk/`` and checks actual HTTP responses against literal asset bytes.

Demos that explicitly report missing optional data or software are labeled
``skipped`` with their reason. Failed demos remain failures and make the CLI
return a nonzero status, preventing publication. Internal exceptions are never
used to retry a demo with different arguments. Missing high-resolution Moon
maps or external benchmark tools are not evidence of successful rendering or
physical validation.

The spacecraft mesh view provides PNG/JPG images and GIF/MP4 animations.
Those exports can be hosted like other assets; this change does not connect
the mesh catalog to an interactive browser renderer. Existing Three.js and
Plotly HTML exports remain separate interactive views.
