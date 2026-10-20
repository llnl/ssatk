"""Browse actual procedural viewing meshes, without ephemerides or downloads.

Exports a standalone searchable HTML catalogue and PNG/JPG contact sheets.
Views are independently framed; these schematic models are not flight CAD.
"""

import base64
from html import escape
from io import BytesIO
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ssatk.plots.spacecraft_view import (
    SpacecraftMesh, _camera_basis, _project, _rasterize_mesh,
)

TITLE = "Spacecraft and Telescope Mesh Catalogue"
DESCRIPTION = (
    "Browse every available mesh preset, inspect its sources and approximation notes, "
    "and select a model for the spacecraft viewer. Includes PNG/JPG overview sheets."
)
GALLERY_CATEGORY = "six_dof"

GROUPS = {
    "generic": "Generic spacecraft",
    "observatory": "Space observatories",
    "small": "Small missions",
    "development": "Development missions",
    "concept": "Concepts",
    "legacy": "Heritage astronomy",
    "partner": "International partners",
    "hosted": "Hosted instruments",
}
COLORS = ("#62ccec", "#e9bb75", "#89dcc1", "#c5b0ed", "#c5b0ed",
          "#e9bb75", "#62ccec", "#89dcc1")


def render_thumbnail(mesh, width=480, height=300, *, outline=False):
    """Render the supplied metre geometry with a depth buffer and isotropic fit.

    A common oblique camera reveals +Z and -Y telescope apertures. Each mesh
    gets its own framing, without changing its vertices or physical dimensions.
    Supersampling retains narrow struts. Optional wireframe outlines show actual
    polygon edges at a one-pixel minimum width for subpixel radio antennas.
    """
    center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
    radius = np.linalg.norm(mesh.vertices - center, axis=1).max()
    direction = np.array([1.1, -1.8, 1.4])
    direction /= np.linalg.norm(direction)
    basis = _camera_basis(-direction, [0, 0, 1])
    uv, depth = _project(mesh.vertices - center - 5 * radius * direction, basis, 1, 1)
    uv -= (uv.min(axis=0) + uv.max(axis=0)) / 2
    uv[:, 0] *= height / width
    uv *= 0.9 / np.abs(uv).max()
    points = mesh.vertices[mesh.triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    light = direction + [-0.25, -0.15, 0.5]
    light /= np.linalg.norm(light)
    colors = mesh.triangle_colors * (0.3 + 0.7 * np.maximum(normals @ light, 0))[:, None]
    rgba, _ = _rasterize_mesh(uv, depth, mesh.triangles, colors,
                              width * 2, height * 2, mesh.triangle_boundaries)
    image = Image.fromarray(np.uint8(np.clip(rgba, 0, 1) * 255)).resize(
        (width, height), Image.Resampling.LANCZOS)
    if outline:
        pixels = (uv * [1, -1] + 1) * [width / 2, height / 2]
        draw = ImageDraw.Draw(image)
        for face in mesh.faces:
            points = [tuple(point) for point in pixels[face]]
            draw.line(points + points[:1], fill=(180, 191, 203, 220), width=1)
    return image


def catalog_entries(*, width=480, height=300):
    """Read every canonical preset once; aliases remain searchable metadata."""
    entries = []
    for name in SpacecraftMesh.presets():
        info = SpacecraftMesh.preset_info(name)
        outline = name in {"rae1", "rae2"}
        if outline:
            info["notes"] += " Display: thin antenna boundaries use one-pixel wireframe outlines for visibility."
        entries.append((info, render_thumbnail(SpacecraftMesh.from_preset(name), width, height, outline=outline)))
    return entries


def _groups(entries):
    available = {info["group"] for info, _ in entries}
    return [key for key in GROUPS if key in available] + sorted(available - GROUPS.keys())


def _image_source(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_catalog(entries, path):
    """Write offline HTML with embedded previews and no external runtime assets."""
    groups = _groups(entries)
    options = ''.join(f'<option value="{escape(g)}">{escape(GROUPS.get(g, g))}</option>' for g in groups)
    sections = []
    for group in groups:
        cards = []
        for info, thumbnail in entries:
            if info["group"] != group:
                continue
            name = escape(info["name"])
            label = escape(info["label"])
            search = escape(' '.join((info["name"], info["label"], *info["aliases"],
                                     GROUPS.get(group, group))).casefold())
            sources = ''.join(f'<li><a href="{escape(url)}" target="_blank" rel="noopener noreferrer">'
                              f'Reference {i + 1}</a></li>' for i, url in enumerate(info["sources"]))
            features = ''.join(f'<li>{escape(feature)}</li>' for feature in info["reference_features"])
            aliases = escape(', '.join(info["aliases"]) or 'None')
            cards.append(f'''<article class="mesh-card" data-name="{name}" data-search="{search}">
  <img src="{_image_source(thumbnail)}" width="{thumbnail.width}" height="{thumbnail.height}"
       alt="Schematic viewing mesh: {label}">
  <div class="card-body"><h3>{label}</h3><code class="preset-id">{name}</code>
  <button class="choose" type="button" data-preset="{name}" aria-pressed="false">Use this mesh</button>
  <pre class="card-code" hidden tabindex="0" aria-label="Preset selection example"></pre>
  <details><summary>Geometry, axes &amp; sources</summary>
    <p class="fidelity">{escape(info['fidelity'])}</p><p>{escape(info['notes'])}</p>
    <p><strong>Visual axes:</strong> {escape(info['axes'])}</p>
    <p><strong>Origin:</strong> {escape(info['origin'])}</p>
    <p><strong>Aliases:</strong> {aliases}</p>
    {'<p>Referenced features:</p><ul>' + features + '</ul>' if features else ''}
    {'<ul class="sources">' + sources + '</ul>' if sources else ''}
  </details></div></article>''')
        color = COLORS[list(GROUPS).index(group)] if group in GROUPS else COLORS[0]
        sections.append(f'''<section class="mesh-group" data-group="{escape(group)}" style="--accent:{color}">
  <h2>{escape(GROUPS.get(group, group))} <span>{len(cards)}</span></h2>
  <div class="mesh-grid">{''.join(cards)}</div></section>''')
    html = _HTML.replace("__OPTIONS__", options).replace("__SECTIONS__", ''.join(sections))
    html = html.replace("__TOTAL__", str(len(entries))).replace("__GROUP_COUNT__", str(len(groups)))
    Path(path).write_text(html, encoding="utf-8")
    return Path(path)


def write_poster(entries, path, *, cell_width=360):
    """A three-column overview of actual meshes; no network or system fonts needed."""
    from matplotlib.font_manager import FontProperties, findfont

    unit = cell_width / 360
    px = lambda value: round(value * unit)
    font_path = findfont(FontProperties(family="DejaVu Sans"))
    bold_path = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    font = lambda size, bold=False: ImageFont.truetype(bold_path if bold else font_path, px(size))
    grouped = {key: [(info, image) for info, image in entries if info["group"] == key]
               for key in _groups(entries)}
    # Preserve readable thematic columns. Unknown future groups are also included.
    columns = [["observatory", "small", "development", "concept"],
               ["generic", "legacy"], ["partner", "hosted"]]
    columns[-1] += [key for key in grouped if key not in GROUPS]
    columns = [[key for key in column if key in grouped] for column in columns]
    gap, margin, header, tile_h = px(16), px(100), px(330), px(295)
    col_width = 4 * cell_width + 3 * gap
    section_h = lambda key: px(110) + math.ceil(len(grouped[key]) / 4) * tile_h
    height = header + max(sum(section_h(key) for key in col) for col in columns) + px(190)
    canvas = Image.new("RGB", (2 * margin + 3 * col_width + 2 * px(60), height), "#0a1324")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, px(50)), "SSATK / SPACECRAFT MESH LIBRARY", font=font(24, True), fill="#7fcee7")
    draw.text((margin, px(100)), "Spacecraft & space telescopes", font=font(84, True), fill="#edf3fa")
    draw.text((margin, px(222)), f"{len(entries)} selectable presets  /  actual procedural meshes  /  independently framed",
              font=font(30), fill="#bac7d8")
    for column_index, column in enumerate(columns):
        x = margin + column_index * (col_width + px(60))
        y = header
        for group in column:
            color = COLORS[list(GROUPS).index(group)] if group in GROUPS else COLORS[0]
            draw.line((x, y, x + col_width, y), fill=color, width=px(3))
            draw.text((x, y + px(18)), f"{GROUPS.get(group, group)} / {len(grouped[group])}",
                      font=font(31, True), fill=color)
            for index, (info, image) in enumerate(grouped[group]):
                left = x + (index % 4) * (cell_width + gap)
                top = y + px(80) + (index // 4) * tile_h
                draw.rounded_rectangle((left, top, left + cell_width, top + tile_h - gap),
                                       radius=px(12), fill="#111f35")
                image = image.copy()
                image.thumbnail((cell_width - px(18), px(215)), Image.Resampling.LANCZOS)
                canvas.paste(image, (left + (cell_width - image.width) // 2,
                                    top + (px(215) - image.height) // 2), image)
                # Canonical ID is unambiguous and fits even for long mission names.
                draw.text((left + cell_width // 2, top + px(232)), info["name"],
                          font=font(22, True), fill="#edf3fa", anchor="mt")
            y += section_h(group)
    draw.text((margin, height - px(125)), "Select by name:  SpacecraftMesh.from_preset(\"jwst\")  or  spacecraft_view(r, t, q, mesh=\"jwst\")",
              font=font(28), fill="#edf3fa")
    draw.text((margin, height - px(65)), "Schematic models; different scales per tile. Thin radio antennas outlined for visibility. Sources, axes and limitations in the HTML catalogue.",
              font=font(26), fill="#bac7d8")
    path = Path(path)
    canvas.save(path)
    canvas.save(path.with_suffix(".jpg"), quality=92)
    return path, path.with_suffix(".jpg")


_HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spacecraft &amp; Telescope Mesh Catalogue | SSATK</title>
<link rel="icon" href="data:,">
<style>
:root{color-scheme:dark;font:16px/1.6 system-ui,sans-serif;color:#edf3fa;background:#091222}
*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:auto;padding:44px 40px 70px}
.eyebrow{font-size:12px;letter-spacing:.18em;color:#7fcee7;font-weight:700}
h1{font-size:clamp(32px,4.4vw,64px);line-height:1.08;letter-spacing:-.035em;margin:18px 0}
.intro{max-width:900px;color:#bac7d8;font-size:18px}.stats{display:flex;gap:30px;color:#7fcee7}
.downloads a{margin-right:22px}a{color:#87d5ed}a:hover{color:white}button,select,input{font:inherit}
button{cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #7fcee7;outline-offset:3px}
.selection{display:grid;grid-template-columns:1fr 2fr;gap:24px;background:#122138;border:1px solid #28425e;
 border-radius:16px;padding:24px;margin:28px 0}.selection h2{margin:0;font-size:20px}
.selection p{color:#bac7d8;font-size:14px;margin-bottom:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;
 background:#0a1526;padding:18px;border-radius:10px;margin:0;font-size:13px;line-height:1.9}
.filters{position:sticky;top:0;z-index:3;background:#091222f5;padding:16px 0;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
label{font-size:13px;color:#bac7d8;display:flex;align-items:center;gap:10px}
input,select{background:#14233a;color:#edf3fa;border:1px solid #395069;border-radius:8px;padding:10px 14px}
input{width:min(340px,60vw)}#result-count{font-size:13px;color:#bac7d8;margin-left:auto}
.mesh-group{margin:26px 0 44px}.mesh-group h2{font-size:22px;border-top:2px solid var(--accent);padding-top:16px}
.mesh-group h2 span{color:var(--accent);font-size:14px;margin-left:10px}
.mesh-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:20px;align-items:start}
.mesh-card{border:1px solid #263a54;background:#101e33;border-radius:14px;overflow:hidden}
.mesh-card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.mesh-card img{width:100%;height:auto;display:block;background:radial-gradient(ellipse at center,#20334d,#101e33 72%)}
.card-body{padding:0 20px 20px}.card-code{font-size:12px;margin-bottom:16px;padding:12px}
.card-body h3{font-size:16px;line-height:1.35;margin:4px 0 8px;min-height:44px}
.preset-id{font-size:13px;color:var(--accent);display:block}.choose{margin:16px 0;width:100%;border:1px solid #3b5877;
 background:#203550;border-radius:7px;color:#edf3fa;padding:8px}.choose:hover{background:#2d4a68}
details{font-size:13px;color:#b8c8dc}summary{cursor:pointer}.fidelity{color:var(--accent)}
ul{padding-left:20px}.sources{display:flex;gap:20px;list-style:none;padding:0;flex-wrap:wrap}
.notes{color:#bac7d8;font-size:14px;border-top:1px solid #2a3b50;padding-top:24px}
[hidden]{display:none!important}#empty{padding:40px;text-align:center;color:#bac7d8}
@media(max-width:650px){main{padding:24px 18px}.selection{grid-template-columns:1fr}.filters{position:static}
 label{width:100%;justify-content:space-between}input,select{width:70%}#result-count{margin-left:0}.stats{gap:15px;font-size:14px}}
</style></head><body><main>
<header><div class="eyebrow">SSATK / VISUAL MODEL LIBRARY</div>
<h1>Spacecraft &amp;<br>space telescopes.</h1>
<p class="intro">Explore the meshes available in the spacecraft viewer. Pick a model, find its preset name,
 and inspect the geometry notes before using it with your trajectory and attitude history.</p>
<div class="stats"><strong>__TOTAL__ presets</strong><span>__GROUP_COUNT__ collections</span><span>No mesh downloads needed</span></div>
<p class="downloads"><a href="spacecraft_mesh_catalog.png" download>PNG overview</a><a href="spacecraft_mesh_catalog.jpg" download>JPG overview</a></p>
</header>
<section class="selection" aria-label="Selected preset">
<div><h2 id="selected-name">Starlink-like (default)</h2><p id="selected-status" role="status">Choose any mesh below to update this example.</p>
<p>Supply your own <code>r, t, q</code> history. Positions are GCRF metres, times are absolute GPS seconds or Astropy Time,
 and quaternions are body-to-GCRF [w, x, y, z]. Align the preset's visual axes with your attitude convention.</p></div>
<pre id="selection-code">from ssatk.plots.spacecraft_view import SpacecraftMesh, spacecraft_view

mesh = SpacecraftMesh.from_preset("starlink")
view = spacecraft_view(r, t, q, mesh=mesh)
view.save("starlink.png")</pre></section>
<div class="filters"><label for="search">Find a mesh<input id="search" type="search" placeholder="Webb, CubeSat, HST…"></label>
<label for="group">Collection<select id="group"><option value="">All collections</option>__OPTIONS__</select></label>
<span id="result-count" role="status">__TOTAL__ of __TOTAL__ meshes</span></div>
<p id="empty" hidden>No meshes match. Try another name or collection.</p>
__SECTIONS__
<footer class="notes"><strong>Viewing templates, with explicit limits.</strong> Every preview is rendered from the actual SSATK preset geometry.
 Each tile is framed independently; these are not common-scale size comparisons. Most surfaces and proportions are schematic;
 only the listed referenced features have published dimensions or counts. Family models are stand-ins, and hosted instruments omit their carrier.
 Concepts and development entries are labeled by collection. Sources establish identity or selected features, not flight-CAD accuracy.
 Thin radio antenna boundaries use one-pixel wireframe outlines for visibility; metre geometry is unchanged.
 These meshes do not provide mass, inertia, mission attitude frames or calibrated optical properties. Custom geometry is supported through
 <code>SpacecraftMesh.from_obj("model.obj", scale=1.0)</code>, where scale is metres per OBJ unit.</footer>
<noscript><p>All meshes and sources are visible without JavaScript. Enable JavaScript to search or change the selection example.</p></noscript>
</main><script>
const cards = [...document.querySelectorAll('.mesh-card')];
const groups = [...document.querySelectorAll('.mesh-group')];
const search = document.getElementById('search'), group = document.getElementById('group');
function filter() {
  const words = search.value.trim().toLowerCase().split(/\\s+/).filter(Boolean);
  let count = 0;
  for (const section of groups) {
    let visible = 0;
    for (const card of section.querySelectorAll('.mesh-card')) {
      card.hidden = !!(group.value && group.value !== section.dataset.group) ||
        !words.every(word => card.dataset.search.includes(word));
      if (!card.hidden) visible++;
    }
    section.hidden = visible === 0;
    section.querySelector('h2 span').textContent = visible;
    count += visible;
  }
  document.getElementById('result-count').textContent = `${count} of ${cards.length} meshes`;
  document.getElementById('empty').hidden = count !== 0;
}
search.addEventListener('input', filter); group.addEventListener('change', filter);
for (const button of document.querySelectorAll('.choose')) button.addEventListener('click', () => {
  const name = button.dataset.preset, card = button.closest('.mesh-card');
  for (const item of cards) {
    const selected = item === card;
    item.classList.toggle('selected', selected);
    item.querySelector('.choose').setAttribute('aria-pressed', String(selected));
    item.querySelector('.choose').textContent = selected ? 'Selected' : 'Use this mesh';
    item.querySelector('.card-code').hidden = !selected;
  }
  document.getElementById('selected-name').textContent = card.querySelector('h3').textContent;
  document.getElementById('selected-status').textContent = `Selected preset: ${name}`;
  document.getElementById('selection-code').textContent =
    `from ssatk.plots.spacecraft_view import SpacecraftMesh, spacecraft_view\n\n` +
    `mesh = SpacecraftMesh.from_preset("${name}")\n` +
    `view = spacecraft_view(r, t, q, mesh=mesh)\nview.save("${name}.png")`;
  card.querySelector('.card-code').textContent = document.getElementById('selection-code').textContent;
});
</script></body></html>'''


def main(make_figures=True, fast=False, output_dir=None):
    """Export the complete inventory; fast mode changes image size, never coverage."""
    if not make_figures:
        return {"title": TITLE, "presets": list(SpacecraftMesh.presets()), "files": []}
    if output_dir is None:
        from ssatk.plots.figpath import figpath
        output_dir = Path(figpath("six_dof/spacecraft_mesh_catalog.html")).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = catalog_entries(width=480 if fast else 720, height=300 if fast else 450)
    # gallery-call: *.html
    html = write_catalog(entries, output_dir / "spacecraft_mesh_catalog.html")
    # gallery-end
    # gallery-call: *.jpg | *.png
    images = write_poster(entries, output_dir / "spacecraft_mesh_catalog.png",
                          cell_width=360 if fast else 480)
    # gallery-end
    return {"title": TITLE, "files": [str(html), *(str(path) for path in images)]}


if __name__ == "__main__":
    main()
