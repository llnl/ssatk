"""Portable, dependency-free concept map of the gallery's physics workflows."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from ssatk._physics_catalog import TOPICS as RADIATION_TOPICS
from ssatk._physics_dynamics_catalog import TOPICS as DYNAMICS_TOPICS


def demo_anchor(name: str) -> str:
    return "demo-" + name


def write_physics_explorer(results, out_root: Path) -> None:
    """Connect a schematic to the outputs actually present in this build.

    No propagation runs in the browser. Results retain their recorded status;
    an absent or failed demo is never represented as a successful simulation.
    """
    by_name = {result.name: result for result in results}
    topics = {topic["id"]: topic for topic in DYNAMICS_TOPICS + RADIATION_TOPICS}
    ordered = []
    for key in ("integrators", "gravity", "shadows", "lighting", "thermal", "attitude", "maneuvers", "burns"):
        topic = dict(topics[key])
        demos = []
        for name in topic["demos"]:
            result = by_name.get(name)
            images = [] if result is None else [
                file for file in result.files
                if Path(file).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            demos.append({
                "name": name,
                "title": result.title if result else name.rsplit("/", 1)[-1].removeprefix("demo_").replace("_", " ").capitalize(),
                "status": result.status if result else "not in this build",
                "href": "index.html#" + quote(demo_anchor(name), safe="") if result else None,
                "preview": images[0] if images else None,
                "command": "python -m demos." + name.replace("/", "."),
                "source": "https://github.com/llnl/ssatk/blob/main/demos/" + quote(name, safe="/") + ".py",
            })
        topic["demos"] = sorted(demos, key=lambda demo: (demo["href"] is None, demo["status"] != "success"))
        ordered.append(topic)
    # JSON is placed in a script element; escape HTML delimiters even for
    # externally supplied manifests, not just the package's own demo names.
    payload = json.dumps(ordered, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    fallback = "".join(
        f'<section><h2>{html.escape(topic["title"])}</h2><p>{html.escape(topic["summary"])}</p>'
        f'<p>{html.escape(topic["limits"])}</p></section>' for topic in ordered
    )
    page = _PAGE.replace("__TOPIC_DATA__", payload).replace("__NO_SCRIPT__", fallback)
    (out_root / "physics-explorer.html").write_text(page, encoding="utf-8")


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Physics explorer · SSATK</title>
<style>
:root { color-scheme:dark; --bg:#080f1c; --panel:#101e30; --ink:#edf4fc; --muted:#a9bed2; --line:#294057; --accent:#6de3d1; }
* { box-sizing:border-box; }
body { margin:0; background:radial-gradient(ellipse at 20% 0%,#163047 0,transparent 58%),var(--bg); color:var(--ink); font:16px/1.55 system-ui,sans-serif; }
main { max-width:1380px; margin:auto; padding:32px 40px 64px; }
a { color:#91d7ff; text-underline-offset:4px; }
button,input { font:inherit; }
button { cursor:pointer; }
button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible { outline:3px solid #ffdc85; outline-offset:4px; }
.masthead { display:flex; justify-content:space-between; gap:20px; align-items:center; }
.brand { font-weight:750; letter-spacing:.14em; font-size:13px; }
.eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-size:12px; font-weight:750; }
h1 { font-size:clamp(38px,5vw,64px); line-height:1.05; letter-spacing:-.045em; margin:10px 0 20px; }
.intro { max-width:780px; color:var(--muted); font-size:18px; }
.intro-block { padding:44px 0 22px; }
.topics { display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:8px; margin:12px 0 24px; }
.topic { background:#101d2c; color:var(--muted); border:1px solid var(--line); border-radius:12px; padding:14px 9px; text-align:left; }
.topic span { display:block; font-size:11px; margin-bottom:12px; opacity:.8; }
.topic strong { font-size:14px; }
.topic:hover { background:#1b3447; }
.topic[aria-pressed="true"] { border-color:var(--accent); background:#173d43; color:#cafff1; box-shadow:inset 0 3px var(--accent); }
.workspace { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr); border:1px solid var(--line); border-radius:20px; overflow:hidden; background:var(--panel); }
.scene { background:radial-gradient(ellipse at 50% 50%,#173047,#09131f 72%); padding:24px; display:flex; flex-direction:column; justify-content:space-between; min-width:0; }
.scene-label { display:flex; justify-content:space-between; gap:14px; color:var(--muted); font-size:12px; }
.scene-label strong { color:var(--ink); }
svg { width:100%; height:auto; display:block; overflow:visible; }
svg text { font:13px system-ui,sans-serif; fill:#bbd1e4; }
.layer { display:none; }
.layer.active { display:block; }
.controls { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
.controls button { border:1px solid #507189; border-radius:8px; background:#163448; color:#e5f7fc; padding:7px 14px; min-width:92px; }
.controls label { flex:1; min-width:150px; color:var(--muted); font-size:13px; }
.controls input { width:100%; accent-color:var(--accent); display:block; }
.scene-note { font-size:12px; color:#a6bacc; margin:14px 0 0; }
.detail { padding:30px; border-left:1px solid var(--line); }
h2 { font-size:29px; line-height:1.2; margin:8px 0 18px; letter-spacing:-.02em; }
.detail p { color:#c8d7e5; }
.chain { display:flex; flex-wrap:wrap; gap:8px; padding:0; list-style:none; margin:22px 0; }
.chain li { font-size:12px; border:1px solid #3d6172; padding:7px 10px; border-radius:6px; color:#b6f1e2; }
.chain li:not(:last-child)::after { content:' →'; color:#79aab9; }
.boundary { border-left:2px solid #e8bb72; padding-left:14px; font-size:13px; }
.boundary strong { color:#f6d09b; display:block; margin-bottom:5px; }
details { margin-top:24px; border-top:1px solid var(--line); padding-top:14px; }
summary { cursor:pointer; color:var(--muted); font-size:13px; }
.apis { list-style:none; padding:0; }
.apis li { padding:5px 0; }
code { font:12px/1.6 ui-monospace,monospace; color:#a7e5f0; overflow-wrap:anywhere; }
.examples { margin-top:34px; }
.examples-heading { display:flex; justify-content:space-between; align-items:baseline; gap:20px; }
.examples-heading h2 { font-size:22px; }
.examples-heading p { font-size:13px; color:var(--muted); }
.demo-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }
.demo { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#101c2b; }
.demo img { width:100%; height:180px; object-fit:contain; display:block; background:#060d17; }
.demo-body { padding:18px; }
.demo h3 { font-size:16px; margin:10px 0; line-height:1.35; }
.status { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:#b5c7d8; }
.status.success { color:#7ae5bf; }
.status.failed { color:#ffb6a8; }
.demo a { font-size:13px; }
.demo code { display:block; margin:12px 0; }
.footer { color:var(--muted); font-size:12px; margin-top:32px; max-width:900px; }
[hidden] { display:none !important; }
@media(max-width:1050px) { .topics { grid-template-columns:repeat(4,minmax(0,1fr)); } .workspace { grid-template-columns:1fr; } .detail { border-left:0; border-top:1px solid var(--line); } .scene svg { max-height:430px; } }
@media(max-width:600px) { main { padding:20px 16px 40px; } .intro-block { padding-top:30px; } .topics { grid-template-columns:repeat(2,minmax(0,1fr)); } .topic { padding:10px; } .topic span { display:inline; margin-right:8px; } .detail,.scene { padding:20px; } .scene-label { flex-direction:column; gap:2px; } .examples-heading { display:block; } .demo-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<nav class="masthead" aria-label="Gallery"><span class="brand">SSATK / EXPLORE</span><a href="index.html">← Demo gallery</a></nav>
<header class="intro-block"><div class="eyebrow">From an orbit to an observable</div><h1>Physics in motion.</h1><p class="intro">Follow the connections between motion, sunlight, spacecraft geometry and propulsion. Choose a layer to see what it does, where its assumptions stop, and which examples to run.</p></header>
<div class="topics" role="group" aria-label="Physics topics" id="topics"></div>
<div class="workspace">
<section class="scene" aria-label="Interactive schematic">
<div class="scene-label"><strong id="scene-title">Mission concept map</strong><span>SCHEMATIC · NOT A PROPAGATION</span></div>
<svg viewBox="0 0 680 440" role="img" aria-labelledby="diagram-title diagram-desc">
<title id="diagram-title">Connections in a spacecraft model</title><desc id="diagram-desc">A central body, orbit, spacecraft and sunlight. Selected physics layers illustrate inputs and effects; trajectories, distances, arrows and sampling are schematic.</desc>
<defs>
 <radialGradient id="body" cx="25%" cy="30%"><stop stop-color="#426781"/><stop offset="1" stop-color="#152a3d"/></radialGradient>
 <linearGradient id="shadow"><stop stop-color="#02060c" stop-opacity=".9"/><stop offset="1" stop-color="#02060c" stop-opacity=".2"/></linearGradient>
 <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6de3d1"/></marker>
 <marker id="gold-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd481"/></marker>
</defs>
<g opacity=".15" stroke="#78a1bd" fill="none"><path d="M50 110H640M50 220H640M50 330H640M130 45V395M340 45V395M550 45V395"/></g>
<ellipse cx="340" cy="220" rx="222" ry="137" fill="none" stroke="#4f6e84" stroke-dasharray="3 7"/>
<g id="sunlight" stroke="#e7bc70" opacity=".55"><path d="M18 180H135M18 220H135M18 260H135" marker-end="url(#gold-arrow)"/></g>
<text x="20" y="155">Sunlight</text>
<g class="layer" data-layer="shadows"><path d="M340 147L655 111V329L340 293Z" fill="#a9a1a0" opacity=".13"/><path d="M340 152L655 180V260L340 288Z" fill="url(#shadow)"/><text x="560" y="214">Umbra</text><text x="552" y="140">Penumbra</text></g>
<circle cx="340" cy="220" r="68" fill="url(#body)" stroke="#607f94"/>
<ellipse cx="340" cy="220" rx="28" ry="68" fill="none" stroke="#65829a" opacity=".4"/>
<ellipse cx="340" cy="220" rx="68" ry="22" fill="none" stroke="#65829a" opacity=".4"/>
<text x="340" y="314" text-anchor="middle">Central body</text>
<g class="layer" data-layer="integrators" id="samples" fill="#6de3d1"></g>
<g class="layer" data-layer="integrators"><text x="165" y="390">Force evaluations → sampled state history</text></g>
<g class="layer" data-layer="gravity" fill="none" stroke="#6de3d1" stroke-width="2"><path d="M152 220H250M528 220H430M340 94V132M340 346V310" marker-end="url(#arrow)"/><ellipse cx="340" cy="220" rx="239" ry="154" transform="rotate(-12 340 220)" stroke-dasharray="5 6" opacity=".65"/></g>
<g class="layer" data-layer="gravity"><text x="192" y="390">Central gravity + selected perturbations</text></g>
<g class="layer" data-layer="maneuvers" fill="none" stroke="#6de3d1" stroke-width="2"><path d="M118 220C118 17 620 17 620 220" stroke-dasharray="8 5"/><path d="M118 220V161M620 220V279" marker-end="url(#arrow)"/></g>
<g class="layer" data-layer="maneuvers"><text x="85" y="250">Δv₁</text><text x="621" y="302">Δv₂</text><text x="199" y="390">Plan a transfer → apply maneuver events</text></g>
<g class="layer" data-layer="lighting"><path id="sightline" d="" stroke="#ffd481" stroke-dasharray="5 5" fill="none"/><path d="M575 345l22-12 16 24-22 12z" fill="#27495c" stroke="#a9d4eb"/><text x="557" y="398">Observer</text></g>
<g id="craft" transform="translate(491 120)">
 <g class="layer" data-layer="burns"><path d="M-16 0L-75 -15L-56 0L-75 15Z" fill="#ffb962" opacity=".85"/><path d="M27 0H97" stroke="#6de3d1" stroke-width="2" marker-end="url(#arrow)"/><text x="30" y="-14">Thrust</text></g>
 <g class="layer" data-layer="thermal"><path d="M-15 -26q-10 -10 0 -20t0 -20M0 -26q-10 -10 0 -20t0 -20M15 -26q-10 -10 0 -20t0 -20" fill="none" stroke="#ffaf82" stroke-width="2"/></g>
 <g class="layer" data-layer="attitude"><path d="M0 0H70M0 0V-65" stroke="#6de3d1" stroke-width="2" marker-end="url(#arrow)"/><path d="M-37 -22A43 43 0 1 1 -41 16" fill="none" stroke="#ffd481" marker-end="url(#gold-arrow)"/><text x="52" y="-46">Torque</text></g>
 <g id="bus"><path d="M-42 -13H-15V13H-42ZM15 -13H42V13H15Z" fill="#224f72" stroke="#86c6df"/><path d="M-29 -13V13M29 -13V13M-42 0H42" stroke="#86c6df" opacity=".5"/><rect x="-13" y="-19" width="26" height="38" rx="3" fill="#becbd0" stroke="#f5fbff"/><path d="M-8 -14H8V4H-8Z" fill="#657b8d"/></g>
</g>
<g class="layer" data-layer="burns"><text x="195" y="390">Thrust history → acceleration + propellant use</text></g>
<g class="layer" data-layer="thermal"><text x="170" y="390">Prescribed temperature → band emission</text></g>
<g class="layer" data-layer="attitude"><text x="174" y="390">Attitude ↔ forces, torques and projected area</text></g>
</svg>
<div><div class="controls"><button id="play" type="button" aria-pressed="false">Play</button><label for="phase">Move around the concept orbit<input id="phase" type="range" min="0" max="360" value="40" step="1"></label></div><p class="scene-note">Geometry and effects are exaggerated to explain connections. The examples below contain the computed results.</p></div>
</section>
<section class="detail" aria-labelledby="topic-title"><div id="kicker" class="eyebrow"></div><h2 id="topic-title"></h2><p id="summary"></p><ol id="chain" class="chain" aria-label="Model connections"></ol><p class="boundary"><strong>Model boundary</strong><span id="limits"></span></p><details><summary>Python entry points</summary><ul id="apis" class="apis"></ul></details></section>
</div>
<section class="examples" aria-labelledby="example-heading"><div class="examples-heading"><h2 id="example-heading">Explore the computed examples</h2><p id="demo-context"></p></div><div id="demos" class="demo-grid"></div></section>
<p class="footer">Space Situational Awareness Toolkit · This map describes SSATK workflows and its SSAPy adapters. Demo status comes from this gallery build. The concept map works offline; external source links require a connection.</p>
<noscript><p>Enable JavaScript to interact with the map. Browse the <a href="index.html">gallery</a> for computed examples.</p>__NO_SCRIPT__</noscript>
</main>
<script type="application/json" id="topic-data">__TOPIC_DATA__</script>
<script>
(() => {
 'use strict';
 const topics = JSON.parse(document.getElementById('topic-data').textContent);
 const el = id => document.getElementById(id);
 let selected = topics[0].id;
 let playing = false;
 let frame = null;
 let lastTime = null;
 let phase = Number(el('phase').value);
 const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
 const labels = {integrators:'Integrators', gravity:'Gravity', shadows:'Shadows', lighting:'Lighting', thermal:'Thermal', attitude:'Attitude', maneuvers:'Maneuvers', burns:'Thrust burns'};
 function node(tag, text, className) {
   const item = document.createElement(tag);
   if (text !== undefined) item.textContent = text;
   if (className) item.className = className;
   return item;
 }
 for (const [index, topic] of topics.entries()) {
   const button = node('button', undefined, 'topic');
   button.type = 'button'; button.dataset.topic = topic.id;
   button.append(node('span', String(index + 1).padStart(2, '0')), node('strong', labels[topic.id]));
   button.addEventListener('click', () => select(topic.id, true));
   el('topics').append(button);
 }
 for (let i = 0; i < 28; i++) {
   const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
   const a = i * Math.PI / 14;
   dot.setAttribute('cx', 340 + 222 * Math.cos(a)); dot.setAttribute('cy', 220 - 137 * Math.sin(a));
   dot.setAttribute('r', i % 4 === 0 ? 4 : 2); el('samples').append(dot);
 }
 function updateScene() {
   const a = Number(el('phase').value) * Math.PI / 180;
   const x = 340 + 222 * Math.cos(a), y = 220 - 137 * Math.sin(a);
   el('craft').setAttribute('transform', `translate(${x} ${y})`);
   el('sightline').setAttribute('d', `M135 ${y}H${x}L590 345`);
   el('bus').setAttribute('transform', selected === 'attitude' ? `rotate(${-Number(el('phase').value)})` : '');
 }
 function select(id, changeHash) {
   const topic = topics.find(t => t.id === id) || topics[0]; selected = topic.id;
   if (changeHash) history.replaceState(null, '', '#' + topic.id);
   document.querySelectorAll('.topic').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.topic === selected)));
   document.querySelectorAll('.layer').forEach(layer => layer.classList.toggle('active', layer.dataset.layer === selected));
   el('scene-title').textContent = topic.title + ' · connections';
   el('diagram-title').textContent = topic.title + ' concept diagram';
   el('kicker').textContent = topic.kicker; el('topic-title').textContent = topic.title;
   el('summary').textContent = topic.summary; el('limits').textContent = topic.limits;
   el('chain').replaceChildren(...topic.chain.map(text => node('li', text)));
   el('apis').replaceChildren(...topic.apis.map(api => { const li = node('li'); li.append(node('code', api)); return li; }));
   el('demo-context').textContent = topic.title + ' / runnable workflows';
   el('demos').replaceChildren();
   for (const demo of topic.demos) {
     const card = node('article', undefined, 'demo');
     if (demo.preview) { const img = node('img'); img.src = demo.preview; img.alt = demo.title + ' output'; img.loading = 'lazy'; card.append(img); }
     const body = node('div', undefined, 'demo-body');
     body.append(node('span', demo.status, 'status ' + (demo.status === 'success' ? 'success' : demo.status === 'failed' ? 'failed' : '')));
     body.append(node('h3', demo.title));
     if (demo.href) { const link = node('a', 'View outputs and build log →'); link.href = demo.href; body.append(link); }
     body.append(node('code', demo.command));
     const source = node('a', 'Read demo source ↗'); source.href = demo.source; body.append(source);
     card.append(body); el('demos').append(card);
   }
   updateScene();
 }
 function tick(time) {
   if (!playing) return;
   if (lastTime !== null) phase = (phase + Math.min(time - lastTime, 100) * .02) % 360;
   el('phase').value = phase;
   lastTime = time; updateScene(); frame = requestAnimationFrame(tick);
 }
 function setPlaying(value) {
   playing = value; el('play').textContent = playing ? 'Pause' : 'Play';
   el('play').setAttribute('aria-pressed', String(playing));
   if (frame !== null) cancelAnimationFrame(frame);
   frame = null; lastTime = null;
   if (playing) frame = requestAnimationFrame(tick);
 }
 el('play').addEventListener('click', () => setPlaying(!playing));
 el('phase').addEventListener('input', () => { setPlaying(false); phase = Number(el('phase').value); updateScene(); });
 document.addEventListener('visibilitychange', () => { if (document.hidden) setPlaying(false); });
 reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) setPlaying(false); });
 window.addEventListener('hashchange', () => select(location.hash.slice(1), false));
 select(location.hash.slice(1), false);
})();
</script>
</body>
</html>
'''
