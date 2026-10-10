"""Per-fragment WebGL Moon for lunar plots.

Why this exists
---------------
Plotly colours 3D surfaces per vertex, so visible detail is capped by the
mesh, not the data: at 180x360 the whole Moon gets about 65k colour samples
regardless of how good the textures are. Per-pixel ray casting removes that
ceiling, but produces an image rather than an interactive scene.

WebGL shades per FRAGMENT, which removes the same ceiling while staying
interactive. It also lets the GPU filter
textures anisotropically, so the per-pixel surface footprint is averaged
rather than point-sampled. That is the direct cure for the limb speckle in
ray-cast renders near the limb, where one screen pixel can span several DEM
cells and nearest-neighbour sampling picks one of them arbitrarily.

The photometry uses Lommel-Seeliger reflectance from real incidence and
emission angles, with an opposition surge. See ``_FRAG`` below.

What it needs
-------------
The baked textures from ``ssatk-bake-moon``, in the cache directory
(~/.ssatk/moon by default, or $SSATK_CACHE). Nothing is
read from the lunar data package and nothing large is committed.

Why it serves over HTTP
-----------------------
WebGL refuses to upload textures fetched over file:// -- Chrome treats
each file as an opaque origin and Firefox has since 68 -- so a
double-click HTML pointing at sibling textures fails with a security
error. Inlining the textures as base64 works but produces a large document.
Instead ``show()`` serves the page and the cache together on a loopback port.

Note this viewer does NOT set logarithmicDepthBuffer. The scene spans a
1737 km Moon and an orbit a few thousand km out, so ordinary depth has
ample precision. That deliberately avoids the trap in the satellite
viewer, where built-in materials write logarithmic depth and hand-written
ShaderMaterials write ordinary depth.

Usage
-----
    from ssatk.plots.moon_webgl import moon_webgl, show

    path = moon_webgl(r=r_m, t=t_gps, save_path="moon.html")
    show(path)                       # serves and opens a browser

    # An SSAPy Orbit can be sampled directly; pass a propagator when needed.
    path = moon_webgl(orbit=orbit, t=t_gps, propagator=prop,
                      save_path="moon_orbit.html")

    # Or load sampled r/t positions, a set of tracks, or Keplerian elements.
    path = moon_webgl(orbit_json="moon_orbit.json",
                      save_path="moon_json.html")

    python -m ssatk.plots.moon_webgl        # demo, no arguments

``r`` is in metres in the GCRF frame and is converted internally to the lunar
body-fixed frame (SSAPy ``MoonOrientation``, DE440 principal axes), whose +X
is 0 deg selenographic longitude, matching the texture. Pass
``r_frame="moon_centered"`` for Moon-centred data already in body-fixed axes.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import socketserver
import threading
import webbrowser

import numpy as np


def load_orbit_json(source):
    """Load orbit JSON through the I/O parser without eager optional imports."""
    from ssatk.io.orbit_json import load_orbit_json as _load_orbit_json

    return _load_orbit_json(source)

R_MOON_KM = 1737.4

_ASSETS = ("moon_albedo.jpg", "moon_normal.png", "moon_horizon_meta.json")


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def cache_dir():
    """Where ``ssatk-bake-moon`` writes."""
    from ssatk._paths import cache_env_dir

    env = cache_env_dir()
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".ssatk", "moon")


_THREE_FILES = ("three.min.js", "OrbitControls.js")


def ensure_three(directory=None):
    """Return local URLs for the packaged three.js runtime."""
    d = os.path.expanduser(directory) if directory else cache_dir()
    urls = {}
    for name in _THREE_FILES:
        local = os.path.join(d, name)
        packaged = os.path.join(os.path.dirname(__file__), name)
        if not os.path.exists(local) and os.path.exists(packaged):
            local = packaged
        if not os.path.exists(local):
            raise FileNotFoundError(
                f"moon_webgl: missing packaged runtime {name}"
            )
        urls[name] = f"/runtime/{name}" if local == packaged else f"/cache/{name}"
    return urls


def find_moon_cache(path=None):
    """
    Locate the baked textures, or say precisely how to make them.

    Returns (directory, metadata dict).
    """
    d = os.path.expanduser(path) if path else cache_dir()
    missing = [f for f in _ASSETS if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise FileNotFoundError(
            f"moon_webgl: missing baked textures in {d}\n"
            f"  absent: {', '.join(missing)}\n\n"
            "Build them once with:\n"
            "  ssatk-bake-moon")

    with open(os.path.join(d, "moon_horizon_meta.json")) as f:
        meta = json.load(f)

    n_files = meta["n_az"] // 4
    for k in range(n_files):
        p = os.path.join(d, f"moon_horizon_{k}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"moon_webgl: {p} missing; re-run the bake")
    return d, meta


# --------------------------------------------------------------------------
# shaders
# --------------------------------------------------------------------------

_VERT = """
varying vec3 vPos;
varying vec2 vUv;
void main() {
    vPos = position;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

# Lommel-Seeliger reflectance with an opposition surge:
#     ls    = mu0 / (mu0 + mu)
#     B     = 1 / (1 + tan(g/2) / 0.06)
#     shade = ls * (1 + B) / (1 + B * 0.5)
_FRAG = """
precision highp float;

uniform sampler2D uAlbedo;
uniform sampler2D uNormal;
uniform sampler2D uHorizon0;
uniform sampler2D uHorizon1;
uniform sampler2D uHorizon2;
uniform sampler2D uHorizon3;
uniform vec3  uSunDir;          // unit, lunar-fixed
uniform float uExposure;
uniform float uSlopeGain;
uniform float uNAz;
uniform float uAngLo;           // horizon decode, degrees
uniform float uAngHi;
uniform float uShadows;         // 1 enable, 0 disable

varying vec3 vPos;
varying vec2 vUv;

const float PI = 3.141592653589793;

// One horizon bearing. Bearings are packed four to an RGBA texture, so
// bearing k lives in file k/4, channel k%4. GLSL ES 1.0 forbids dynamic
// indexing of samplers, hence the explicit chain.
float horizonAt(vec2 uv, int idx) {
    int tex = idx / 4;
    int ch  = idx - tex * 4;
    vec4 v;
    if      (tex == 0) v = texture2D(uHorizon0, uv);
    else if (tex == 1) v = texture2D(uHorizon1, uv);
    else if (tex == 2) v = texture2D(uHorizon2, uv);
    else               v = texture2D(uHorizon3, uv);
    if      (ch == 0) return v.r;
    else if (ch == 1) return v.g;
    else if (ch == 2) return v.b;
    return v.a;
}

void main() {
    vec3 up = normalize(vPos);

    // local tangent frame; east = z_hat x up, north = up x east.
    // East/north tangent basis for the equirectangular texture.
    vec3 zAxis = vec3(0.0, 0.0, 1.0);
    vec3 east  = cross(zAxis, up);
    float eLen = length(east);
    east  = eLen > 1e-6 ? east / eLen : vec3(1.0, 0.0, 0.0);
    vec3 north = cross(up, east);

    // XY normal map; z is reconstructed because storing it wastes the
    // encoding -- near flat ground one 8-bit level of z spans 7.2 degrees.
    vec2 nxy = texture2D(uNormal, vUv).rg * 2.0 - 1.0;
    nxy *= uSlopeGain;
    float nz = sqrt(max(1.0 - dot(nxy, nxy), 0.0));
    vec3 N = normalize(nxy.x * east + nxy.y * north + nz * up);

    vec3 viewDir = normalize(cameraPosition - vPos);

    float mu0 = max(dot(N, uSunDir), 0.0);
    float mu  = max(dot(N, viewDir), 0.0);

    // One horizon lookup replaces a per-fragment 3D ray march.
    // Bearing 0 is north, increasing east.
    if (uShadows > 0.5 && mu0 > 0.0) {
        float sunElev = asin(clamp(dot(uSunDir, up), -1.0, 1.0));
        float sunAz   = atan(dot(uSunDir, east), dot(uSunDir, north));
        if (sunAz < 0.0) sunAz += 2.0 * PI;

        float f  = sunAz / (2.0 * PI) * uNAz;
        float i0 = floor(f);
        float wa = f - i0;
        int a0 = int(mod(i0, uNAz));
        int a1 = int(mod(i0 + 1.0, uNAz));

        float h = mix(horizonAt(vUv, a0), horizonAt(vUv, a1), wa);
        float hAng = radians(uAngLo + h * (uAngHi - uAngLo));
        if (sunElev < hAng) mu0 = 0.0;
    }

    // Lommel-Seeliger with opposition surge.
    float ls = (mu0 + mu > 0.0) ? mu0 / (mu0 + mu + 1e-9) : 0.0;
    float g  = acos(clamp(dot(uSunDir, viewDir), -1.0, 1.0));
    float B  = 1.0 / (1.0 + tan(min(g, PI * 0.5 - 1e-3) * 0.5) / 0.06);
    float shade = ls * (1.0 + B) / (1.0 + B * 0.5);

    vec3 alb = texture2D(uAlbedo, vUv).rgb;
    gl_FragColor = vec4(clamp(alb * shade * uExposure, 0.0, 1.0), 1.0);
}
"""


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;
            font-family:system-ui,sans-serif;color:#EAEAEA}
  #c{display:block;width:100%;height:100%}
  #hud{position:absolute;left:14px;top:12px;font-size:13px;line-height:1.5;
       text-shadow:0 1px 3px #000;pointer-events:none}
  #hud b{font-size:15px}
  #load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
        font-size:14px;color:#9aa}
  #playback{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);
            display:flex;align-items:center;gap:9px;padding:6px 10px;
            border:1px solid rgba(255,255,255,.16);border-radius:999px;
            background:rgba(8,10,14,.64);backdrop-filter:blur(5px);
            font-size:11px;color:#d9dde5}
  #playback[hidden]{display:none}
  #playback button{width:28px;height:28px;padding:0;border-radius:50%;
                   border:1px solid rgba(255,190,62,.65);background:#17130c;
                   color:#ffc04d;font-size:12px;cursor:pointer}
  #playback button:hover{background:#2a2111}
  #orbit-toolbar{position:absolute;right:14px;top:14px;width:min(310px,calc(100vw - 28px));
                 max-height:calc(100vh - 28px);box-sizing:border-box;overflow:auto;
                 border:1px solid rgba(255,255,255,.16);border-radius:12px;
                 background:rgba(8,10,14,.82);backdrop-filter:blur(8px);
                 box-shadow:0 10px 32px rgba(0,0,0,.38);font-size:12px;color:#d9dde5}
  #orbit-toolbar.collapsed .toolbar-body{display:none}
  .toolbar-heading{display:flex;align-items:center;justify-content:space-between;
                   gap:8px;padding:10px 11px;border-bottom:1px solid rgba(255,255,255,.1)}
  .toolbar-heading strong{font-size:13px;color:#fff}
  .toolbar-body{padding:10px 11px 12px}
  .toolbar-section+.toolbar-section{margin-top:12px;padding-top:12px;
                                    border-top:1px solid rgba(255,255,255,.1)}
  .toolbar-label{display:block;margin-bottom:6px;color:#f0f2f6;font-weight:600}
  .toolbar-help{margin:5px 0 0;color:#98a1b2;font-size:10px;line-height:1.35}
  .toolbar-row{display:flex;align-items:center;gap:6px}
  .toolbar-row+.toolbar-row{margin-top:7px}
  #orbit-toolbar button,#orbit-toolbar select,#orbit-toolbar input[type=number],
  .file-button{box-sizing:border-box;border:1px solid rgba(255,255,255,.18);
               border-radius:6px;background:#151922;color:#e8ebf2;font:inherit}
  #orbit-toolbar button,.file-button{padding:6px 8px;cursor:pointer;text-align:center}
  #orbit-toolbar button:hover,.file-button:hover{border-color:rgba(255,190,62,.7);
                                                  background:#211d15}
  #orbit-toolbar input:disabled{opacity:.5;cursor:not-allowed}
  #toolbar-collapse{padding:2px 7px!important;font-size:15px;line-height:1}
  #orbit-file-input{position:absolute;width:1px;height:1px;overflow:hidden;
                    clip:rect(0,0,0,0);white-space:nowrap}
  .file-button{display:block;flex:1;background:#2a2111;color:#ffd17a;
               border-color:rgba(255,190,62,.55)}
  #clear-orbits{white-space:nowrap}
  #orbit-load-status{min-height:16px;margin-top:6px;color:#9bea66;font-size:10px}
  #orbit-load-status.error{color:#ff8a96}
  #orbit-list{display:grid;gap:6px;max-height:238px;overflow:auto}
  .orbit-empty{padding:9px;border:1px dashed rgba(255,255,255,.15);border-radius:7px;
               color:#8992a2;text-align:center}
  .orbit-row{display:grid;grid-template-columns:auto 10px minmax(0,1fr) auto auto;
             align-items:center;gap:6px;padding:7px;border:1px solid rgba(255,255,255,.1);
             border-radius:7px;background:rgba(255,255,255,.035)}
  .orbit-swatch{width:9px;height:9px;border-radius:50%}
  .orbit-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#f4f5f8}
  .orbit-action{padding:3px 6px!important;line-height:1.2}
  .orbit-period{grid-column:2/6;display:flex;align-items:center;gap:5px;color:#929bad;
                font-size:10px}
  .orbit-period input{width:86px;padding:4px 5px}
  #time-step-mode{flex:1;padding:6px}
  #time-step-value{width:82px;padding:6px}
  #time-step-unit{min-width:48px;color:#aeb6c5}
  #step-back,#step-forward{flex:1}
  @media(max-width:700px){
    #orbit-toolbar{right:8px;top:8px;max-height:calc(100vh - 70px)}
    #hud{max-width:calc(100vw - 345px)}
  }
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><b>{title}</b><br><span id="sub">{subtitle}</span></div>
<div id="load">loading textures…</div>
<div id="playback" hidden>
  <button id="playback-toggle" type="button" aria-label="Pause animation">&#10074;&#10074;</button>
  <span id="playback-status"></span>
</div>
<aside id="orbit-toolbar" aria-label="Orbit controls">
  <div class="toolbar-heading">
    <strong>Orbit manager</strong>
    <button id="toolbar-collapse" type="button" aria-label="Collapse orbit manager"
            aria-controls="orbit-toolbar-body" aria-expanded="true">&#8722;</button>
  </div>
  <div id="orbit-toolbar-body" class="toolbar-body">
    <section class="toolbar-section">
      <span class="toolbar-label">Load orbit files</span>
      <div class="toolbar-row">
        <label class="file-button" for="orbit-file-input">Add orbit JSON</label>
        <input id="orbit-file-input" type="file" accept=".json,application/json" multiple>
        <button id="clear-orbits" type="button">Clear</button>
      </div>
      <p class="toolbar-help">Sampled Moon-centred positions; choose one or several files.</p>
      <div id="orbit-load-status" role="status" aria-live="polite"></div>
    </section>
    <section class="toolbar-section">
      <span class="toolbar-label">Loaded orbits <span id="orbit-count"></span></span>
      <div id="orbit-list"></div>
    </section>
    <section class="toolbar-section">
      <label class="toolbar-label" for="time-step-mode">Manual time step</label>
      <div class="toolbar-row">
        <select id="time-step-mode">
          <option value="orbit">Per orbit</option>
          <option value="day">Per day</option>
        </select>
        <input id="time-step-value" type="number" min="0.000001" step="any" value="0.05"
               aria-label="Time step amount">
        <span id="time-step-unit">orbits</span>
      </div>
      <div class="toolbar-row">
        <button id="step-back" type="button">&#8722; step</button>
        <button id="step-forward" type="button">+ step</button>
        <button id="reset-time" type="button">Reset</button>
      </div>
      <p id="time-step-help" class="toolbar-help">
        Buttons jump by the selected fraction; playback runs at the rate shown below.
      </p>
    </section>
  </div>
</aside>

{three_script}
{orbit_script}
<script>
const META   = {meta};
const ORBITS = {orbit};
const STARS  = {stars};
const ASSET_URLS = {asset_urls};
const R      = {radius};
const VERT   = {vert};
const FRAG   = {frag};
const ANIMATION_SECONDS = {animation_seconds};

const scene  = new THREE.Scene();
let orbitRadius = R;
if (Array.isArray(ORBITS)) {
  for (const track of ORBITS) {
    const p = track && track.xyz;
    if (p && p.length >= 3) {
      for (let i = 0; i < p.length; i += 3) {
        orbitRadius = Math.max(orbitRadius,
          Math.hypot(p[i], p[i + 1], p[i + 2]));
      }
    }
  }
}
const camera = new THREE.PerspectiveCamera(35, innerWidth/innerHeight, 1, 1e7);
function fitDistance(radius) {
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * camera.aspect);
  return Math.max(R * 3.75,
    radius * 1.15 / Math.sin(Math.min(verticalHalfFov, horizontalHalfFov)));
}
const viewDistance = fitDistance(orbitRadius);
let framedRadius = orbitRadius;
camera.position.set(viewDistance*0.856, -viewDistance*0.428, viewDistance*0.294);
camera.up.set(0,0,1);

const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),
                                          antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = R*1.05;
controls.maxDistance = Math.max(R*40, orbitRadius*8);

const loader = new THREE.TextureLoader();
const maxAniso = renderer.capabilities.getMaxAnisotropy();

// Anisotropy is the point of moving to WebGL: it averages the surface
// footprint of each pixel instead of point-sampling it, which is what
// removes the limb speckle. Horizon textures are exempt -- they store a
// max over a bearing, and mip-averaging a maximum is not a maximum.
function tex(name, aniso) {
  const t = loader.load(ASSET_URLS[name]);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.minFilter = aniso ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = aniso;
  if (aniso) t.anisotropy = maxAniso;
  return t;
}

const uniforms = {
  uAlbedo:  {value: tex('moon_albedo.jpg', true)},
  uNormal:  {value: tex('moon_normal.png', true)},
  uHorizon0:{value: tex('moon_horizon_0.png', false)},
  uHorizon1:{value: tex('moon_horizon_1.png', false)},
  uHorizon2:{value: tex('moon_horizon_2.png', false)},
  uHorizon3:{value: tex('moon_horizon_3.png', false)},
  uSunDir:  {value: new THREE.Vector3(1,0,0)},
  uExposure:{value: {exposure}},
  uSlopeGain:{value: 1.0},
  uNAz:     {value: META.n_az},
  uAngLo:   {value: META.angle_lo_deg},
  uAngHi:   {value: META.angle_hi_deg},
  uShadows: {value: 1.0}
};

// SphereGeometry is Y-up; rotate so +Z is lunar north, matching the frame
// the texture bake uses. Its own UVs are used rather than deriving
// them from position, because a derived longitude wraps discontinuously and
// the mip selection then shows a seam.
const geom = new THREE.SphereGeometry(R, 256, 128);
geom.rotateX(Math.PI/2);
const moon = new THREE.Mesh(geom, new THREE.ShaderMaterial({
  uniforms: uniforms, vertexShader: VERT, fragmentShader: FRAG
}));
scene.add(moon);

// Use the Toolkit's catalogue-backed Moon-fixed starfield.
if (STARS) {
  const starGeometry = new THREE.BufferGeometry();
  starGeometry.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(STARS.p), 3));
  starGeometry.setAttribute('color',
    new THREE.BufferAttribute(new Float32Array(STARS.c), 3));
  starGeometry.setAttribute('ps',
    new THREE.BufferAttribute(new Float32Array(STARS.s), 1));
  const stars = new THREE.Points(starGeometry, new THREE.ShaderMaterial({
    vertexShader: `attribute float ps;varying vec3 vC;void main(){vC=color;
      gl_PointSize=ps;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader: `varying vec3 vC;void main(){vec2 d=gl_PointCoord-vec2(.5);
      float a=smoothstep(.5,.1,length(d));if(a<0.35)discard;
      gl_FragColor=vec4(vC*a,1.);}`,
    vertexColors: true, transparent: false, depthWrite: true, depthTest: true
  }));
  stars.renderOrder = -1;
  scene.add(stars);
}

const orbitVisuals = [];
let nextOrbitId = 1;
const markerCanvas = document.createElement('canvas');
markerCanvas.width = markerCanvas.height = 64;
const markerContext = markerCanvas.getContext('2d');
const markerGlow = markerContext.createRadialGradient(32, 32, 2, 32, 32, 31);
markerGlow.addColorStop(0.0, 'rgba(255,255,255,1)');
markerGlow.addColorStop(0.22, 'rgba(255,205,96,.98)');
markerGlow.addColorStop(0.48, 'rgba(255,157,37,.58)');
markerGlow.addColorStop(1.0, 'rgba(255,128,0,0)');
markerContext.fillStyle = markerGlow;
markerContext.fillRect(0, 0, 64, 64);
const markerMap = new THREE.CanvasTexture(markerCanvas);
const orbitPalette = [0xffb52e, 0x53d5ff, 0xff5d8f, 0x9bea66,
                      0xc28cff, 0xff754b, 0x57e3c1, 0xf5e663];
const MAX_ANIMATED_MARKERS = 12;
const MAX_RUNTIME_POINTS = 500000;
let orbitLineOpacity = 0.62;
let markerOpacity = 0.9;

const playback = document.getElementById('playback');
const playbackToggle = document.getElementById('playback-toggle');
const playbackStatus = document.getElementById('playback-status');
const orbitToolbar = document.getElementById('orbit-toolbar');
const orbitList = document.getElementById('orbit-list');
const orbitCount = document.getElementById('orbit-count');
const orbitFileInput = document.getElementById('orbit-file-input');
const orbitLoadStatus = document.getElementById('orbit-load-status');
const timeStepMode = document.getElementById('time-step-mode');
const timeStepValue = document.getElementById('time-step-value');
const timeStepUnit = document.getElementById('time-step-unit');
const timeStepHelp = document.getElementById('time-step-help');
let playing = true;
let timelineMode = 'orbit';
let timelineValue = 0;
let lastAnimationTime = performance.now();

function positiveFinite(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function flattenJsonValues(value, output) {
  if (Array.isArray(value)) {
    for (const item of value) flattenJsonValues(item, output);
  } else {
    output.push(value);
  }
  return output;
}

function normaliseUnits(value, fallback) {
  const aliases = {
    m: 'm', meter: 'm', meters: 'm', metre: 'm', metres: 'm',
    km: 'km', kilometer: 'km', kilometers: 'km', kilometre: 'km',
    kilometres: 'km', auto: 'auto'
  };
  const key = String(value === undefined ? fallback : value).toLowerCase();
  if (!aliases[key]) throw new Error("position units must be 'm', 'km', or 'auto'");
  return aliases[key];
}

function normaliseMoonFrame(value) {
  const frame = String(value || 'moon_centered').toLowerCase().replace(/[ -]/g, '_');
  if (['moon_centered', 'moon_centered_earth_moon_rotating',
       'moon_fixed', 'lunar_fixed', 'lunar_centered'].includes(frame)) {
    return 'moon_centered';
  }
  throw new Error(
    "the browser loader accepts sampled Moon-centred positions only (frame='moon_centered'); " +
    'use moon_webgl() in Python to transform GCRF or SSAPy Orbit data'
  );
}

function normalisePositions(value, units) {
  const raw = flattenJsonValues(value, []);
  if (raw.length < 6 || raw.length % 3 !== 0) {
    throw new Error('each orbit needs at least two complete 3-D positions');
  }
  if (raw.length / 3 > MAX_RUNTIME_POINTS) {
    throw new Error('orbit exceeds the browser limit of ' + MAX_RUNTIME_POINTS + ' points');
  }
  const xyz = raw.map(Number);
  if (xyz.some(item => !Number.isFinite(item))) {
    throw new Error('orbit positions must contain only finite numbers');
  }
  let largest = 0;
  for (const item of xyz) largest = Math.max(largest, Math.abs(item));
  const scale = units === 'm' || (units === 'auto' && largest > 1e6) ? 0.001 : 1.0;
  return scale === 1.0 ? xyz : xyz.map(item => item * scale);
}

function normaliseTimes(value, pointCount) {
  if (value === undefined || value === null) return null;
  const raw = flattenJsonValues(value, []);
  if (raw.length !== pointCount) {
    throw new Error('time data must contain one value per orbit position');
  }
  const seconds = raw.map(item => {
    if (typeof item === 'number' && Number.isFinite(item)) return item;
    const numeric = Number(item);
    if (String(item).trim() !== '' && Number.isFinite(numeric)) return numeric;
    const milliseconds = Date.parse(item);
    if (!Number.isFinite(milliseconds)) throw new Error('orbit times must be numeric or ISO dates');
    return milliseconds / 1000;
  });
  const offsets = seconds.map(item => item - seconds[0]);
  for (let index = 1; index < offsets.length; index += 1) {
    if (offsets[index] < offsets[index - 1]) {
      throw new Error('orbit times must be in nondecreasing order');
    }
  }
  return offsets[offsets.length - 1] > 0 ? offsets : null;
}

function timingSeconds(record, defaults, timeOffsets, pointCount) {
  const sources = [record, defaults];
  const definitions = [
    [['duration_seconds', 'period_seconds', 'duration_s', 'period_s',
      'durationSeconds', 'periodSeconds'], 1.0, false],
    [['duration_days', 'period_days', 'durationDays', 'periodDays'], 86400.0, false],
    [['step_seconds', 'sample_step_seconds', 'time_step_seconds',
      'stepSeconds'], 1.0, true],
    [['stride_hours'], 3600.0, true],
    [['step_days', 'sample_step_days', 'stepDays'], 86400.0, true]
  ];
  for (const source of sources) {
    if (!source || typeof source !== 'object') continue;
    for (const [keys, scale, perSample] of definitions) {
      for (const key of keys) {
        if (source[key] === undefined) continue;
        const value = positiveFinite(source[key]);
        if (value === null) throw new Error(key + ' must be a positive finite number');
        return value * scale * (perSample ? pointCount - 1 : 1);
      }
    }
  }
  return timeOffsets ? timeOffsets[timeOffsets.length - 1] : null;
}

function normaliseOrbitJson(payload, fileName) {
  if (!payload || typeof payload !== 'object') {
    throw new Error('orbit JSON root must be an object or array');
  }
  let defaults = Array.isArray(payload) ? {} : payload;
  if (!Array.isArray(payload) && payload.orbit && typeof payload.orbit === 'object' &&
      !Array.isArray(payload.orbit) && !payload.orbits) {
    defaults = Object.assign({}, payload.orbit, payload);
  }
  const records = Array.isArray(payload) ? payload :
    (Array.isArray(defaults.orbits) ? defaults.orbits : [defaults]);
  if (!records.length) throw new Error("orbit JSON 'orbits' array cannot be empty");
  const stem = String(fileName || 'Loaded orbit').replace(/[.]json$/i, '');
  return records.map((record, index) => {
    if (!record || typeof record !== 'object' || Array.isArray(record)) {
      throw new Error('orbit record ' + (index + 1) + ' must be an object');
    }
    const positionKey = ['xyz', 'r', 'positions', 'position'].find(
      key => record[key] !== undefined
    );
    if (!positionKey) {
      if (record.elements || record.keplerian || record.a !== undefined) {
        throw new Error('Keplerian/SSAPy Orbit JSON must be sampled by moon_webgl() in Python first');
      }
      throw new Error('orbit record ' + (index + 1) + ' has no positions or xyz field');
    }
    const frame = record.r_frame !== undefined ? record.r_frame :
      (record.frame !== undefined ? record.frame :
        (defaults.r_frame !== undefined ? defaults.r_frame : defaults.frame));
    normaliseMoonFrame(frame);
    const inheritedUnits = defaults.units !== undefined
      ? defaults.units : defaults.position_units;
    const defaultUnits = inheritedUnits !== undefined
      ? inheritedUnits : (positionKey === 'xyz' ? 'km' : 'auto');
    const units = normaliseUnits(
      record.units !== undefined ? record.units : record.position_units,
      defaultUnits
    );
    const xyz = normalisePositions(record[positionKey], units);
    const timeKey = ['t', 'times', 'time'].find(key => record[key] !== undefined);
    const inheritedTimeKey = ['t', 'times', 'time'].find(key => defaults[key] !== undefined);
    const timeValue = timeKey ? record[timeKey] :
      (inheritedTimeKey ? defaults[inheritedTimeKey] : null);
    const timeOffsets = normaliseTimes(timeValue, xyz.length / 3);
    const durationSeconds = timingSeconds(record, defaults, timeOffsets, xyz.length / 3);
    return {
      name: String(record.name !== undefined ? record.name :
        (record.oid !== undefined ? record.oid :
          (record.id !== undefined ? record.id :
            (records.length === 1 ? stem : stem + ' ' + (index + 1))))),
      xyz: xyz,
      times: timeOffsets,
      durationSeconds: durationSeconds,
      source: fileName || 'Loaded JSON'
    };
  });
}

function colourCss(colour) {
  return '#' + colour.toString(16).padStart(6, '0');
}

function trackMaximumRadius(positions) {
  let maximum = R;
  for (let index = 0; index < positions.length; index += 3) {
    maximum = Math.max(maximum, Math.hypot(
      positions[index], positions[index + 1], positions[index + 2]
    ));
  }
  return maximum;
}

function addOrbitTrack(track, source) {
  if (!track || !track.xyz || track.xyz.length < 6) return null;
  const positions = new Float32Array(track.xyz);
  const pointCount = positions.length / 3;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const colour = orbitPalette[(nextOrbitId - 1) % orbitPalette.length];
  const material = new THREE.LineBasicMaterial({
    color: colour, transparent: true, opacity: orbitLineOpacity,
    depthTest: true, depthWrite: false
  });
  const line = new THREE.Line(geometry, material);
  const name = String(track.name || ('Orbit ' + nextOrbitId));
  line.name = name;
  scene.add(line);

  const maximumRadius = trackMaximumRadius(positions);
  const endGap = Math.hypot(
    positions[0] - positions[positions.length - 3],
    positions[1] - positions[positions.length - 2],
    positions[2] - positions[positions.length - 1]
  );
  const timeOffsets = Array.isArray(track.times) && track.times.length === pointCount
    ? track.times.map(Number) : null;
  const visual = {
    id: nextOrbitId++, name: name, source: source || track.source || 'Embedded',
    positions: positions, pointCount: pointCount, geometry: geometry, material: material,
    line: line, marker: null, colour: colour, visible: true,
    closed: typeof track.closed === 'boolean' ? track.closed :
      endGap < Math.max(R * 0.05, maximumRadius * 0.015),
    timeOffsets: timeOffsets,
    durationSeconds: positiveFinite(track.durationSeconds || track.duration_seconds),
    maximumRadius: maximumRadius
  };
  if (!visual.durationSeconds && timeOffsets) {
    visual.durationSeconds = positiveFinite(timeOffsets[timeOffsets.length - 1]);
  }
  orbitVisuals.push(visual);
  orbitRadius = Math.max(orbitRadius, maximumRadius);
  controls.maxDistance = Math.max(R * 40, orbitRadius * 8);
  refreshOrbitPresentation();
  renderOrbitList();
  return visual;
}

function removeMarker(visual) {
  if (!visual.marker) return;
  scene.remove(visual.marker);
  visual.marker.material.dispose();
  visual.marker = null;
}

function ensureMarker(visual) {
  if (visual.marker) return;
  visual.marker = new THREE.Sprite(new THREE.SpriteMaterial({
    map: markerMap, color: visual.colour, transparent: true, opacity: markerOpacity,
    depthTest: true, depthWrite: false
  }));
  visual.marker.name = visual.name;
  scene.add(visual.marker);
}

function updatePlaybackStatus() {
  const visible = orbitVisuals.filter(visual => visual.visible);
  const animated = visible.filter(visual => visual.marker).length;
  playback.hidden = visible.length === 0;
  if (!visible.length) return;
  let description = visible.length === 1 ? '1 orbit' : visible.length + ' orbits';
  if (animated < visible.length) description += ' · ' + animated + ' animated';
  description += timelineMode === 'orbit'
    ? ' · 1 orbit / ' + ANIMATION_SECONDS + ' s'
    : ' · 1 day / ' + ANIMATION_SECONDS + ' s';
  if (timelineMode === 'day') {
    const untimed = visible.filter(visual => !visual.durationSeconds).length;
    if (untimed) description += ' · ' + untimed + ' need timing';
  }
  playbackStatus.textContent = description;
}

function refreshOrbitPresentation() {
  const visible = orbitVisuals.filter(visual => visual.visible);
  orbitLineOpacity = visible.length === 1
    ? 0.62 : Math.max(0.06, 0.50 / Math.sqrt(Math.max(visible.length, 1)));
  markerOpacity = visible.length <= 8
    ? 0.9 : Math.max(0.28, 1.0 / Math.sqrt(visible.length));
  let markerIndex = 0;
  for (const visual of orbitVisuals) {
    visual.line.visible = visual.visible;
    visual.material.opacity = orbitLineOpacity;
    if (visual.visible && markerIndex < MAX_ANIMATED_MARKERS) {
      ensureMarker(visual);
      visual.marker.visible = true;
      visual.marker.material.opacity = markerOpacity;
      markerIndex += 1;
    } else {
      removeMarker(visual);
    }
  }
  orbitCount.textContent = '(' + orbitVisuals.length + ')';
  updatePlaybackStatus();
}

function recomputeOrbitRadius() {
  orbitRadius = orbitVisuals.reduce(
    (maximum, visual) => Math.max(maximum, visual.maximumRadius), R
  );
  controls.maxDistance = Math.max(R * 40, orbitRadius * 8);
}

function removeOrbit(visual) {
  const index = orbitVisuals.indexOf(visual);
  if (index < 0) return;
  orbitVisuals.splice(index, 1);
  scene.remove(visual.line);
  removeMarker(visual);
  visual.geometry.dispose();
  visual.material.dispose();
  recomputeOrbitRadius();
  refreshOrbitPresentation();
  renderOrbitList();
}

function focusRadius(radius) {
  framedRadius = radius;
  const direction = camera.position.clone().sub(controls.target).normalize();
  controls.target.set(0, 0, 0);
  camera.position.copy(direction.multiplyScalar(fitDistance(radius)));
  controls.update();
}

function focusOrbit(visual) {
  focusRadius(visual.maximumRadius);
}

function renderOrbitList() {
  orbitList.textContent = '';
  orbitList.setAttribute('role', 'list');
  if (!orbitVisuals.length) {
    const empty = document.createElement('div');
    empty.className = 'orbit-empty';
    empty.textContent = 'No orbits loaded';
    orbitList.appendChild(empty);
    return;
  }
  for (const visual of orbitVisuals) {
    const row = document.createElement('div');
    row.className = 'orbit-row';
    row.setAttribute('role', 'listitem');
    const visibility = document.createElement('input');
    visibility.type = 'checkbox';
    visibility.checked = visual.visible;
    visibility.setAttribute('aria-label', 'Show ' + visual.name);
    visibility.addEventListener('change', () => {
      visual.visible = visibility.checked;
      refreshOrbitPresentation();
    });
    const swatch = document.createElement('span');
    swatch.className = 'orbit-swatch';
    swatch.style.background = colourCss(visual.colour);
    const label = document.createElement('span');
    label.className = 'orbit-name';
    label.textContent = visual.name;
    label.title = visual.name + ' — ' + visual.source;
    const focus = document.createElement('button');
    focus.className = 'orbit-action';
    focus.type = 'button';
    focus.textContent = 'Fit';
    focus.setAttribute('aria-label', 'Fit ' + visual.name + ' in view');
    focus.addEventListener('click', () => focusOrbit(visual));
    const remove = document.createElement('button');
    remove.className = 'orbit-action';
    remove.type = 'button';
    remove.innerHTML = '&times;';
    remove.setAttribute('aria-label', 'Remove ' + visual.name);
    remove.addEventListener('click', () => removeOrbit(visual));
    const period = document.createElement('label');
    period.className = 'orbit-period';
    period.textContent = 'Day-mode period/span';
    const periodInput = document.createElement('input');
    periodInput.type = 'number';
    periodInput.min = '0.000001';
    periodInput.step = 'any';
    periodInput.placeholder = 'unknown';
    periodInput.disabled = timelineMode !== 'day';
    periodInput.value = visual.durationSeconds
      ? String(Number((visual.durationSeconds / 86400).toPrecision(7))) : '';
    periodInput.setAttribute('aria-label', visual.name + ' period or span in days');
    periodInput.addEventListener('change', () => {
      if (periodInput.value.trim() === '') {
        visual.durationSeconds = null;
      } else {
        const days = positiveFinite(periodInput.value);
        if (days === null) {
          periodInput.value = visual.durationSeconds
            ? String(visual.durationSeconds / 86400) : '';
          setLoadStatus('Period/span must be a positive number of days.', true);
          return;
        }
        visual.durationSeconds = days * 86400;
      }
      updatePlaybackStatus();
    });
    const dayUnit = document.createElement('span');
    dayUnit.textContent = 'days';
    period.append(periodInput, dayUnit);
    row.append(visibility, swatch, label, focus, remove, period);
    orbitList.appendChild(row);
  }
}

function setSun(az, el) {
  const a = az*Math.PI/180, e = el*Math.PI/180;
  uniforms.uSunDir.value.set(Math.cos(e)*Math.cos(a),
                             Math.cos(e)*Math.sin(a),
                             Math.sin(e)).normalize();
}
setSun({sun_az}, {sun_el});

function togglePlayback() {
  playing = !playing;
  lastAnimationTime = performance.now();
  if (!playing) {
    playbackToggle.innerHTML = '&#9654;';
    playbackToggle.setAttribute('aria-label', 'Play animation');
  } else {
    playbackToggle.innerHTML = '&#10074;&#10074;';
    playbackToggle.setAttribute('aria-label', 'Pause animation');
  }
  updatePlaybackStatus();
}
playbackToggle.addEventListener('click', togglePlayback);

function readStepAmount() {
  const value = positiveFinite(timeStepValue.value);
  if (value !== null) return value;
  timeStepValue.value = timelineMode === 'orbit' ? '0.05' : '0.25';
  setLoadStatus('Time step must be a positive finite number.', true);
  return Number(timeStepValue.value);
}

function stepTimeline(direction) {
  if (playing) togglePlayback();
  timelineValue += direction * readStepAmount();
  updateOrbitAnimation(performance.now(), false);
}

function setTimelineMode(mode) {
  timelineMode = mode === 'day' ? 'day' : 'orbit';
  timelineValue = 0;
  lastAnimationTime = performance.now();
  timeStepUnit.textContent = timelineMode === 'day' ? 'days' : 'orbits';
  timeStepValue.value = timelineMode === 'day' ? '0.25' : '0.05';
  timeStepHelp.textContent = timelineMode === 'day'
    ? "Buttons jump by days; playback runs 1 day per " + ANIMATION_SECONDS +
      " s. Untimed tracks hold position; open tracks hold at their endpoints."
    : 'Buttons jump by orbit fraction; playback runs 1 orbit per ' +
      ANIMATION_SECONDS + ' s.';
  renderOrbitList();
  updateOrbitAnimation(lastAnimationTime, false);
  updatePlaybackStatus();
}

timeStepMode.addEventListener('change', () => setTimelineMode(timeStepMode.value));
document.getElementById('step-back').addEventListener('click', () => stepTimeline(-1));
document.getElementById('step-forward').addEventListener('click', () => stepTimeline(1));
document.getElementById('reset-time').addEventListener('click', () => {
  if (playing) togglePlayback();
  timelineValue = 0;
  lastAnimationTime = performance.now();
  updateOrbitAnimation(lastAnimationTime, false);
});

function setLoadStatus(message, isError) {
  orbitLoadStatus.textContent = message || '';
  orbitLoadStatus.classList.toggle('error', Boolean(isError));
}

function readOrbitFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('could not read ' + file.name));
    reader.onload = () => {
      try {
        let source = String(reader.result);
        if (source.charCodeAt(0) === 0xfeff) source = source.slice(1);
        resolve(normaliseOrbitJson(JSON.parse(source), file.name));
      } catch (error) {
        reject(new Error(file.name + ': ' + error.message));
      }
    };
    reader.readAsText(file);
  });
}

async function loadOrbitFiles(files) {
  let added = 0;
  const errors = [];
  orbitToolbar.setAttribute('aria-busy', 'true');
  for (const file of Array.from(files)) {
    setLoadStatus('Loading ' + file.name + '…');
    try {
      const tracks = await readOrbitFile(file);
      for (const track of tracks) {
        if (addOrbitTrack(track, file.name)) added += 1;
      }
    } catch (error) {
      errors.push(error.message);
    }
  }
  orbitToolbar.setAttribute('aria-busy', 'false');
  if (added) focusRadius(orbitRadius);
  if (errors.length) {
    const prefix = added ? 'Added ' + added + ' orbit(s). ' : '';
    setLoadStatus(prefix + errors.join(' '), true);
  } else if (added) {
    setLoadStatus(
      'Added ' + added + ' orbit' + (added === 1 ? '. ' : 's. ') +
      'Use Fit to frame a selected path.'
    );
  }
}

orbitFileInput.addEventListener('change', async () => {
  await loadOrbitFiles(orbitFileInput.files);
  orbitFileInput.value = '';
});
document.getElementById('clear-orbits').addEventListener('click', () => {
  for (const visual of [...orbitVisuals]) removeOrbit(visual);
  timelineValue = 0;
  setLoadStatus('All orbits cleared.');
});
document.getElementById('toolbar-collapse').addEventListener('click', event => {
  const collapsed = orbitToolbar.classList.toggle('collapsed');
  event.currentTarget.innerHTML = collapsed ? '&#43;' : '&#8722;';
  event.currentTarget.setAttribute('aria-expanded', String(!collapsed));
  event.currentTarget.setAttribute(
    'aria-label', collapsed ? 'Expand orbit manager' : 'Collapse orbit manager'
  );
});

addEventListener('keydown', event => {
  const tag = event.target && event.target.tagName;
  const editing = ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(tag);
  if (event.code === 'Space' && orbitVisuals.length && !editing) {
    event.preventDefault();
    togglePlayback();
  }
});

function wrappedUnit(value) {
  return ((value % 1) + 1) % 1;
}

function setVisualPosition(visual, rawPhase, elapsedMode) {
  if (!visual.marker) return;
  const cycle = wrappedUnit(rawPhase);
  // Orbit-fraction animation reverses partial paths rather than teleporting.
  // Elapsed-day mode instead clamps them to their sampled time span.
  const phase = visual.closed ? cycle : elapsedMode
    ? Math.max(0, Math.min(1, rawPhase))
    : 1 - Math.abs(1 - 2 * cycle);
  let i0;
  let fraction;
  if (visual.timeOffsets && visual.timeOffsets.length === visual.pointCount) {
    const lastOffset = visual.timeOffsets[visual.timeOffsets.length - 1];
    const target = phase * lastOffset;
    if (target >= lastOffset) {
      i0 = visual.pointCount - 2;
      fraction = 1;
    } else {
      let low = 0;
      let high = visual.timeOffsets.length - 1;
      while (low + 1 < high) {
        const middle = Math.floor((low + high) / 2);
        if (visual.timeOffsets[middle] <= target) low = middle;
        else high = middle;
      }
      i0 = Math.min(visual.pointCount - 2, low);
      const interval = visual.timeOffsets[i0 + 1] - visual.timeOffsets[i0];
      fraction = interval > 0 ? (target - visual.timeOffsets[i0]) / interval : 0;
    }
  } else {
    const cursor = phase * (visual.pointCount - 1);
    i0 = Math.min(visual.pointCount - 2, Math.floor(cursor));
    fraction = cursor - i0;
  }
  const index = i0 * 3;
  const positions = visual.positions;
  visual.marker.position.set(
    positions[index] + (positions[index + 3] - positions[index]) * fraction,
    positions[index + 1] + (positions[index + 4] - positions[index + 1]) * fraction,
    positions[index + 2] + (positions[index + 5] - positions[index + 2]) * fraction
  );
  const markerSize = Math.max(
    R * 0.018, visual.marker.position.distanceTo(camera.position) * 0.014
  );
  visual.marker.scale.set(markerSize, markerSize, 1);
}

function updateOrbitAnimation(now, advance=true) {
  const delta = Math.max(0, Math.min(now - lastAnimationTime, 1000));
  lastAnimationTime = now;
  if (advance && playing) timelineValue += delta / (ANIMATION_SECONDS * 1000);
  for (const visual of orbitVisuals) {
    if (!visual.marker) continue;
    const phase = timelineMode === 'orbit' ? timelineValue :
      (visual.durationSeconds ? timelineValue * 86400 / visual.durationSeconds : 0);
    setVisualPosition(visual, phase, timelineMode === 'day');
  }
}

if (Array.isArray(ORBITS)) {
  ORBITS.forEach(track => addOrbitTrack(track, 'Embedded'));
}
renderOrbitList();
refreshOrbitPresentation();
updateOrbitAnimation(lastAnimationTime, false);

THREE.DefaultLoadingManager.onLoad = () =>
  document.getElementById('load').style.display = 'none';

addEventListener('resize', () => {
  const previousFitDistance = fitDistance(framedRadius);
  camera.aspect = innerWidth/innerHeight;
  camera.position.sub(controls.target).multiplyScalar(
    fitDistance(framedRadius) / previousFitDistance
  ).add(controls.target);
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

(function loop(now){
  requestAnimationFrame(loop);
  updateOrbitAnimation(now);
  controls.update();
  renderer.render(scene, camera);
})(performance.now());
</script></body></html>
"""


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _time_as_gps(t):
    """Return numeric GPS seconds for an SSAPy time-like value."""
    if hasattr(t, "gps"):
        t = t.gps
    elif np.size(t):
        values = np.asarray(t, dtype=object).reshape(-1)
        if hasattr(values[0], "gps"):
            t = [value.gps for value in values]
    return np.asarray(t, dtype=float)


def _browser_track_timing(t, point_count, *, duration_seconds=None,
                          step_seconds=None):
    """Build compact relative times and duration for browser animation."""
    result = {}
    if t is not None:
        values = _time_as_gps(t).reshape(-1)
        if values.size not in {1, point_count}:
            raise ValueError("orbit time data must contain one value per position")
        if values.size == point_count:
            if not np.all(np.isfinite(values)):
                raise ValueError("orbit time data must contain only finite values")
            offsets = values - values[0]
            if np.any(np.diff(offsets) < 0.0):
                raise ValueError("orbit time data must be in nondecreasing order")
            if offsets[-1] > 0.0:
                result["times"] = [round(float(value), 6) for value in offsets]
                result["durationSeconds"] = float(offsets[-1])

    if duration_seconds is not None:
        duration = float(duration_seconds)
    elif step_seconds is not None:
        duration = float(step_seconds) * max(0, point_count - 1)
    else:
        duration = None
    if duration is not None:
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("orbit duration must be a positive finite number")
        result["durationSeconds"] = duration
    return result


def _orbit_times(orbit, t, n_steps=360, n_orbits=1.0):
    """Choose sampling times for an SSAPy Orbit when none were supplied."""
    if t is not None:
        return t

    orbit_t = getattr(orbit, "t", 0.0)
    try:
        orbit_t = _time_as_gps(orbit_t).reshape(-1)
    except (TypeError, ValueError):
        orbit_t = np.asarray([0.0])
    start = float(orbit_t[0]) if orbit_t.size else 0.0

    # A vector-valued Orbit already carries its own time samples.  Reuse them
    # instead of asking rv() to resample an already-propagated trajectory.
    try:
        n_positions = (
            np.asarray(getattr(orbit, "r"), dtype=float)
            .reshape(-1, 3)
            .shape[0]
        )
    except (TypeError, ValueError):
        n_positions = 1
    if orbit_t.size == n_positions and n_positions > 1:
        return orbit_t

    try:
        period = float(np.asarray(getattr(orbit, "period"), dtype=float).reshape(-1)[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        period = np.nan
    if np.isfinite(period) and period > 0.0:
        count = max(2, int(n_steps))
        return start + np.linspace(0.0, period * float(n_orbits), count)
    raise ValueError(
        "t is required for an SSAPy Orbit without a finite period; "
        "pass t= explicitly."
    )


def _orbit_xyz(r, t, r_frame, *, units="auto"):
    if r is None:
        return []
    if r_frame not in {"gcrf", "moon_centered"}:
        raise ValueError("r_frame must be 'gcrf' or 'moon_centered'")
    r_arr = np.asarray(r, dtype=float)
    if r_arr.size % 3:
        raise ValueError(f"r must contain 3-vectors; got shape {r_arr.shape}")
    if not np.all(np.isfinite(r_arr)):
        raise ValueError("r must contain only finite position values")
    r_arr = r_arr.reshape(-1, 3)
    if r_frame == "moon_centered":
        if units == "m":
            xyz = r_arr / 1e3
        elif units == "km":
            xyz = r_arr
        else:
            try:
                from .plotutils import normalize_orbit_trajectory
            except ImportError:  # pragma: no cover - direct module loading
                from ssatk.plots.plotutils import normalize_orbit_trajectory
            xyz, _, _ = normalize_orbit_trajectory(r=r_arr, t=t, r_units=units)
    else:
        try:
            from ..coordinates import gcrf_to_lunar_body
        except ImportError:
            from ssatk.coordinates import gcrf_to_lunar_body
        if t is None:
            raise ValueError("t is required to place GCRF positions on the Moon")
        # Resolve units in GCRF, where Earth-centred cislunar magnitudes
        # (~4e5 km or ~4e8 m) are unambiguous, before subtracting the Moon's
        # position in metres.
        if units == "km":
            r_m = r_arr * 1e3
        elif units == "m":
            r_m = r_arr
        else:
            try:
                from .plotutils import normalize_orbit_trajectory
            except ImportError:  # pragma: no cover - direct module loading
                from ssatk.plots.plotutils import normalize_orbit_trajectory
            r_km, _, _ = normalize_orbit_trajectory(r=r_arr, t=t, r_units=units)
            r_m = np.asarray(r_km, dtype=float).reshape(-1, 3) * 1e3
        # The texture's +X is 0 deg selenographic longitude, so tracks must be
        # in the lunar body frame, not the Earth-Moon rotating frame.
        xyz = gcrf_to_lunar_body(r_m, _time_as_gps(t).reshape(-1)) / 1e3
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return [round(float(v), 3) for v in xyz.ravel()]


def _track_from_sample(sample):
    """Convert one parsed JSON track to the compact browser representation."""
    xyz = _orbit_xyz(sample["r"], sample.get("t"), sample["r_frame"], units=sample["units"])
    if len(xyz) < 6:
        raise ValueError("each orbit track must contain at least two positions")
    point_count = len(xyz) // 3
    track = {"xyz": xyz, "name": str(sample.get("name", "Orbit"))}
    track.update(_browser_track_timing(
        sample.get("t"),
        point_count,
        duration_seconds=sample.get("duration_seconds"),
        step_seconds=sample.get("step_seconds"),
    ))
    return track


def _resolve_orbit_positions(r, t, r_frame, orbit, propagator,
                             n_steps, n_orbits, orbit_json=None):
    """Resolve raw positions or an SSAPy Orbit into viewer track samples."""
    if orbit_json is not None:
        if r is not None or orbit is not None:
            raise ValueError("Provide orbit_json separately from r= or orbit=.")
        loaded = load_orbit_json(orbit_json)
        if "tracks" in loaded:
            tracks = [_track_from_sample(sample) for sample in loaded["tracks"]]
            if not tracks:
                raise ValueError("orbit JSON did not contain any tracks")
            return tracks
        r = loaded.get("r")
        t = loaded.get("t", t)
        orbit = loaded.get("orbit")
        r_frame = loaded.get("r_frame", r_frame)
        json_units = loaded.get("units", "auto")
    else:
        json_units = "auto"
    if r_frame not in {"gcrf", "moon_centered"}:
        raise ValueError("r_frame must be 'gcrf' or 'moon_centered'")
    if orbit is None and r is not None and all(
            hasattr(r, attr) for attr in ("r", "v", "t")):
        orbit, r = r, None
    if orbit is not None and r is not None:
        raise ValueError("Provide either orbit= or raw r= positions, not both.")
    if orbit is None:
        if r is None:
            return []
        return [_track_from_sample({
            "r": r,
            "t": t,
            "r_frame": r_frame,
            "units": json_units,
            "name": loaded.get("name", "Orbit") if orbit_json is not None else "Orbit",
            "duration_seconds": loaded.get("duration_seconds")
            if orbit_json is not None else None,
            "step_seconds": loaded.get("step_seconds") if orbit_json is not None else None,
        })]

    times = _orbit_times(orbit, t, n_steps=n_steps, n_orbits=n_orbits)
    try:
        from ssapy import rv as ssapy_rv
    except ImportError as exc:  # pragma: no cover - SSAPy is a package dependency
        raise ImportError("moon_webgl orbit= requires SSAPy's ssapy.rv") from exc
    if propagator is None:
        positions, _ = ssapy_rv(orbit, times)
    else:
        positions, _ = ssapy_rv(orbit, times, propagator=propagator)
    # SSAPy Orbit states are always metres.  Do not use the raw-array
    # magnitude heuristic here: a small metre-valued test orbit is valid too.
    return [_track_from_sample({
        "r": positions,
        "t": times,
        "r_frame": r_frame,
        "units": "m",
        "name": "Orbit",
    })]


def _viewer_page_assets(directory, runtime_urls, embed_assets):
    """Build script tags and texture URLs for served or portable HTML."""
    texture_names = (
        "moon_albedo.jpg",
        "moon_normal.png",
        "moon_horizon_0.png",
        "moon_horizon_1.png",
        "moon_horizon_2.png",
        "moon_horizon_3.png",
    )
    if not embed_assets:
        return (
            f'<script src="{runtime_urls["three.min.js"]}"></script>',
            f'<script src="{runtime_urls["OrbitControls.js"]}"></script>',
            {name: f"cache/{name}" for name in texture_names},
        )

    runtime_paths = {}
    for name in _THREE_FILES:
        local = os.path.join(directory, name)
        if not os.path.exists(local):
            packaged = os.path.join(os.path.dirname(__file__), name)
            if os.path.exists(packaged):
                local = packaged
        if not os.path.exists(local):
            raise FileNotFoundError(
                f"moon_webgl: portable export needs local {name}; "
                "install the package asset or provide a cached runtime"
            )
        runtime_paths[name] = local

    def inline_script(path):
        with open(path, encoding="utf-8") as handle:
            source = handle.read().replace("</script>", "<\\/script>")
        return f"<script>\n{source}\n</script>"

    asset_urls = {}
    for name in texture_names:
        path = os.path.join(directory, name)
        mime = "image/jpeg" if name.endswith(".jpg") else "image/png"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        asset_urls[name] = f"data:{mime};base64,{encoded}"

    return (
        inline_script(runtime_paths["three.min.js"]),
        inline_script(runtime_paths["OrbitControls.js"]),
        asset_urls,
    )


def _starfield_payload(epoch=None):
    """Reuse the Toolkit's catalogue-backed Moon-fixed WebGL starfield."""
    try:
        from .starfield import moon_fixed_webgl_stars
    except ImportError:  # pragma: no cover - script mode
        from ssatk.plots.starfield import moon_fixed_webgl_stars
    return moon_fixed_webgl_stars(epoch=epoch)


def moon_webgl(r=None, t=None, r_frame="gcrf",
               title="Moon — lunar-fixed frame", subtitle=None,
               sun_azimuth_deg=35.0, sun_elevation_deg=8.0,
               exposure=1.4, cache=None, save_path=None,
               orbit=None, propagator=None, n_steps=360, n_orbits=1.0,
               orbit_json=None, animation_seconds=24.0,
               embed_assets=False):
    """
    Write an interactive Moon viewer page.

    Parameters
    ----------
    r : array-like, optional
        Trajectory positions, as accepted by ``moon_plot_3d``.
    t : array-like, optional
        Trajectory times.
    r_frame : {"gcrf", "moon_centered"}
        Frame of ``r``. ``"gcrf"`` positions are moved into the lunar
        body-fixed frame with :func:`ssatk.coordinates.gcrf_to_lunar_body`
        and require ``t``. ``"moon_centered"`` positions are drawn as given,
        so they must already be in body-fixed axes.
    title, subtitle : str, optional
        Page heading and supporting text.
    sun_azimuth_deg, sun_elevation_deg : float
        Fixed Sun direction in the lunar-fixed frame, in degrees.
    exposure : float
        Display gain, not radiometry.
    cache : path-like, optional
        Directory containing baked Moon assets.
    save_path : path-like, optional
        Output path; defaults beside the cache.
    orbit : ssapy.Orbit, optional
        Orbit sampled with ``ssapy.rv``.
    propagator : object, optional
        Propagator passed to ``ssapy.rv``.
    n_steps : int
        Samples used for an automatically generated orbit track.
    n_orbits : float
        Periods covered by an automatically generated track.
    orbit_json : path-like or mapping, optional
        Sampled positions, orbit records, or Keplerian elements. See
        :func:`load_orbit_json`.
    animation_seconds : float
        Browser seconds per orbit or simulated day.
    embed_assets : bool
        Inline textures and JavaScript for a portable HTML file.

    Returns
    -------
    str
        Path to the written HTML page.
    """
    if r_frame not in {"gcrf", "moon_centered"}:
        raise ValueError("r_frame must be 'gcrf' or 'moon_centered'")
    animation_seconds = float(animation_seconds)
    if not np.isfinite(animation_seconds) or animation_seconds <= 0.0:
        raise ValueError("animation_seconds must be a positive finite number")
    d, meta = find_moon_cache(cache)
    tracks = _resolve_orbit_positions(
        r, t, r_frame, orbit, propagator, n_steps, n_orbits, orbit_json)

    if subtitle is None:
        if tracks:
            radii = np.concatenate([
                np.linalg.norm(np.asarray(track["xyz"]).reshape(-1, 3), axis=1)
                for track in tracks
            ])
            subtitle = (f"{len(tracks)} orbit{'s' if len(tracks) != 1 else ''} · "
                        f"altitude {radii.min() - R_MOON_KM:.0f}–{radii.max() - R_MOON_KM:.0f} km · "
                        f"LOLA relief · Lommel-Seeliger")
        else:
            subtitle = ("LOLA relief · Lommel-Seeliger · "
                        f"{meta['n_az']} bearing horizon shadows")

    three = ensure_three(d)
    three_script, orbit_script, asset_urls = _viewer_page_assets(
        d, three, bool(embed_assets))
    star_epoch = t
    if star_epoch is None and orbit is not None:
        star_epoch = getattr(orbit, "t", None)
    if star_epoch is None and orbit_json is not None:
        loaded = load_orbit_json(orbit_json)
        if "tracks" in loaded:
            star_epoch = next(
                (track.get("t") for track in loaded["tracks"] if track.get("t") is not None),
                None,
            )
        else:
            star_epoch = getattr(loaded.get("orbit"), "t", None)
    if star_epoch is not None:
        star_epoch = _time_as_gps(star_epoch).reshape(-1)[0]
    stars = _starfield_payload(star_epoch)
    doc = _HTML
    for key, val in (("{title}", title),
                     ("{three_script}", three_script),
                     ("{orbit_script}", orbit_script),
                     ("{subtitle}", subtitle),
                     ("{meta}", json.dumps(meta)),
                     ("{orbit}", json.dumps(tracks)),
                     ("{stars}", json.dumps(stars)),
                     ("{asset_urls}", json.dumps(asset_urls)),
                     ("{radius}", repr(R_MOON_KM)),
                     ("{vert}", json.dumps(_VERT)),
                     ("{frag}", json.dumps(_FRAG)),
                     ("{sun_az}", repr(float(sun_azimuth_deg))),
                     ("{sun_el}", repr(float(sun_elevation_deg))),
                     ("{exposure}", repr(float(exposure))),
                     ("{animation_seconds}", repr(animation_seconds))):
        doc = doc.replace(key, val)

    out = os.path.expanduser(save_path) if save_path \
        else os.path.join(d, "moon_webgl.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[moon_webgl] wrote {out}  ({len(doc) / 1024:.0f} KB)")
    return out


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------

def _handler(html_path, cache_path):
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if not self._allowed():
                self.send_error(404)
                return
            super().do_GET()

        def do_HEAD(self):
            if not self._allowed():
                self.send_error(404)
                return
            super().do_HEAD()

        def _allowed(self):
            p = self.path.split("?", 1)[0].split("#", 1)[0]
            return p in ("/", "/index.html") or (
                p.startswith("/cache/") and os.path.basename(p) == p[7:]
            ) or p in {f"/runtime/{name}" for name in _THREE_FILES}

        def translate_path(self, path):
            p = path.split("?", 1)[0].split("#", 1)[0]
            if p in ("/", "/index.html"):
                return html_path
            if p.startswith("/cache/"):
                return os.path.join(cache_path, os.path.basename(p))
            if p.startswith("/runtime/"):
                return os.path.join(os.path.dirname(__file__), os.path.basename(p))
            return html_path

        def log_message(self, *a):
            pass
    return H


def show(html_path=None, cache=None, port=0, open_browser=True):
    """
    Serve the page and its textures on loopback, and open a browser.

    Textures cannot be loaded over file:// -- WebGL rejects them as
    cross-origin -- so the page needs an HTTP origin even locally.
    Blocks until interrupted.
    """
    d, _ = find_moon_cache(cache)
    html_path = html_path or os.path.join(d, "moon_webgl.html")
    if not os.path.exists(html_path):
        html_path = moon_webgl(cache=cache)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port),
                                _handler(html_path, d)) as srv:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        print(f"[moon_webgl] serving {url}   (ctrl-C to stop)")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[moon_webgl] stopped")


if __name__ == "__main__":
    show(moon_webgl())
