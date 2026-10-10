// ============================================================================
// Satellite 3D scene -- Three.js
//
// Built to exceed the reference viewer's Earth quality:
//   reference: SphereGeometry(96,96) + MeshPhongMaterial(map=texture,
//              shininess=4) -- one flat-lit texture, no day/night, no ocean
//              specular variation, no clouds, no atmosphere.
//   this file: custom shader blending real day + night (city lights)
//              textures across a soft terminator, a real ocean-only
//              specular mask (land stays matte, water glints), a separate
//              translucent cloud shell, and a Fresnel atmosphere glow.
//
// Textures: NASA day imagery + matteason/live-cloud-maps' night-lights,
// ocean specular mask, and cloud-alpha layer (public domain / CC0, made
// for exactly this use case).
// ============================================================================

const R_EARTH_KM = SSAPY_CONSTANTS.R_EARTH_KM;
const R_MOON_KM = SSAPY_CONSTANTS.R_MOON_KM;
const R_SUN_KM = SSAPY_CONSTANTS.R_SUN_KM;
const AU_KM = SSAPY_CONSTANTS.AU_KM;
const MOON_MEAN_DISTANCE_KM = SSAPY_CONSTANTS.MOON_MEAN_DISTANCE_KM;
// Render the Sun and Moon at physical Earth-centered distances and radii. This
// preserves their nearly equal angular diameters (~0.5 deg) instead of using
// oversized nearby markers that look like scene props.
const SUN_RENDER_DISTANCE_KM = AU_KM;
const SUN_RENDER_RADIUS_KM = R_SUN_KM;
const SUN_GLOW_RADIUS_KM = R_SUN_KM * 1.15;
const MOON_RENDER_DISTANCE_KM = MOON_MEAN_DISTANCE_KM;
const MOON_RENDER_RADIUS_KM = R_MOON_KM;
const STARFIELD_RADIUS_KM = AU_KM * 1.45;
const CAMERA_FAR_KM = STARFIELD_RADIUS_KM * 1.08;

// ---------------------------------------------------------------------------
// No external TLEs are bundled. Supply a licensed JSON/CSV catalog through
// build(database_path=...), or pass state vectors through satellite_viewer().
//
// This covers every base category in tle_updater.py's SATELLITE_GROUPS (the
// same taxonomy that project already uses for TLE fetching), not just a
// handful. Each entry has one of these `mode`s:
//
//   mode: 'tle'        -- real SGP4 propagation from an actual sourced TLE.
//   mode: 'fixed'      -- JWST only. It orbits the Sun-Earth L2 point in a
//                         halo orbit ~1.5 million km away -- a real TLE
//                         exists but describes an eccentricity of ~0.988 and
//                         a ~51-day period that SGP4 cannot meaningfully
//                         propagate the way it does for a near-Earth
//                         satellite (multiple independent sources confirm
//                         this outright: Webb's deep-space trajectory needs
//                         a full Sun-Earth-Moon gravitational model, not
//                         TLE/SGP4). Shown at a fixed, clearly-labeled
//                         illustrative position instead of a fabricated orbit.
//   mode: 'state'      -- supplied r/v/t samples, plus optional q attitude
//                         samples, interpolated without browser propagation.
// ---------------------------------------------------------------------------
const SATELLITE_CATALOG = {
  jwst: {
    name: 'James Webb Space Telescope', type: 'JWST', mode: 'fixed', noradId: 50463,
    note: 'JWST orbits the Sun-Earth L2 point (~1.5 million km from Earth) ' +
          'in a halo orbit, not a near-Earth ellipse -- SGP4 cannot ' +
          'meaningfully propagate it. Shown at a fixed, real-distance ' +
          'illustrative position along the anti-sunward direction, not a ' +
          'real-time-propagated one.',
  },
  beidou: {
    name: 'BeiDou MEO (representative)', type: 'BEIDOU', mode: 'metadata',
    a: R_EARTH_KM + 21528, e: 0.001, inc: 55.0, raan: 45, argPerigee: 0,
    note: 'No literal TLE was efficiently sourced for this one -- built from ' +
          'BeiDou-3 MEO orbit fact-sheet parameters (~21,528 km altitude, ' +
          '55 degrees inclination); provide state vectors for visualization.',
  },
  weather_noaa: {
    name: 'NOAA POES (representative)', type: 'WEATHER_LEO', mode: 'metadata',
    a: R_EARTH_KM + 850, e: 0.001, inc: 98.7, raan: 100, argPerigee: 0,
    note: 'Representative sun-synchronous polar-orbit parameters for the ' +
          'NOAA POES series (~850 km altitude, 98.7 degrees), not a specific TLE.',
  },
  weather_goes: {
    name: 'GOES-R (representative)', type: 'WEATHER_GEO', mode: 'metadata',
    a: R_EARTH_KM + 35786, e: 0.0005, inc: 0.5, raan: 0, argPerigee: 0,
    note: 'Representative geostationary parameters (~35,786 km altitude), not a specific TLE.',
  },
  weather_metop: {
    name: 'MetOp (representative)', type: 'WEATHER_LEO', mode: 'metadata',
    a: R_EARTH_KM + 817, e: 0.001, inc: 98.7, raan: 200, argPerigee: 0,
    note: 'Representative sun-synchronous polar-orbit parameters for MetOp (~817 km, 98.7 degrees).',
  },
  landsat: {
    name: 'Landsat (representative)', type: 'EARTH_OBS', mode: 'metadata',
    a: R_EARTH_KM + 705, e: 0.001, inc: 98.2, raan: 30, argPerigee: 0,
    note: 'Representative sun-synchronous parameters for the Landsat series (~705 km, 98.2 degrees).',
  },
  sentinel: {
    name: 'Sentinel (representative)', type: 'EARTH_OBS', mode: 'metadata',
    a: R_EARTH_KM + 700, e: 0.001, inc: 98.6, raan: 130, argPerigee: 0,
    note: 'Representative sun-synchronous parameters for the Copernicus Sentinel series (~700 km, 98.6 degrees).',
  },
  terra_aqua: {
    name: 'Terra/Aqua (representative)', type: 'EARTH_OBS', mode: 'metadata',
    a: R_EARTH_KM + 705, e: 0.001, inc: 98.2, raan: 230, argPerigee: 0,
    note: 'Representative "A-train" sun-synchronous parameters for Terra/Aqua (~705 km, 98.2 degrees).',
  },
  icesat: {
    name: 'ICESat-2 (representative)', type: 'EARTH_OBS', mode: 'metadata',
    a: R_EARTH_KM + 496, e: 0.001, inc: 92.0, raan: 330, argPerigee: 0,
    note: 'Representative near-polar parameters for ICESat-2 (~496 km, 92 degrees).',
  },
  chandra: {
    name: 'Chandra X-ray Observatory (representative)', type: 'OBSERVATORY', mode: 'metadata', noradId: 25867,
    a: 80900, e: 0.72, inc: 76.7, raan: 0, argPerigee: 300,
    note: 'Representative highly-elliptical parameters for Chandra ' +
          '(perigee ~16,000 km, apogee ~133,000 km, 76.7 degrees) -- its huge, ' +
          'stretched-out orbit is real and central to the mission (it spends ' +
          'most of a ~64-hour period lingering near apogee for uninterrupted ' +
          'observing), so this deliberately is NOT approximated as circular.',
  },
  fermi: {
    name: 'Fermi Gamma-ray Space Telescope (representative)', type: 'OBSERVATORY', mode: 'metadata', noradId: 33053,
    a: R_EARTH_KM + 535, e: 0.001, inc: 25.58, raan: 29, argPerigee: 131,
    note: 'Representative parameters from Fermi\'s published orbital fact sheet (~535 km, 25.58 degrees).',
  },
  swift: {
    name: 'Swift Observatory (representative)', type: 'OBSERVATORY', mode: 'metadata', noradId: 28485,
    a: R_EARTH_KM + 600, e: 0.001, inc: 20.6, raan: 60, argPerigee: 0,
    note: 'Representative parameters for Swift (~600 km, 20.6 degrees).',
  },
  tess: {
    name: 'TESS (representative)', type: 'TESS', mode: 'metadata', noradId: 43435,
    a: 247900, e: 0.54, inc: 37.0, raan: 90, argPerigee: 0,
    note: 'Representative parameters for TESS\'s real lunar-resonant P/2 orbit ' +
          '(perigee ~108,000 km, apogee ~375,000 km, ~37 degrees) -- deliberately ' +
          'not circularized, since the whole point of this orbit design is the ' +
          'huge apogee for near-continuous sky viewing.',
  },
  cislunar_demo: {
    name: 'Cislunar orbit demo', type: 'TESS', mode: 'metadata',
    a: 205000, e: 0.86, inc: 28.5, raan: 45, argPerigee: 180,
    note: 'Representative Earth-centered cislunar transfer-class orbit with ' +
          'perigee ~28,700 km and apogee ~381,300 km near lunar distance. This ' +
          'is an idealized SSATK viewer example, not a mission ephemeris.',
  },
};

// ---------------------------------------------------------------------------
// Support for searching/selecting from a user's full local satellite
// database (JSON or CSV), not just
// the curated entries above. Any of those ~31,000 real satellites can be
// searched and added -- but only the curated examples above have a bespoke or archetype
// model built for them specifically. For everything else, this classifies
// by name (mirroring tle_updater.py's own _classify_object() exactly, for
// consistency between the Python tooling and this viewer) and falls back to
// a generic payload / rocket-body / debris model. If a searched satellite's
// NORAD ID happens to match one of the curated ones, it reuses that
// entry's real bespoke model and real note instead of the generic fallback.
// ---------------------------------------------------------------------------
function noradFromTle(line1) {
  const n = parseInt((line1 || '').substring(2, 7), 10);
  return isNaN(n) ? null : n;
}

const CURATED_NORAD_TO_KEY = {};
for (const [key, entry] of Object.entries(SATELLITE_CATALOG)) {
  let nid = null;
  if (entry.mode === 'tle') nid = noradFromTle(entry.tle1);
  if (nid !== null) CURATED_NORAD_TO_KEY[nid] = key;
}

function classifyByName(name) {
  const upper = (name || '').toUpperCase();
  if (['DEB', 'COOLANT', 'SHROUD', 'WESTFORD NEEDLES'].some(s => upper.includes(s))) return 'DEBRIS';
  if (['R/B', 'AKM', 'PKM'].some(s => upper.includes(s))) return 'ROCKET_BODY';
  // Constellation name patterns -- deliberately conservative (precision over
  // recall): skips ambiguous substrings that could mislabel unrelated
  // satellites (e.g. "COSMOS" covers many non-GLONASS Soviet/Russian
  // programs; "GSAT" collides with India's unrelated ISRO GSAT series).
  if (upper.includes('GPS ') || upper.includes('NAVSTAR')) return 'GPS';
  if (upper.includes('GALILEO')) return 'GALILEO';
  if (upper.includes('GLONASS')) return 'GLONASS';
  if (upper.includes('BEIDOU')) return 'BEIDOU';
  if (upper.includes('GOES')) return 'WEATHER_GEO';
  if (upper.includes('NOAA') || upper.includes('METOP')) return 'WEATHER_LEO';
  if (upper.includes('LANDSAT') || upper.includes('SENTINEL')) return 'EARTH_OBS';
  if (upper.includes('STARLINK')) return 'STARLINK';
  if (upper.includes('ONEWEB')) return 'STARLINK'; // not the same design -- but the closest existing archetype (flat bus + single wing) for another flat-panel LEO broadband constellation, better than falling to a fully generic box
  return 'GENERIC_PAYLOAD';
}

const EMBEDDED_STATE_VECTOR_TRACKS =
  (typeof STATE_VECTOR_TRACKS !== 'undefined' && Array.isArray(STATE_VECTOR_TRACKS))
    ? STATE_VECTOR_TRACKS : [];
const ACTIVE_PROPAGATOR =
  (typeof PROPAGATOR !== 'undefined' && typeof PROPAGATOR === 'string')
    ? PROPAGATOR : 'sgp4';
const STATE_VECTOR_CATALOG = {};
for (let index = 0; index < EMBEDDED_STATE_VECTOR_TRACKS.length; index++) {
  const track = EMBEDDED_STATE_VECTOR_TRACKS[index];
  const key = `state_${index}`;
  STATE_VECTOR_CATALOG[key] = {
    name: track.name || `State vector track ${index + 1}`,
    type: classifyByName(track.name || ''), mode: 'state',
    stateVector: track,
    note: 'Propagated state vectors supplied by ssapy.rv; no browser propagation is applied.',
  };
}
Object.assign(SATELLITE_CATALOG, STATE_VECTOR_CATALOG);
const DEFAULT_DEMO_SATELLITES = Object.keys(STATE_VECTOR_CATALOG);

// Resolves one record from the loaded database into an { key, entry } pair
// ready for addSatellite(). Reuses a curated key/entry (bespoke model, real
// note) when the NORAD ID matches a metadata entry above.
function resolveDbRecord(record) {
  const nid = record._nid != null ? record._nid :
    (record.line1 ? noradFromTle(record.line1) : record.noradId);
  if (nid !== null && CURATED_NORAD_TO_KEY[nid]) {
    const key = CURATED_NORAD_TO_KEY[nid];
    return { key, entry: SATELLITE_CATALOG[key] };
  }
  if (record.mode === 'metadata' || record.type === 'metadata') {
    const matchingTle = loadedDatabase && loadedDatabase.find(item =>
      item.mode === 'tle' && item._nid === nid);
    if (matchingTle) {
      const key = 'db_' + nid;
      return {
        key,
        entry: {
          name: record.name, type: classifyByName(record.name), mode: 'tle',
          tle1: matchingTle.line1, tle2: matchingTle.line2,
          note: 'DISCOS metadata matched to a TLE record with the same NORAD ID.',
        },
      };
    }
    const key = 'db_' + (nid !== null ? nid : record.name);
    return {
      key,
      entry: {
        name: record.name, type: classifyByName(record.name), mode: 'metadata',
        noradId: nid, metadata: record.metadata,
        note: 'DISCOS metadata record. A current TLE or state vector is required to plot this object.',
      },
    };
  }
  const type = classifyByName(record.name);
  const key = 'db_' + (nid !== null ? nid : record.name);
  const entry = {
    name: record.name, type, mode: 'tle',
    tle1: record.line1, tle2: record.line2,
    note: 'From your loaded satellite database -- not one ' +
          'of the curated types above, so this uses a generic model ' +
          `classified by name as "${type.toLowerCase().replace('_', ' ')}".`,
  };
  return { key, entry };
}

let scene, camera, renderer, earthMesh, cloudMesh, atmosphereMesh, starfield, sunLight, earthShineLight;
let sunMesh, sunGlowMesh, moonMesh;
let sunDirection = new THREE.Vector3(1, 0.35, 0.25).normalize();
let camTheta = 0.9, camPhi = 1.1, camDist = 26000;
const camTarget = new THREE.Vector3(0, 0, 0);
let isDragging = false, lastX = 0, lastY = 0;
let cloudsGroup;

// TLE-driven "selected" satellite state
// Multiple satellites can be active at once (default: none without supplied
// state vectors or a licensed catalog). Each entry
// in this Map holds its own model, orbit line, and sim-clock state --
// there's no shared "the selected satellite" anymore, since more than one
// can be animating independently at the same time.
const activeSatellites = new Map(); // key -> { entry, satrec, model, orbitLine, periodMin, fixedPos, r0, framingR }

// ---------------------------------------------------------------------------
// Global simulation clock -- shared by every active satellite, replacing
// the old per-satellite scheme where each one independently compressed its
// own orbit into a ~24-second loop with no relationship to any other
// satellite's displayed moment (two satellites shown together never
// actually corresponded to the same point in time). Default scale=1 means
// real time: propagate to the actual current moment, so accuracy is
// limited only by how fresh the underlying TLE is -- not by anything this
// visualization adds on top. Scale > 1 speeds things up for watching
// motion happen (useful given some of these periods run to days), while
// keeping every active satellite reading off the same clock.
// ---------------------------------------------------------------------------
let simTimeScale = 1;
let simClockAnchorRealMs = INITIAL_SIM_TIME_MS == null ? Date.now() : INITIAL_SIM_TIME_MS;
let simClockAnchorPerfMs = performance.now();
let cloudDriftAccumulator = 0; // clouds drift slowly relative to the surface, on top of GMST-tracked Earth rotation
let lastClockDisplayUpdateMs = 0;
let labelsEnabled = true;

function getCurrentSimMs() {
  return simClockAnchorRealMs + (performance.now() - simClockAnchorPerfMs) * simTimeScale;
}

function setSimTimeScale(newScale) {
  // Reanchor at the current simulated moment so changing speed doesn't
  // cause a jump -- only the rate going forward changes.
  simClockAnchorRealMs = getCurrentSimMs();
  simClockAnchorPerfMs = performance.now();
  simTimeScale = newScale;
}

// Jumps the sim clock to an arbitrary moment (past or future), keeping
// whatever speed is currently selected running forward from there.
function setSimTime(targetMs) {
  simClockAnchorRealMs = targetMs;
  simClockAnchorPerfMs = performance.now();
}

function stepSimTime(deltaMs) {
  setSimTime(getCurrentSimMs() + deltaMs);
}

// Pause is just scale=0 -- reuses setSimTimeScale's own reanchoring so
// there's no special-case freeze/unfreeze logic to keep in sync separately.
let isPaused = false;
function togglePause() {
  isPaused = !isPaused;
  const select = document.getElementById('time-scale-select');
  const btn = document.getElementById('time-pause-btn');
  if (isPaused) {
    setSimTimeScale(0);
    if (btn) btn.textContent = 'Play';
  } else {
    setSimTimeScale(select ? parseFloat(select.value) : 1);
    if (btn) btn.textContent = 'Pause';
  }
}
let loadedDatabase = null; // normalized {name, type, line1, line2} records from JSON or CSV
const MAX_ACTIVE_SATELLITES = 1000; // raised from 300. Detailed multi-mesh models don't scale to this many, so LOD kicks in automatically -- see SIMPLE_MODEL_ABOVE / ORBIT_LINE_ABOVE below.
// Satellite models are deliberately stylized display glyphs, but their size is
// fixed across LEO/MEO/GEO/cislunar regimes so a distant orbit does not turn
// into a giant spacecraft. Labels and orbit lines carry the scale context.
const SATELLITE_GLYPH_SIZE_KM = 900;
// Level of detail. A detailed model is ~4 meshes (ISS is 21) and each orbit
// line is another draw call, so 1000 fully-detailed satellites would be ~5,000
// draw calls -- enough to stutter on integrated graphics. Past these
// thresholds, newly added satellites get a single-mesh glyph and no orbit
// line, which keeps 1000 active comfortably under ~1,200 draw calls. Detail is
// decided per satellite at add time, so your first picks stay fully detailed
// and the bulk load behind them is lightweight.
const SIMPLE_MODEL_ABOVE = 150; // beyond this many active, new satellites use a simple glyph
const ORBIT_LINE_ABOVE = 150;   // beyond this many active, new satellites skip the orbit line
// Above this active count, floating text labels are suppressed (they overlap
// into an unreadable mass and the DOM writes dominate the frame); the info
// panel still lists everything in compact form.
const LABEL_DECLUTTER_ABOVE = 60;
let _labelsDecluttered = false;

// --- Ground tracks (sub-satellite path on the Earth surface) ---------------
// Toggled globally for all active satellites. Distinct cyan so a track reads
// apart from the white orbit lines and the yellow labels.
let groundTracksEnabled = false;
let quaternionAttitudeEnabled = false;
const GROUND_TRACK_COLOR = 0x5fd6e6;
const GROUND_TRACK_SAMPLES = 180;       // vertices along the painted trail
const GROUND_TRACK_MAX_SPAN_MIN = 24 * 60; // cap the trailing window (one full period for LEO/MEO; keeps GEO/HEO to <=1 sidereal day)
const GROUND_TRACK_SURFACE_R = R_EARTH_KM * 1.004; // hair above the surface so the line/marker aren't z-fought by the globe

// --- Conjunction screening -------------------------------------------------
// On-demand closest-approach search between all active propagatable pairs over
// a forward window. Results feed a clickable event list (jump to TCA) and a
// live risk-colored connector line drawn near each time of closest approach.
let conjunctionEvents = [];               // [{keyA,keyB,nameA,nameB,tcaMs,missKm,relVelKmS}]
const conjunctionLines = new Map();       // "keyA|keyB" -> THREE.Line connector
let conjunctionThresholdKm = 10;
let conjunctionWindowHours = 24;
const CONJ_COARSE_STEP_MS = 60000;        // 1-min coarse grid; refined per candidate
const CONJ_LINE_SHOW_MIN = 12;            // draw the connector within +/- this of TCA

function init() {
  const container = document.getElementById('scene-container');
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 10, CAMERA_FAR_KM);
  renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  // Faint ambient only -- the Earth shader handles its own day/night
  // lighting internally; this mainly lights the satellites.
  scene.add(new THREE.AmbientLight(0x404050, 0.38));
  sunLight = new THREE.DirectionalLight(0xffffff, 0.85);
  // Earthshine: real spacecraft on the night side are weakly lit by sunlight
  // reflected off Earth, so their nadir face isn't a flat black silhouette.
  // A dim, cool fill from the origin direction reads as that, and keeps model
  // detail legible against the dark side of the globe.
  earthShineLight = new THREE.DirectionalLight(0x8fa8c8, 0.22);
  earthShineLight.position.set(0, 0, 0);
  scene.add(earthShineLight);
  sunLight.position.copy(sunDirection.clone().multiplyScalar(100000));
  scene.add(sunLight);

  buildStarfield();
  buildEarth();
  buildClouds();
  buildAtmosphere();
  buildCelestialMarkers();
  setupControls();
  setupTimeControls();
  setupDatabaseSearch();
  setupAnalysisControls();
  if (EMBEDDED_STATE_VECTOR_TRACKS.length && EMBEDDED_STATE_VECTOR_TRACKS[0].t.length) {
    setSimTime(EMBEDDED_STATE_VECTOR_TRACKS[0].t[0]);
  }
  DEFAULT_DEMO_SATELLITES.forEach(key => addSatellite(key));
  reframeCamera();
  updateInfoPanel();
  animate();
  window.addEventListener('resize', onResize);
}

// ---------------------------------------------------------------------------
// Search across a user-loaded satellite database. This is now the only
// selection mechanism (the
// original curated dropdown was removed once search could resolve
// those examples by NORAD ID/name -- see CURATED_NORAD_TO_KEY and
// classifyByName above -- plus everything else in the loaded database).
// A standalone HTML file cannot fetch a local file directly, so this uses a
// file picker and parses the selected file in the browser.
// ---------------------------------------------------------------------------
const SEARCH_RESULTS_LIMIT = 60; // cap rendered rows -- don't build 31,000 DOM nodes for a broad query
function formatDateForInput(date) {
  const pad = n => String(n).padStart(2, '0');
  return date.getUTCFullYear() + '-' + pad(date.getUTCMonth() + 1) + '-' + pad(date.getUTCDate()) +
    'T' + pad(date.getUTCHours()) + ':' + pad(date.getUTCMinutes());
}

function setupTimeControls() {
  const select = document.getElementById('time-scale-select');
  const pauseBtn = document.getElementById('time-pause-btn');
  const jumpInput = document.getElementById('time-jump-input');
  const jumpBtn = document.getElementById('time-jump-btn');
  const nowBtn = document.getElementById('time-now-btn');
  if (!select) return;

  select.addEventListener('change', () => {
    if (!isPaused) setSimTimeScale(parseFloat(select.value)); // if paused, just remember the preference for when Play is next clicked
  });

  if (pauseBtn) pauseBtn.addEventListener('click', togglePause);

  document.querySelectorAll('.time-step-btn').forEach(btn => {
    btn.addEventListener('click', () => stepSimTime(parseInt(btn.dataset.step, 10)));
  });

  if (nowBtn) nowBtn.addEventListener('click', () => setSimTime(Date.now()));

  if (jumpBtn && jumpInput) {
    jumpBtn.addEventListener('click', () => {
      if (!jumpInput.value) return;
      // Treat the input as UTC explicitly (append 'Z'), consistent with the
      // UTC-labeled clock readout elsewhere in this panel -- avoids the
      // ambiguity of datetime-local's usual local-timezone interpretation.
      const ms = Date.parse(jumpInput.value + 'Z');
      if (!isNaN(ms)) setSimTime(ms);
    });
    jumpInput.value = formatDateForInput(new Date(getCurrentSimMs()));
  }
}

// ---------------------------------------------------------------------------
// Flexible satellite-record parsing -- accepts multiple real-world record
// shapes, not just this project's own {name,line1,line2} format, so an
// analyst or intern can drop in a file from Space-Track, the Unified Data
// Library (UDL), or a similar source and have it "just work" without
// knowing or caring which exact format it is.
//
// IMPORTANT / HONESTY NOTE: the exact UDL field names could not be verified
// against a trustworthy public schema at build time. Rather than hardcode a
// single guessed schema (which would silently mis-parse real files), this
// checks a RANGE of plausible field-name variants for each piece of data,
// and -- critically -- if it can't find a usable TLE in a record, that
// record is reported as skipped rather than silently turned into garbage.
// If a real UDL file doesn't load correctly, the on-screen diagnostic will
// say what was and wasn't recognized, which is the fix-it signal.
// ---------------------------------------------------------------------------

// Case-insensitive "find the first present, non-empty field from a list of
// candidate names" -- how this tolerates naming differences across sources.
function pickField(obj, candidates) {
  for (const c of candidates) {
    if (obj[c] != null && obj[c] !== '') return obj[c];
  }
  // case-insensitive fallback pass catches casing differences without
  // listing every variant.
  const lowerMap = {};
  for (const k of Object.keys(obj)) lowerMap[k.toLowerCase()] = obj[k];
  for (const c of candidates) {
    const v = lowerMap[c.toLowerCase()];
    if (v != null && v !== '') return v;
  }
  return null;
}

const TLE_LINE1_FIELDS = ['line1', 'TLE_LINE1', 'tleLine1', 'LINE1'];
const TLE_LINE2_FIELDS = ['line2', 'TLE_LINE2', 'tleLine2', 'LINE2'];
const TLE_NAME_FIELDS = ['name', 'OBJECT_NAME', 'objectName', 'satNo',
  'satno', 'idOnOrbit', 'origObjectId', 'NORAD_CAT_ID', 'noradId'];

function recordAttributes(raw) {
  if (!raw || typeof raw !== 'object') return {};
  // ESA DISCOS uses JSON:API records: the useful catalog fields live under
  // `attributes`, while ordinary TLE catalogs put them at the top level.
  return raw.attributes && typeof raw.attributes === 'object'
    ? Object.assign({}, raw, raw.attributes) : raw;
}

function noradFromRecord(raw) {
  const fields = recordAttributes(raw);
  const value = pickField(fields, ['noradId', 'NORAD_CAT_ID', 'satno', 'satNo']);
  const nid = Number(value);
  return Number.isInteger(nid) && nid > 0 ? nid : null;
}

// Normalizes one raw record (any recognized shape) into this project's
// internal {name, line1, line2} form, or returns null if it has no usable
// TLE. Only TLE/elset data is handled here -- state vectors are a separate,
// non-TLE representation and are explicitly reported as unsupported rather
// than silently dropped (see loadSatelliteDatabase()).
function normalizeSatelliteRecord(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const fields = recordAttributes(raw);

  // TLE line 1 / line 2, across known and plausible field names:
  //   this project:     line1 / line2
  //   uppercase JSON:   TLE_LINE1 / TLE_LINE2
  //   UDL elset (plausible variants): line1/line2, tleLine1/tleLine2
  const line1 = pickField(fields, TLE_LINE1_FIELDS);
  const line2 = pickField(fields, TLE_LINE2_FIELDS);
  const noradId = noradFromRecord(fields);
  if (typeof line1 !== 'string' || typeof line2 !== 'string' ||
      line1.length < 60 || line2.length < 60) {
    // Metadata-only DISCOS records remain searchable. They can be plotted
    // when their NORAD ID matches a curated TLE; otherwise the UI reports
    // that an orbit source is still required.
    if (noradId === null) return null;
    const metadataName = pickField(fields, TLE_NAME_FIELDS) || `NORAD ${noradId}`;
    return {
      name: String(metadataName), type: 'metadata', mode: 'metadata',
      noradId, metadata: fields,
    };
  }

  // Name, across known/plausible variants (fall back to a catalog-number
  // label if no name field is present, which real elset feeds sometimes omit)
  let name = pickField(fields, TLE_NAME_FIELDS);
  if (name == null) {
    const nid = noradFromTle(line1);
    name = nid != null ? ('NORAD ' + nid) : 'Unknown object';
  }
  return { name: String(name), type: 'tle', mode: 'tle', line1, line2,
    noradId: noradId === null ? noradFromTle(line1) : noradId };
}

// Unwraps the top-level structure to an array of records, tolerating the
// several ways this kind of data gets wrapped: a bare array, or an object
// with the array under a common key.
function extractRecordArray(data) {
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') {
    for (const key of ['data', 'records', 'results', 'elsets', 'stateVectors', 'items']) {
      if (Array.isArray(data[key])) return data[key];
    }
  }
  return null;
}

// Detects records that ARE orbital data but in a form this viewer can't
// propagate (state vectors: position+velocity snapshots rather than TLEs).
// Used only to give an accurate skip reason, not to process them.
function looksLikeStateVector(raw) {
  if (!raw || typeof raw !== 'object') return false;
  const fields = recordAttributes(raw);
  return pickField(fields, ['xpos', 'x', 'posX', 'xPos']) != null &&
         pickField(fields, ['xvel', 'xdot', 'velX', 'xVel']) != null;
}

function parseCsvDatabase(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"' && field === '') {
      quoted = true;
    } else if (ch === ',') {
      row.push(field); field = '';
    } else if (ch === '\n') {
      row.push(field); field = '';
      if (row.some(value => value.trim() !== '')) rows.push(row);
      row = [];
    } else if (ch !== '\r') {
      field += ch;
    }
  }
  if (quoted) throw new Error('CSV has an unterminated quoted field');
  if (field !== '' || row.length) {
    row.push(field);
    if (row.some(value => value.trim() !== '')) rows.push(row);
  }
  if (rows.length === 0) return [];
  const headers = rows.shift().map(value => value.trim());
  if (headers.some(header => !header)) throw new Error('CSV has an empty column name');
  return rows.map(values => Object.fromEntries(
    headers.map((header, index) => [header, values[index] == null ? '' : values[index]])
  ));
}

async function parseSatelliteDatabaseFile(file) {
  const extension = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
  if (extension === '.json') return JSON.parse(await file.text());
  if (extension === '.csv') return parseCsvDatabase(await file.text());
  throw new Error('unsupported format; choose JSON or CSV');
}

function showDatabaseLoadError(message) {
  loadedDatabase = null;
  const statusEl = document.getElementById('db-status');
  const searchInput = document.getElementById('db-search-input');
  if (statusEl) statusEl.textContent = message;
  if (searchInput) {
    searchInput.disabled = true;
    searchInput.placeholder = 'Load a database first...';
  }
  renderSearchResults('');
  return false;
}

function loadSatelliteDatabase(parsed, sourceName) {
  const statusEl = document.getElementById('db-status');
  const searchInput = document.getElementById('db-search-input');
  const rawRecords = extractRecordArray(parsed);
  if (!rawRecords) {
    return showDatabaseLoadError('Unrecognized database structure: expected an array of ' +
      'records, or an object containing one under "data"/"records"/"elsets"/etc.');
  }

  const valid = [];
  let stateVectorCount = 0, metadataCount = 0, unrecognizedCount = 0;
  for (const raw of rawRecords) {
    const norm = normalizeSatelliteRecord(raw);
    if (norm) {
      norm._nid = norm.noradId != null ? norm.noradId :
        (norm.line1 ? noradFromTle(norm.line1) : null); // precompute once, not per-keystroke
      if (norm.mode === 'metadata') metadataCount++;
      valid.push(norm);
    } else if (looksLikeStateVector(raw)) {
      stateVectorCount++;
    } else {
      unrecognizedCount++;
    }
  }

  if (valid.length === 0) {
    let reason;
    if (stateVectorCount > 0) {
      reason = `found ${stateVectorCount.toLocaleString()} state-vector records, ` +
        `but this viewer accepts state vectors through satellite_viewer(), not as a catalog file.`;
    } else if (unrecognizedCount > 0) {
      reason = `none of the ${unrecognizedCount.toLocaleString()} records contained a ` +
        `recognizable TLE (checked line1/line2, TLE_LINE1/2, tleLine1/2). If this is a ` +
        `valid orbit file, the field names may differ from what's expected.`;
    } else {
      reason = 'the file contained no records.';
    }
    return showDatabaseLoadError('No satellites loaded: ' + reason);
  }

  loadedDatabase = valid;
  const notes = [];
  if (metadataCount > 0) notes.push(`${metadataCount.toLocaleString()} DISCOS metadata records`);
  if (stateVectorCount > 0) notes.push(`${stateVectorCount.toLocaleString()} state-vector records skipped (use satellite_viewer(r, v, t))`);
  if (unrecognizedCount > 0) notes.push(`${unrecognizedCount.toLocaleString()} unrecognized records skipped`);
  statusEl.textContent = `Loaded ${valid.length.toLocaleString()} satellites from ${sourceName}` +
    (notes.length ? ` (${notes.join('; ')})` : '');
  searchInput.disabled = false;
  searchInput.placeholder = 'Search by name or NORAD ID...';
  renderSearchResults(searchInput.value);
  return true;
}

function setupDatabaseSearch() {
  const fileInput = document.getElementById('db-file-input');
  const loadBtn = document.getElementById('db-load-btn');
  const statusEl = document.getElementById('db-status');
  const searchInput = document.getElementById('db-search-input');
  if (!fileInput || !loadBtn || !statusEl || !searchInput) return;

  loadBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    statusEl.textContent = `Loading ${file.name}...`;
    try {
      loadSatelliteDatabase(await parseSatelliteDatabaseFile(file), file.name);
    } catch (e) {
      showDatabaseLoadError(`Could not read ${file.name}: ${e && e.message ? e.message : String(e)}`);
    } finally {
      fileInput.value = '';
    }
  });

  if (BUNDLED_SATELLITE_DATABASE) {
    loadSatelliteDatabase(BUNDLED_SATELLITE_DATABASE, 'bundled database');
    BUNDLED_SATELLITE_DATABASE = null;
  }

  let debounceTimer = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => renderSearchResults(searchInput.value), 80);
  });
}

function renderSearchResults(query) {
  const resultsEl = document.getElementById('db-search-results');
  if (!resultsEl) return;

  if (!loadedDatabase) {
    resultsEl.innerHTML = `<div class="db-hint">Load a satellite JSON or CSV database above to search it.</div>`;
    return;
  }
  const q = (query || '').trim().toUpperCase();
  if (!q) {
    resultsEl.innerHTML = `<div class="db-hint">Type a name (or NORAD ID) to search ${loadedDatabase.length.toLocaleString()} satellites.</div>`;
    return;
  }

  const qIsNumeric = /^\d+$/.test(q);
  const matches = loadedDatabase.filter(r =>
    String(r.name || '').toUpperCase().includes(q) ||
    (qIsNumeric && r._nid !== null && String(r._nid) === q)
  );
  if (matches.length === 0) {
    resultsEl.innerHTML = `<div class="db-hint">No matches for "${escapeHtml(query)}".</div>`;
    return;
  }

  const shown = matches.slice(0, SEARCH_RESULTS_LIMIT);
  const rows = shown.map(record => {
    const { key } = resolveDbRecord(record);
    const active = activeSatellites.has(key);
    return `<div class="db-result-row${active ? ' active' : ''}" data-key="${escapeHtml(key)}">` +
      `<span class="db-result-check">${active ? '\u2713' : ''}</span>` +
      `<span class="db-result-name">${escapeHtml(record.name)}</span></div>`;
  }).join('');
  const moreNote = matches.length > SEARCH_RESULTS_LIMIT
    ? `<div class="db-hint">Showing ${SEARCH_RESULTS_LIMIT} of ${matches.length.toLocaleString()} matches -- refine your search for more.</div>`
    : '';
  resultsEl.innerHTML = rows + moreNote;

  resultsEl.querySelectorAll('.db-result-row').forEach((rowEl, i) => {
    rowEl.addEventListener('click', () => toggleDbSatellite(shown[i]));
  });
}

function toggleDbSatellite(record) {
  const { key, entry } = resolveDbRecord(record);
  const statusEl = document.getElementById('db-status');
  const wasEmpty = activeSatellites.size === 0; // only auto-frame the very first pick
  if (activeSatellites.has(key)) {
    removeSatellite(key);
  } else {
    const added = addSatellite(key, entry);
    if (added === false) {
      if (statusEl) statusEl.textContent = `Limit reached (${MAX_ACTIVE_SATELLITES} satellites max) -- remove one to add another.`;
      return;
    }
    if (added === 'metadata') {
      if (statusEl) statusEl.textContent = `${record.name} is a DISCOS metadata record; load a matching TLE or pass state vectors to satellite_viewer() to plot it.`;
      return;
    }
    // Frame the first satellite so it isn't off-screen, but don't yank the
    // camera on any later selection -- keep whatever view the user has set.
    if (wasEmpty) reframeCamera();
  }
  updateInfoPanel();
  const searchInput = document.getElementById('db-search-input');
  if (searchInput) renderSearchResults(searchInput.value);
}


// ---------------------------------------------------------------------------
// Real catalog stars in TEME-of-date directions. They share one arbitrary sky
// radius because the catalog supplies angular directions, not display-scale
// distances. The ECI-to-scene remap matches every other inertial vector here.
// ---------------------------------------------------------------------------
function buildStarfield() {
  const vectors = STAR_CATALOG.v;
  const positions = new Float32Array(vectors.length);
  for (let i = 0; i < vectors.length; i += 3) {
    positions[i] = vectors[i] * STARFIELD_RADIUS_KM;
    positions[i + 1] = vectors[i + 2] * STARFIELD_RADIUS_KM;
    positions[i + 2] = -vectors[i + 1] * STARFIELD_RADIUS_KM;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(STAR_CATALOG.c), 3));
  geom.setAttribute('starSize', new THREE.BufferAttribute(new Float32Array(STAR_CATALOG.s), 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { pixelRatio: { value: renderer.getPixelRatio() } },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      uniform float pixelRatio;
      attribute float starSize;
      varying vec3 vColor;
      void main() {
        vColor = color;
        gl_PointSize = starSize * pixelRatio;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: `
      #include <logdepthbuf_pars_fragment>
      varying vec3 vColor;
      void main() {
        #include <logdepthbuf_fragment>
        float alpha = 1.0 - smoothstep(0.12, 0.5, length(gl_PointCoord - vec2(0.5)));
        if (alpha <= 0.01) discard;
        gl_FragColor = vec4(vColor * alpha, 1.0);
      }
    `,
    vertexColors: true, transparent: false, depthTest: false, depthWrite: false,
  });
  starfield = new THREE.Points(geom, mat);
  starfield.frustumCulled = false;
  starfield.renderOrder = -1000;
  scene.add(starfield);
}

// ---------------------------------------------------------------------------
// Earth -- custom day/night/specular shader on a smooth 128-segment sphere
// ---------------------------------------------------------------------------
function buildEarth() {
  const loader = new THREE.TextureLoader();
  const dayTex = loader.load(DAY_TEXTURE_DATAURI);
  const nightTex = loader.load(NIGHT_TEXTURE_DATAURI);
  const specTex = loader.load(SPECULAR_TEXTURE_DATAURI);
  [dayTex, nightTex, specTex].forEach(t => { t.colorSpace = THREE.SRGBColorSpace; });

  const geom = new THREE.SphereGeometry(R_EARTH_KM, 128, 128);
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      dayTexture: { value: dayTex },
      nightTexture: { value: nightTex },
      specularTexture: { value: specTex },
      sunDirection: { value: sunDirection },
    },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vPositionW;
      void main() {
        vUv = uv;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vPositionW = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: `
      #include <logdepthbuf_pars_fragment>
      uniform sampler2D dayTexture;
      uniform sampler2D nightTexture;
      uniform sampler2D specularTexture;
      uniform vec3 sunDirection;
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vPositionW;
      void main() {
        #include <logdepthbuf_fragment>
        vec3 normal = normalize(vNormalW);
        float NdotL = dot(normal, normalize(sunDirection));
        // Keep the dark side visually stable: a narrow physical terminator,
        // not a broad grey blend that drags day texture onto night-side pixels.
        float dayNightMix = smoothstep(-0.015, 0.075, NdotL);

        vec3 dayColor = texture2D(dayTexture, vUv).rgb;
        vec3 nightColor = texture2D(nightTexture, vUv).rgb;
        float specMask = texture2D(specularTexture, vUv).r;

        vec3 viewDir = normalize(cameraPosition - vPositionW);
        vec3 halfDir = normalize(normalize(sunDirection) + viewDir);
        float specAngle = max(dot(normal, halfDir), 0.0);
        float specStrength = pow(specAngle, 32.0) * specMask * smoothstep(0.05, 0.25, NdotL);

        // Treat the night texture as emissive city lights plus a tiny blue-black
        // surface floor. This avoids the blotchy, shadow-like dark-side artifact
        // caused by amplifying the entire night raster as if it were albedo.
        vec3 nightEmission = max(nightColor - vec3(0.020), vec3(0.0)) * 0.95;
        vec3 nightBase = vec3(0.008, 0.011, 0.022) + nightEmission;
        vec3 base = mix(nightBase, dayColor, dayNightMix);

        // A thin warm terminator only on the sunlit edge; it should not bleed
        // onto the fully dark hemisphere.
        float termBand = (1.0 - smoothstep(0.0, 0.09, abs(NdotL))) * smoothstep(0.0, 0.055, NdotL);
        vec3 goldenHour = vec3(1.0, 0.55, 0.30) * termBand * 0.07;
        base += goldenHour;

        vec3 color = clamp(base + specStrength * vec3(1.0, 0.98, 0.9) * 0.55, 0.0, 1.0);
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });
  earthMesh = new THREE.Mesh(geom, mat);
  scene.add(earthMesh);
}

// ---------------------------------------------------------------------------
// Clouds -- separate translucent shell just above the surface, lit by the
// same directional light as everything else so it shares one consistent
// sun direction with the Earth shader.
// ---------------------------------------------------------------------------
function buildClouds() {
  const loader = new THREE.TextureLoader();
  const cloudTex = loader.load(CLOUDS_TEXTURE_DATAURI);
  // Mipmaps are safe here now: the texture itself was reprocessed so every
  // pixel's RGB sits near true cloud-white, with only the alpha channel
  // varying to encode cloud shape. Previously, "transparent" pixels held a
  // mismatched mid-gray RGB baked in; mipmapping blends RGB blindly across
  // neighboring pixels regardless of alpha, so that mismatch bled into the
  // white clouds as dark patchy fringing at certain zoom levels. Disabling
  // mipmaps entirely (the first fix attempted) removed that, but introduced
  // visible shimmer/aliasing when the cloud layer was minified at middle
  // zoom levels -- fixing the source texture avoids both problems at once.
  const geom = new THREE.SphereGeometry(R_EARTH_KM * 1.006, 96, 96);
  const mat = new THREE.MeshLambertMaterial({
    map: cloudTex, transparent: true, depthWrite: false, opacity: 0.85,
  });
  cloudMesh = new THREE.Mesh(geom, mat);
  cloudsGroup = new THREE.Group();
  cloudsGroup.add(cloudMesh);
  scene.add(cloudsGroup);
}

// ---------------------------------------------------------------------------
// Atmosphere -- Fresnel rim glow, additive-blended, brighter on the day side
// ---------------------------------------------------------------------------
function buildAtmosphere() {
  const geom = new THREE.SphereGeometry(R_EARTH_KM * 1.025, 96, 96);
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      glowColor: { value: new THREE.Color(0x4fa8ff) },
      sunDirection: { value: sunDirection },
    },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec3 vNormalW;
      varying vec3 vPositionW;
      void main() {
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vPositionW = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: `
      #include <logdepthbuf_pars_fragment>
      uniform vec3 glowColor;
      uniform vec3 sunDirection;
      varying vec3 vNormalW;
      varying vec3 vPositionW;
      void main() {
        #include <logdepthbuf_fragment>
        vec3 normal = normalize(vNormalW);
        vec3 viewDir = normalize(cameraPosition - vPositionW);
        float rim = 1.0 - max(dot(normal, viewDir), 0.0);
        float intensity = pow(rim, 2.5);
        float sunFactor = clamp(dot(normal, sunDirection) * 0.6 + 0.5, 0.15, 1.0);
        gl_FragColor = vec4(glowColor * intensity * sunFactor, intensity * 0.9);
      }
    `,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  atmosphereMesh = new THREE.Mesh(geom, mat);
  scene.add(atmosphereMesh);
}

// ---------------------------------------------------------------------------
// Axis convention fix: satellite.js's SGP4 output uses the standard
// astrodynamics convention where Z is the pole
// (perpendicular to the equatorial plane) and X/Y span the equatorial
// plane. This scene's Earth mesh, however, is a plain THREE.SphereGeometry
// with no compensating rotation -- Three.js's default sphere convention has
// Y as the pole. Without remapping, every orbit's inclination ends up
// applied relative to the wrong axis: verified concretely by checking an
// inclination=0 ("equatorial") orbit, which should stay in a single fixed
// plane for every true anomaly, but without this fix sweeps its Y
// coordinate across the full orbital radius instead of staying at Y=0.
// This is a pure right-handed rotation (verified: no mirroring, handedness
// preserved), applied once here so every caller gets correctly-oriented
// output automatically rather than needing to remember to apply it.
// ---------------------------------------------------------------------------
function eciToScene(v) {
  return new THREE.Vector3(v.x, v.z, -v.y);
}

// ---------------------------------------------------------------------------
// Sun direction, computed from the actual current date instead of a fixed
// hardcoded vector. This is the standard "low precision" solar position
// formula from the Astronomical Almanac (page C24) -- accurate to about
// 0.01 degree over 1950-2050, cross-checked against several independent
// published sources (NASA/LASP code, an arXiv paper, a patent filing, all
// citing the identical coefficients) before using it here. Result is a unit
// vector from Earth toward the Sun, in the same Z-up astrodynamics
// convention as everything else in this file -- eciToScene() converts it to
// scene coordinates, same as any other ECI vector.
function computeSunDirectionEci(date) {
  const JD = date.getTime() / 86400000 + 2440587.5; // Julian Date from Unix ms
  const n = JD - 2451545.0; // days since J2000.0
  const L = ((280.460 + 0.9856474 * n) % 360 + 360) % 360; // mean longitude, degrees
  const g = ((357.528 + 0.9856003 * n) % 360 + 360) % 360; // mean anomaly, degrees
  const gRad = g * Math.PI / 180;
  const lambdaDeg = L + 1.915 * Math.sin(gRad) + 0.020 * Math.sin(2 * gRad); // ecliptic longitude
  const lambda = lambdaDeg * Math.PI / 180;
  const epsilon = (23.439 - 0.0000004 * n) * Math.PI / 180; // obliquity of the ecliptic
  return new THREE.Vector3(
    Math.cos(lambda),
    Math.cos(epsilon) * Math.sin(lambda),
    Math.sin(epsilon) * Math.sin(lambda)
  ).normalize();
}

function computeMoonDirectionEci(date) {
  const JD = date.getTime() / 86400000 + 2440587.5;
  const n = JD - 2451545.0;
  const L = ((218.316 + 13.176396 * n) % 360 + 360) % 360;
  const Mm = ((134.963 + 13.064993 * n) % 360 + 360) % 360;
  const F = ((93.272 + 13.229350 * n) % 360 + 360) % 360;
  const lambda = (L + 6.289 * Math.sin(Mm * Math.PI / 180)) * Math.PI / 180;
  const beta = (5.128 * Math.sin(F * Math.PI / 180)) * Math.PI / 180;
  const epsilon = (23.439 - 0.0000004 * n) * Math.PI / 180;
  const cb = Math.cos(beta);
  const xEcl = cb * Math.cos(lambda);
  const yEcl = cb * Math.sin(lambda);
  const zEcl = Math.sin(beta);
  return new THREE.Vector3(
    xEcl,
    Math.cos(epsilon) * yEcl - Math.sin(epsilon) * zEcl,
    Math.sin(epsilon) * yEcl + Math.cos(epsilon) * zEcl
  ).normalize();
}

function buildCelestialMarkers() {
  const sunMat = new THREE.MeshBasicMaterial({ color: 0xffc247, transparent: true, opacity: 0.98, depthTest: true, depthWrite: false });
  sunMesh = new THREE.Mesh(new THREE.SphereGeometry(SUN_RENDER_RADIUS_KM, 48, 32), sunMat);
  sunMesh.renderOrder = 20;
  scene.add(sunMesh);

  const glowMat = new THREE.MeshBasicMaterial({
    color: 0xffd36a, transparent: true, opacity: 0.16,
    depthWrite: false, depthTest: true, blending: THREE.AdditiveBlending,
  });
  sunGlowMesh = new THREE.Mesh(new THREE.SphereGeometry(SUN_GLOW_RADIUS_KM, 32, 24), glowMat);
  sunGlowMesh.renderOrder = 19;
  scene.add(sunGlowMesh);

  const moonMat = new THREE.ShaderMaterial({
    uniforms: {
      sunDirection: { value: sunDirection },
      dayColor: { value: new THREE.Color(0xbfc3cc) },
      nightColor: { value: new THREE.Color(0x222833) },
    },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec3 vNormalW;
      void main() {
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: `
      #include <logdepthbuf_pars_fragment>
      uniform vec3 sunDirection;
      uniform vec3 dayColor;
      uniform vec3 nightColor;
      varying vec3 vNormalW;
      void main() {
        #include <logdepthbuf_fragment>
        vec3 normal = normalize(vNormalW);
        float NdotL = dot(normal, normalize(sunDirection));
        float phase = smoothstep(-0.04, 0.08, NdotL);
        float diffuse = 0.10 + 0.90 * pow(max(NdotL, 0.0), 0.78);
        vec3 color = mix(nightColor, dayColor * diffuse, phase);
        gl_FragColor = vec4(color, 1.0);
      }
    `,
    depthTest: true,
    depthWrite: false,
  });
  moonMesh = new THREE.Mesh(new THREE.SphereGeometry(MOON_RENDER_RADIUS_KM, 48, 32), moonMat);
  moonMesh.renderOrder = 20;
  scene.add(moonMesh);
}

function updateCelestialMarkers(date) {
  if (sunMesh) sunMesh.position.copy(sunDirection).multiplyScalar(SUN_RENDER_DISTANCE_KM);
  if (sunGlowMesh) sunGlowMesh.position.copy(sunDirection).multiplyScalar(SUN_RENDER_DISTANCE_KM);
  if (moonMesh) {
    const moonDirection = eciToScene(computeMoonDirectionEci(date));
    moonMesh.position.copy(moonDirection).multiplyScalar(MOON_RENDER_DISTANCE_KM);
    if (moonMesh.material.uniforms && moonMesh.material.uniforms.sunDirection) {
      // Use true near-parallel solar illumination for lunar phase. The
      // rendered Sun is physically distant, so marker-to-marker geometry is no
      // longer needed and would make the phase sensitive to display scaling.
      moonMesh.material.uniforms.sunDirection.value.copy(sunDirection);
    }
  }
}

// Earth's rotation, from the actual current date instead of a static mesh.
// Uses satellite.js's own gstime() (Greenwich Mean Sidereal Time) rather
// than hand-rolling the formula -- it's already a trusted dependency this
// whole file relies on for SGP4 itself. GMST is, by definition, a rotation
// about the true pole (Z in the Z-up astrodynamics convention), which
// eciToScene()'s remap sends to this scene's Y-axis -- so this is just
// earthMesh.rotation.y = gstime(now), no separate calibration transform
// needed. Verified empirically (see the day-texture pixel-sampling check
// done before implementing this) that the day texture's own UV mapping
// already puts real longitude 0 (Greenwich) at this scene's +X axis at
// rotation.y=0, matching where GMST=0 says it should be.
function updateEarthRotation(date) {
  const gmst = satellite.gstime(date);
  earthMesh.rotation.y = gmst;
  cloudsGroup.rotation.y = gmst + cloudDriftAccumulator;
}

function orbitFrame(position, velocity) {
  const r = position.length();
  let zHat = r > 1e-9 ? position.clone().multiplyScalar(-1 / r) : new THREE.Vector3(0, 0, 1);
  const vLen = velocity.length();
  let xHat = new THREE.Vector3(0, 0, 0), xNorm = 0;
  if (vLen > 1e-9) {
    const vHat = velocity.clone().normalize();
    xHat = vHat.clone().sub(zHat.clone().multiplyScalar(vHat.dot(zHat)));
    xNorm = xHat.length();
  }
  if (xNorm < 1e-6) {
    let seed = new THREE.Vector3(1, 0, 0);
    if (Math.abs(seed.dot(zHat)) > 0.9) seed = new THREE.Vector3(0, 1, 0);
    xHat = new THREE.Vector3().crossVectors(zHat, seed).normalize();
  } else {
    xHat.divideScalar(xNorm);
  }
  const yHat = new THREE.Vector3().crossVectors(zHat, xHat);
  return { xHat, yHat, zHat };
}

function applyOrbitOrientation(obj3d, position, velocity) {
  const { xHat, yHat, zHat } = orbitFrame(position, velocity);
  const m = new THREE.Matrix4().makeBasis(xHat, yHat, zHat);
  obj3d.quaternion.setFromRotationMatrix(m);
  obj3d.position.copy(position);
}

// SSATK quaternions are body-to-TEME in [w, x, y, z] order. Convert
// the three body axes through the same ECI-to-scene rotation used for position
// and velocity, then let Three.js build the scene-frame orientation.
function applyQuaternionOrientation(obj3d, position, quaternion) {
  const qEci = new THREE.Quaternion(
    quaternion.x, quaternion.y, quaternion.z, quaternion.w
  ).normalize();
  const basis = new THREE.Matrix4().makeBasis(
    eciToScene(new THREE.Vector3(1, 0, 0).applyQuaternion(qEci)),
    eciToScene(new THREE.Vector3(0, 1, 0).applyQuaternion(qEci)),
    eciToScene(new THREE.Vector3(0, 0, 1).applyQuaternion(qEci)),
  );
  obj3d.quaternion.setFromRotationMatrix(basis);
  obj3d.position.copy(position);
}

// ---------------------------------------------------------------------------
// Realistic per-type satellite models -- built from primitives, but each
// with a genuinely distinct, recognizable silhouette rather than one
// generic "box + 2 panels" stand-in for everything. All built in the same
// local frame as orbitFrame(): local X = ram/velocity direction, local Y =
// orbit-normal-ish, local Z = nadir (toward Earth).
// ---------------------------------------------------------------------------

function buildISSModel(size) {
  const group = new THREE.Group();
  const trussMat = new THREE.MeshPhongMaterial({ color: 0xcfd4da, shininess: 25, specular: 0x222222 });
  const panelMat = buildPanelMaterial(0x0d1b3a, 35, 0x334466, 5, 2); // paddle: long in X/Z, thin in Y
  const moduleMat = new THREE.MeshPhongMaterial({ color: 0xe0dbc8, shininess: 15, specular: 0x111111 });
  const radiatorMat = new THREE.MeshPhongMaterial({ color: 0xf2f4f6, shininess: 70, specular: 0xffffff });

  // Long main truss along local Y -- the real ISS's integrated truss structure
  const truss = new THREE.Mesh(new THREE.BoxGeometry(size * 0.045, size * 1.0, size * 0.045), trussMat);
  group.add(truss);

  // 4 pairs of solar array paddles (8 total, matching the real ISS) spaced
  // along the truss, each pair extending outward in +/-X
  const paddleGeom = new THREE.BoxGeometry(size * 0.46, size * 0.018, size * 0.20);
  [-0.42, -0.20, 0.20, 0.42].forEach(yFrac => {
    [1, -1].forEach(side => {
      const paddle = new THREE.Mesh(paddleGeom, panelMat);
      paddle.position.set(side * size * 0.26, yFrac * size, 0);
      group.add(paddle);
    });
  });

  // White radiator panels near the center -- a real, distinctive ISS
  // feature (thermal control radiators) that reads very differently from
  // the dark solar arrays and was missing before.
  const radiatorGeom = new THREE.BoxGeometry(size * 0.015, size * 0.30, size * 0.13);
  [1, -1].forEach(side => {
    const rad = new THREE.Mesh(radiatorGeom, radiatorMat);
    rad.position.set(side * size * 0.16, 0, size * 0.10);
    group.add(rad);
  });

  // Pressurized module cluster near the center (Unity/Destiny/Zvezda-style)
  const mod1 = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.05, size * 0.05, size * 0.34, 12), moduleMat);
  mod1.rotation.z = Math.PI / 2;
  mod1.position.set(0, 0, size * 0.07);
  group.add(mod1);
  const mod2 = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.042, size * 0.042, size * 0.22, 12), moduleMat);
  mod2.rotation.x = Math.PI / 2;
  mod2.position.set(size * 0.02, size * 0.09, size * 0.07);
  group.add(mod2);

  // Small equipment/handrail greebles on the module cluster -- real ISS
  // modules are covered in this kind of detail, and a bare cylinder reads
  // as too clean/simple by comparison.
  addGreebles(group, moduleMat, size * 0.30, size * 0.10, size * 0.10, 6, 'iss-greebles');

  // Status/beacon lights, a small "this is an active machine" detail
  addIndicatorLight(group, size, new THREE.Vector3(size * 0.06, 0, size * 0.12), 0xff3b30);
  addIndicatorLight(group, size, new THREE.Vector3(-size * 0.06, 0, size * 0.12), 0x30d158);

  return group;
}

function buildHubbleModel(size) {
  const group = new THREE.Group();
  const tubeMat = new THREE.MeshPhongMaterial({ color: 0xcfc9a8, shininess: 45, specular: 0x555540 });
  const panelMat = buildPanelMaterial(0x0e1c3d, 30, 0x334466, 3, 5);

  // Main telescope tube, long along local X (Hubble's iconic cylindrical body)
  const tube = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.09, size * 0.09, size * 0.85, 24), tubeMat);
  tube.rotation.z = Math.PI / 2;
  group.add(tube);

  // 2 solar panels extending from local Y+/-, mounted mid-body
  const panelGeom = new THREE.BoxGeometry(size * 0.30, size * 0.02, size * 0.46);
  const panelA = new THREE.Mesh(panelGeom, panelMat);
  panelA.position.set(0, size * 0.34, 0);
  group.add(panelA);
  const panelB = new THREE.Mesh(panelGeom, panelMat);
  panelB.position.set(0, -size * 0.34, 0);
  group.add(panelB);

  // small aft antenna boom
  const antenna = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.018, size * 0.018, size * 0.16, 8), tubeMat);
  antenna.rotation.z = Math.PI / 2;
  antenna.position.set(-size * 0.3, 0, size * 0.1);
  group.add(antenna);

  return group;
}

function buildStarlinkModel(size) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: 0x1a1c22, shininess: 55, specular: 0x666666 });
  const panelMat = buildPanelMaterial(0x0c1830, 25, 0x334466, 6, 3);

  // Very flat, wide "flat-packed" bus -- Starlink's distinctive silhouette
  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.5, size * 0.34, size * 0.035), busMat);
  group.add(bus);

  // ONE solar panel wing from a single side only (not a mirrored pair --
  // this is the real, recognizable difference from most other satellites)
  const panel = new THREE.Mesh(new THREE.BoxGeometry(size * 0.02, size * 0.62, size * 0.34), panelMat);
  panel.position.set(0, size * 0.48, 0);
  group.add(panel);

  return group;
}

function buildGPSModel(size) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: 0xb8bcc4, shininess: 20, specular: 0x222222 });
  const panelMat = buildPanelMaterial(0x15294f, 18, 0x223344, 4, 2);

  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.22, size * 0.20, size * 0.22), busMat);
  group.add(bus);

  const panelGeom = new THREE.BoxGeometry(size * 0.022, size * 0.34, size * 0.16);
  const panelOffset = size * 0.10 + size * 0.17;
  const panelA = new THREE.Mesh(panelGeom, panelMat);
  panelA.position.set(0, panelOffset, 0);
  group.add(panelA);
  const panelB = new THREE.Mesh(panelGeom, panelMat);
  panelB.position.set(0, -panelOffset, 0);
  group.add(panelB);

  // small dish antenna facing nadir
  const dish = new THREE.Mesh(new THREE.ConeGeometry(size * 0.07, size * 0.05, 16), busMat);
  dish.rotation.x = Math.PI;
  dish.position.set(0, 0, size * 0.14);
  group.add(dish);

  return group;
}

function buildNavSatModel(size, busColor, panelColor) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: busColor, shininess: 16, specular: 0x141414 });
  const panelMat = buildPanelMaterial(panelColor, 20, 0x223344, 5, 2);

  // Hexagonal-prism bus, drum-oriented along nadir/zenith -- distinguishes
  // this family's actual design from GPS's plain box
  const bus = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.15, size * 0.15, size * 0.20, 6), busMat);
  bus.rotation.x = Math.PI / 2;
  group.add(bus);

  const panelGeom = new THREE.BoxGeometry(size * 0.02, size * 0.42, size * 0.20);
  const panelOffset = size * 0.15 + size * 0.21;
  const panelA = new THREE.Mesh(panelGeom, panelMat);
  panelA.position.set(0, panelOffset, 0);
  group.add(panelA);
  const panelB = new THREE.Mesh(panelGeom, panelMat);
  panelB.position.set(0, -panelOffset, 0);
  group.add(panelB);

  // small nadir antenna horn (navigation signal antenna array)
  const horn = new THREE.Mesh(new THREE.ConeGeometry(size * 0.085, size * 0.07, 12), busMat);
  horn.rotation.x = Math.PI;
  horn.position.set(0, 0, size * 0.135);
  group.add(horn);

  return group;
}

function buildGalileoModel(size) {
  return buildNavSatModel(size, 0xd8dce2, 0x122540); // silver bus, dark navy panels
}

function buildGlonassModel(size) {
  return buildNavSatModel(size, 0xc9c2b4, 0x5a1414); // warmer bus tone, dark red panels -- distinct livery, same real silhouette family
}

function buildBeidouModel(size) {
  return buildNavSatModel(size, 0xcdd6c8, 0x143a1e); // pale bus, dark green panels
}


function buildGOESModel(size) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: 0xe8e4d8, shininess: 15, specular: 0x131313 });
  const panelMat = buildPanelMaterial(0x0c1830, 22, 0x223344, 8, 2); // GOES' single long wing
  const boomMat = new THREE.MeshPhongMaterial({ color: 0x999999, shininess: 40, specular: 0x444444 });

  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.20, size * 0.24, size * 0.20), busMat);
  group.add(bus);

  // GOES uses a single sun-tracking solar array on a yoke, not a mirrored
  // pair -- one long wing is the realistic silhouette here
  const panel = new THREE.Mesh(new THREE.BoxGeometry(size * 0.022, size * 0.85, size * 0.24), panelMat);
  panel.position.set(0, size * 0.12 + size * 0.43, 0);
  group.add(panel);

  // long magnetometer boom
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.008, size * 0.008, size * 0.7, 6), boomMat);
  boom.rotation.z = Math.PI / 2;
  boom.position.set(-size * 0.45, -size * 0.10, 0);
  group.add(boom);

  // Earth-facing instrument sunshade/aperture
  const shade = new THREE.Mesh(new THREE.ConeGeometry(size * 0.10, size * 0.07, 16), busMat);
  shade.rotation.x = Math.PI;
  shade.position.set(0, 0, size * 0.155);
  group.add(shade);

  return group;
}

function buildLandsatModel(size) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: 0xc7cbd1, shininess: 15, specular: 0x131313 });
  const panelMat = buildPanelMaterial(0x102040, 25, 0x334466, 6, 3);
  const sensorMat = new THREE.MeshPhongMaterial({ color: 0x14161c, shininess: 60, specular: 0x555555 });

  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.26, size * 0.30, size * 0.24), busMat);
  group.add(bus);

  // single deployable solar wing (Landsat 9's real design, not a pair)
  const panel = new THREE.Mesh(new THREE.BoxGeometry(size * 0.02, size * 0.62, size * 0.28), panelMat);
  panel.position.set(0, size * 0.15 + size * 0.31, 0);
  group.add(panel);

  // nadir-facing sensor barrel (OLI-2 / TIRS-2 imaging instruments)
  const sensor = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.08, size * 0.09, size * 0.14, 16), sensorMat);
  sensor.rotation.x = Math.PI / 2;
  sensor.position.set(0, 0, size * 0.19);
  group.add(sensor);

  return group;
}

function buildTiangongModel(size) {
  const group = new THREE.Group();
  const moduleMat = new THREE.MeshPhongMaterial({ color: 0xd4cfa0, shininess: 20, specular: 0x222211 });
  const panelMat = buildPanelMaterial(0x0d1b3a, 30, 0x334466, 2, 5); // lab-module panel proportions; core panel gets its own material below

  // Tianhe core module, along local X
  const core = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.11, size * 0.11, size * 0.50, 16), moduleMat);
  core.rotation.z = Math.PI / 2;
  group.add(core);

  // Core's own solar arrays, near the aft end
  const corePanelMat = buildPanelMaterial(0x0d1b3a, 30, 0x334466, 2, 6); // core panel proportions differ from the lab panels above
  const corePanelGeom = new THREE.BoxGeometry(size * 0.02, size * 0.10, size * 0.32);
  [1, -1].forEach(side => {
    const p = new THREE.Mesh(corePanelGeom, corePanelMat);
    p.position.set(-size * 0.30, side * size * 0.16, 0);
    group.add(p);
  });

  // Wentian & Mengtian lab modules extending to the sides, each with their
  // own large arrays at the outer tip -- the real T/cross shape that
  // distinguishes Tiangong's compact 3-module layout from ISS's one long truss
  [1, -1].forEach(side => {
    const lab = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.10, size * 0.10, size * 0.40, 16), moduleMat);
    lab.rotation.x = Math.PI / 2;
    lab.position.set(size * 0.05, side * size * 0.32, 0);
    group.add(lab);

    const labPanel = new THREE.Mesh(new THREE.BoxGeometry(size * 0.02, size * 0.46, size * 0.20), panelMat);
    labPanel.position.set(size * 0.05, side * (size * 0.32 + size * 0.35), 0);
    group.add(labPanel);
  });

  return group;
}

// ---------------------------------------------------------------------------
// JWST -- bespoke, not an archetype. Its silhouette (segmented hexagonal
// primary mirror + huge kite-shaped sunshield) is too iconic and too
// different from every other satellite here to fold into a generic
// "tube + panels" shape without doing it a disservice.
// ---------------------------------------------------------------------------
function buildJWSTModel(size) {
  const group = new THREE.Group();
  const mirrorMat = new THREE.MeshPhongMaterial({ color: 0xd4af37, shininess: 70, specular: 0x887733 }); // gold-coated beryllium segments
  const shieldMat = new THREE.MeshPhongMaterial({ color: 0xc9c2a8, shininess: 8, specular: 0x222222, side: THREE.DoubleSide });
  const structMat = new THREE.MeshPhongMaterial({ color: 0x888888, shininess: 30, specular: 0x333333 });

  // Segmented hexagonal primary mirror: 7 hexagons (center + 6 around),
  // facing local -X (away from the sunshield, toward deep space)
  const hexRadius = size * 0.075;
  const hexGeom = new THREE.CylinderGeometry(hexRadius, hexRadius, size * 0.012, 6);
  const mirrorCenterX = size * 0.20;
  const centerHex = new THREE.Mesh(hexGeom, mirrorMat);
  centerHex.rotation.z = Math.PI / 2;
  centerHex.position.set(mirrorCenterX, 0, 0);
  group.add(centerHex);
  for (let i = 0; i < 6; i++) {
    const ang = (i / 6) * Math.PI * 2;
    const hex = new THREE.Mesh(hexGeom, mirrorMat);
    hex.rotation.z = Math.PI / 2;
    hex.position.set(mirrorCenterX, Math.cos(ang) * hexRadius * 1.78, Math.sin(ang) * hexRadius * 1.78);
    group.add(hex);
  }

  // Secondary mirror boom, extending forward from the primary
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.006, size * 0.006, size * 0.34, 6), structMat);
  boom.rotation.z = Math.PI / 2;
  boom.position.set(mirrorCenterX + size * 0.20, 0, 0);
  group.add(boom);
  const secondary = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.02, size * 0.02, size * 0.01, 12), mirrorMat);
  secondary.rotation.z = Math.PI / 2;
  secondary.position.set(mirrorCenterX + size * 0.37, 0, 0);
  group.add(secondary);

  // Sunshield: large flat diamond/kite shape on the opposite side, always
  // facing the Sun/Earth/Moon to keep the telescope in permanent shadow
  const shieldShape = new THREE.Shape();
  shieldShape.moveTo(0, size * 0.30);
  shieldShape.lineTo(size * 0.46, size * 0.10);
  shieldShape.lineTo(size * 0.40, -size * 0.20);
  shieldShape.lineTo(0, -size * 0.34);
  shieldShape.lineTo(-size * 0.40, -size * 0.20);
  shieldShape.lineTo(-size * 0.46, size * 0.10);
  shieldShape.lineTo(0, size * 0.30);
  const shieldGeom = new THREE.ShapeGeometry(shieldShape);
  const shield = new THREE.Mesh(shieldGeom, shieldMat);
  shield.rotation.y = Math.PI / 2; // lay flat in the YZ plane
  shield.position.set(-size * 0.10, 0, 0);
  group.add(shield);

  // Bus (spacecraft body) sits at the sunshield side, between shield and mirror
  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.10, size * 0.14, size * 0.06), structMat);
  bus.position.set(-size * 0.06, 0, size * 0.05);
  group.add(bus);

  return group;
}

// ---------------------------------------------------------------------------
// Compact science observatory archetype -- Chandra, Fermi, Swift, TESS.
// Smaller box/cylinder bus + 2 panels, distinct from Hubble's single big
// tube. TESS gets 4 small camera barrels added, matching its actual
// 4-wide-field-camera design -- the one detail worth breaking from the
// shared archetype for, since it's TESS's most recognizable feature.
// ---------------------------------------------------------------------------
function buildObservatoryModel(size, variant) {
  const group = new THREE.Group();
  const busMat = new THREE.MeshPhongMaterial({ color: 0xb7bcc4, shininess: 25, specular: 0x222222 });
  const panelMat = buildPanelMaterial(0x17284a, 20, 0x223344, 5, 2);
  const tubeMat = new THREE.MeshPhongMaterial({ color: 0x1c1f26, shininess: 45, specular: 0x444444 });
  const mliMat = buildMLIMaterial(MLI_GOLD, 70);

  if (variant === 'chandra') {
    // Chandra's real distinguishing feature is an elongated, tube-like
    // spacecraft body (the X-ray mirror assembly is long, not a compact
    // box like the others) -- deliberately breaks from the shared
    // box+2-panel base for this one.
    const body = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.09, size * 0.09, size * 0.55, 20), mliMat);
    body.rotation.z = Math.PI / 2;
    group.add(body);
    const aperture = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.10, size * 0.095, size * 0.05, 20), tubeMat);
    aperture.rotation.z = Math.PI / 2;
    aperture.position.set(size * 0.29, 0, 0);
    group.add(aperture);
    const panelGeom = new THREE.BoxGeometry(size * 0.02, size * 0.32, size * 0.16);
    [1, -1].forEach(s => {
      const p = new THREE.Mesh(panelGeom, panelMat);
      p.position.set(-size * 0.05, s * (size * 0.10 + size * 0.16), 0);
      group.add(p);
    });
    return group;
  }

  // Shared box+2-panel bus for Fermi/Swift/TESS/generic -- these three
  // really do share a similar compact-bus silhouette in reality; the
  // instrument section on top is where they actually differ.
  const bus = new THREE.Mesh(new THREE.BoxGeometry(size * 0.24, size * 0.22, size * 0.24), busMat);
  group.add(bus);
  const panelGeom = new THREE.BoxGeometry(size * 0.02, size * 0.36, size * 0.18);
  const panelOffset = size * 0.11 + size * 0.19;
  [1, -1].forEach(s => {
    const p = new THREE.Mesh(panelGeom, panelMat);
    p.position.set(0, s * panelOffset, 0);
    group.add(p);
  });

  if (variant === 'fermi') {
    // Fermi's LAT: a large flat wide instrument on top, not a cylindrical
    // aperture -- plus small GBM detector domes ringing the bus, matching
    // its real distinctive silhouette.
    const lat = new THREE.Mesh(new THREE.BoxGeometry(size * 0.22, size * 0.10, size * 0.22), tubeMat);
    lat.position.set(0, 0, size * 0.17);
    group.add(lat);
    for (let i = 0; i < 6; i++) {
      const ang = (i / 6) * Math.PI * 2;
      const dome = new THREE.Mesh(new THREE.SphereGeometry(size * 0.02, 8, 8), tubeMat);
      dome.position.set(Math.cos(ang) * size * 0.13, Math.sin(ang) * size * 0.13, size * 0.12);
      group.add(dome);
    }
  } else if (variant === 'swift') {
    // Swift's 3 real, distinctly-sized co-aligned instruments (UVOT/XRT/BAT)
    // instead of one uniform aperture.
    const specs = [[size * 0.05, size * 0.14], [size * 0.035, size * 0.10], [size * 0.06, size * 0.08]];
    const offsets = [-size * 0.06, size * 0.03, size * 0.09];
    specs.forEach(([r, l], i) => {
      const tube = new THREE.Mesh(new THREE.CylinderGeometry(r, r, l, 16), tubeMat);
      tube.rotation.x = Math.PI / 2;
      tube.position.set(offsets[i], 0, size * 0.16 + l * 0.15);
      group.add(tube);
    });
  } else if (variant === 'tess') {
    // TESS's 4 wide-field camera barrels, arranged in a 2x2 cluster facing nadir
    const tubeGeom = new THREE.CylinderGeometry(size * 0.045, size * 0.045, size * 0.16, 16);
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sy]) => {
      const tube = new THREE.Mesh(tubeGeom, tubeMat);
      tube.rotation.x = Math.PI / 2;
      tube.position.set(sx * size * 0.055, sy * size * 0.055, size * 0.19);
      group.add(tube);
    });
  } else {
    const instrument = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.09, size * 0.09, size * 0.16, 20), tubeMat);
    instrument.rotation.x = Math.PI / 2;
    instrument.position.set(0, 0, size * 0.19);
    group.add(instrument);
  }

  return group;
}


function buildModelForType(type, size, seedStr) {
  switch (type) {
    case 'ISS': return buildISSModel(size);
    case 'HUBBLE': return buildHubbleModel(size);
    case 'STARLINK': return buildStarlinkModel(size);
    case 'GPS': return buildGPSModel(size);
    case 'GALILEO': return buildGalileoModel(size);
    case 'GLONASS': return buildGlonassModel(size);
    case 'BEIDOU': return buildBeidouModel(size);
    case 'WEATHER_GEO': return buildGOESModel(size);
    case 'EARTH_OBS': return buildLandsatModel(size);
    case 'WEATHER_LEO': return buildLandsatModel(size); // same real-world silhouette family as EARTH_OBS (boxy sun-sync bus + single wing)
    case 'CSS': return buildTiangongModel(size);
    case 'JWST': return buildJWSTModel(size);
    case 'TESS': return buildObservatoryModel(size, 'tess');
    case 'OBSERVATORY': {
      // 'OBSERVATORY' only ever comes from the chandra/fermi/swift curated
      // entries (see SATELLITE_CATALOG) -- their real names are stable, so
      // matching on the seedStr (their name) to pick a variant is safe here.
      const upper = (seedStr || '').toUpperCase();
      if (upper.includes('CHANDRA')) return buildObservatoryModel(size, 'chandra');
      if (upper.includes('FERMI')) return buildObservatoryModel(size, 'fermi');
      if (upper.includes('SWIFT')) return buildObservatoryModel(size, 'swift');
      return buildObservatoryModel(size, 'generic');
    }
    case 'DEBRIS': return buildDebrisModel(size, seedStr);
    case 'ROCKET_BODY': return buildRocketBodyModel(size, seedStr);
    case 'GENERIC_PAYLOAD': return buildGenericPayloadModel(size, seedStr);
    default: return buildGPSModel(size);
  }
}

// ---------------------------------------------------------------------------
// Seeded randomness -- deterministic per-satellite variation (same NORAD ID
// always renders the same way across reloads, but different satellites
// look like different satellites, not clones of one template) plus a
// shared MLI gold/silver thermal-blanket material, the recognizable foil
// look most real spacecraft buses have that none of these models used
// before.
// ---------------------------------------------------------------------------
function hashStringToSeed(str) {
  let h = 2166136261;
  const s = String(str || 'default');
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function seededRandom(seedStr) {
  let seed = hashStringToSeed(seedStr);
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MLI_GOLD = 0xc9a227;
const MLI_SILVER = 0xc7c9cc;
function buildMLIMaterial(colorHex, shininess) {
  return new THREE.MeshPhongMaterial({ color: colorHex, shininess: shininess || 65, specular: 0xfff2d0 });
}

// ---------------------------------------------------------------------------
// Solar panel material -- every panel across every model was previously a
// flat solid-color box. Real solar panels have an immediately-recognizable
// cell grid, and panels appear on almost every satellite here, so this is
// the single highest-value visual upgrade available: fix it once, in one
// place, and it improves the whole scene at once.
//
// The texture encodes the grid as a BRIGHTNESS mask (light cells, darker
// gridlines) rather than baking in a specific color, so the same texture
// works correctly multiplied against any panel's existing tint (navy,
// dark red, dark green, etc.) without needing a separate texture per
// livery -- built once and cloned (cheap) per material so each panel can
// set its own UV repeat to match its own proportions.
// ---------------------------------------------------------------------------
let _solarCellTextureCache = null;
function getSolarCellTexture() {
  if (_solarCellTextureCache) return _solarCellTextureCache;
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#3a3a3a'; // gridline base
  ctx.fillRect(0, 0, 128, 128);
  const cell = 16;
  for (let cy = 0; cy < 128; cy += cell) {
    for (let cx = 0; cx < 128; cx += cell) {
      const shade = 205 + Math.floor(Math.random() * 35); // slight per-cell brightness variation, not perfectly uniform
      ctx.fillStyle = `rgb(${shade},${shade},${Math.min(255, shade + 10)})`; // faint blue-ish cast, like real cell coatings
      ctx.fillRect(cx + 1, cy + 1, cell - 2, cell - 2);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  _solarCellTextureCache = texture;
  return texture;
}

function buildPanelMaterial(colorHex, shininess, specularHex, repeatX, repeatY) {
  const tex = getSolarCellTexture().clone();
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX || 2, repeatY || 5);
  tex.needsUpdate = true;
  return new THREE.MeshPhongMaterial({ color: colorHex, shininess: shininess, specular: specularHex, map: tex });
}

// Scatters a handful of small detail boxes across a span, breaking up the
// too-clean look of a bare primitive surface (real spacecraft buses are
// covered in small boxes, connectors, and instrument housings). Seeded so
// the same satellite always gets the same greebles.
function addGreebles(parentGroup, mat, spanX, spanY, spanZ, count, seedStr) {
  const rand = seededRandom(seedStr || 'greeble');
  for (let i = 0; i < count; i++) {
    const w = spanX * (0.05 + rand() * 0.06);
    const h = spanY * (0.04 + rand() * 0.05);
    const d = spanZ * (0.05 + rand() * 0.06);
    const box = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    box.position.set(
      (rand() - 0.5) * spanX * 0.75,
      (rand() - 0.5) * spanY * 0.75,
      (rand() - 0.5) * spanZ * 0.75
    );
    parentGroup.add(box);
  }
}

// Small steady emissive point -- the "instrument status light" detail
// that helps a model read as an active machine rather than a static prop.
function addIndicatorLight(parentGroup, size, position, colorHex) {
  const mat = new THREE.MeshBasicMaterial({ color: colorHex || 0xff3b30 });
  const light = new THREE.Mesh(new THREE.SphereGeometry(size * 0.012, 6, 6), mat);
  light.position.copy(position);
  parentGroup.add(light);
  return light;
}

// ---------------------------------------------------------------------------
// Generic fallback models -- for the ~31,000 satellites in a loaded
// database that aren't one of the curated types above. Classified by
// name only (see classifyByName()). Each takes a seedStr (the satellite's
// own name, stable across reloads) so a field of many unclassified
// satellites doesn't render as visibly identical clones of one template.
// ---------------------------------------------------------------------------
function buildDebrisModel(size, seedStr) {
  const rand = seededRandom(seedStr);
  const group = new THREE.Group();
  const mat = new THREE.MeshPhongMaterial({ color: 0x55524a, shininess: 5, specular: 0x111111, flatShading: true });
  // Real debris varies from angular shards to rounder fragments -- alternate
  // geometry type per instance, not just per-instance scale, for more
  // visible variety across many debris entries shown at once.
  const baseSize = size * (0.07 + rand() * 0.06);
  const geom = rand() < 0.5
    ? new THREE.TetrahedronGeometry(baseSize, 0)
    : new THREE.IcosahedronGeometry(baseSize, 0);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.scale.set(0.6 + rand() * 0.9, 0.6 + rand() * 0.9, 0.6 + rand() * 0.9);
  mesh.rotation.set(rand() * Math.PI, rand() * Math.PI, rand() * Math.PI);
  group.add(mesh);
  return group;
}

function buildRocketBodyModel(size, seedStr) {
  const rand = seededRandom(seedStr);
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshPhongMaterial({ color: 0x9a9a9a, shininess: 20, specular: 0x333333 });
  const ringMat = buildMLIMaterial(MLI_GOLD, 55); // interstage rings are commonly foil-wrapped on real stages

  const bodyLen = size * (0.42 + rand() * 0.18); // spent stages vary in length; not every one is identical
  const bodyR = size * 0.06;
  const body = new THREE.Mesh(new THREE.CylinderGeometry(bodyR, bodyR, bodyLen, 16), bodyMat);
  body.rotation.z = Math.PI / 2;
  group.add(body);

  // Interstage ring -- a wider, short section common on real spent stages,
  // not present on the old plain-cylinder version.
  const ring = new THREE.Mesh(new THREE.CylinderGeometry(bodyR * 1.25, bodyR * 1.25, size * 0.05, 16), ringMat);
  ring.rotation.z = Math.PI / 2;
  ring.position.set(bodyLen / 2 - size * 0.02, 0, 0);
  group.add(ring);

  // Wider engine bell than before, tapering out rather than just a small taper
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(bodyR * 0.9, bodyR * 1.7, size * 0.11, 16), bodyMat);
  nozzle.rotation.z = Math.PI / 2;
  nozzle.position.set(-bodyLen / 2 - size * 0.04, 0, 0);
  group.add(nozzle);

  return group;
}

function buildGenericPayloadModel(size, seedStr) {
  const rand = seededRandom(seedStr);
  const group = new THREE.Group();
  const busColor = rand() < 0.5 ? MLI_GOLD : MLI_SILVER; // real buses show up in both foil tones
  const busMat = buildMLIMaterial(busColor, 60);

  const busW = size * (0.18 + rand() * 0.12);
  const busH = size * (0.16 + rand() * 0.10);
  const busD = size * (0.18 + rand() * 0.12);
  const bus = new THREE.Mesh(new THREE.BoxGeometry(busW, busH, busD), busMat);
  group.add(bus);

  const panelLen = size * (0.26 + rand() * 0.16);
  const panelWid = size * (0.12 + rand() * 0.08);
  // Repeat scaled to this instance's own randomized proportions, since they
  // vary per satellite -- not a fixed guess that would stretch on some sizes.
  const panelMat = buildPanelMaterial(0x15294f, 18, 0x223344,
    Math.max(2, Math.round(panelLen / (size * 0.07))), Math.max(2, Math.round(panelWid / (size * 0.07))));
  const panelGeom = new THREE.BoxGeometry(size * 0.02, panelLen, panelWid);
  const panelOffset = busH / 2 + panelLen / 2;
  [1, -1].forEach(s => {
    const p = new THREE.Mesh(panelGeom, panelMat);
    p.position.set(0, s * panelOffset, 0);
    group.add(p);
  });

  if (rand() < 0.4) { // roughly 4 in 10 generic satellites get a small dish, for variety
    const dish = new THREE.Mesh(new THREE.ConeGeometry(size * 0.06, size * 0.045, 16), busMat);
    dish.rotation.x = Math.PI;
    dish.position.set(0, 0, busD / 2 + size * 0.02);
    group.add(dish);
  }

  const detailMat = new THREE.MeshPhongMaterial({ color: 0x2a2e38, shininess: 30, specular: 0x333333 });
  addGreebles(group, detailMat, busW, busH, busD, 3, (seedStr || '') + '-greebles');

  if (rand() < 0.3) { // roughly 3 in 10 get a small visible status light
    addIndicatorLight(group, size, new THREE.Vector3(busW * 0.3, busH * 0.3, busD / 2), rand() < 0.5 ? 0xff3b30 : 0x30d158);
  }

  return group;
}


// ---------------------------------------------------------------------------
// TLE-driven selected satellite -- real SGP4 propagation (via satellite.js)
// instead of an idealized circular orbit, with a model matched to the
// vehicle's actual real-world design.
// ---------------------------------------------------------------------------
function tleEpochToMs(satrec) {
  return (satrec.jdsatepoch - 2440587.5) * 86400000;
}

function computeOrbitPeriodMinutes(satrec) {
  return (2 * Math.PI) / satrec.no;
}

// Browser-side propagation is deliberately kept behind one small dispatch
// point. State-vector tracks never enter this path.
function tleStateAtMs(satrec, ms) {
  if (!satrec) return null;
  const pv = satellite.propagate(satrec, new Date(ms));
  return pv && pv.position ? pv : null;
}

function stateVectorStateAtMs(inst, ms) {
  const track = inst && inst.stateTrack;
  if (!track || !Array.isArray(track.t) || !track.t.length) return null;
  const times = track.t, positions = track.r, velocities = track.v;
  if (ms < times[0] || ms > times[times.length - 1]) return null;
  let i = 0;
  if (ms >= times[times.length - 1]) i = times.length - 1;
  else if (ms > times[0]) {
    let lo = 0, hi = times.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= ms) lo = mid; else hi = mid;
    }
    i = lo;
  }
  const j = Math.min(i + 1, times.length - 1);
  const dt = j === i ? 0 : (times[j] - times[i]) / 1000;
  const u = j === i ? 0 : (ms - times[i]) / (times[j] - times[i]);
  const u2 = u * u, u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;
  const dh00 = j === i ? 0 : (6 * u2 - 6 * u) / dt;
  const dh10 = j === i ? 0 : (3 * u2 - 4 * u + 1);
  const dh01 = j === i ? 0 : (-6 * u2 + 6 * u) / dt;
  const dh11 = j === i ? 0 : (3 * u2 - 2 * u);
  const component = (axis, derivative) => j === i
    ? derivative[i][axis]
    : h00 * positions[i][axis] + h10 * dt * velocities[i][axis]
      + h01 * positions[j][axis] + h11 * dt * velocities[j][axis];
  const rate = axis => j === i
    ? velocities[i][axis]
    : dh00 * positions[i][axis] + dh10 * velocities[i][axis]
      + dh01 * positions[j][axis] + dh11 * velocities[j][axis];
  let quaternion = null;
  if (Array.isArray(track.q) && track.q.length) {
    const qa = new THREE.Quaternion(track.q[i][1], track.q[i][2], track.q[i][3], track.q[i][0]);
    const qb = new THREE.Quaternion(track.q[j][1], track.q[j][2], track.q[j][3], track.q[j][0]);
    quaternion = new THREE.Quaternion().slerpQuaternions(qa, qb, u).normalize();
  }
  return {
    position: {
      x: component(0, positions), y: component(1, positions), z: component(2, positions),
    },
    velocity: {
      x: rate(0), y: rate(1), z: rate(2),
    },
    quaternion,
  };
}

// Builds a Line whose vertices fade from bright (at the head -- the
// satellite's current position, index 0) to dim along the path, so the
// orbit reads as DIRECTIONAL at a glance (you can see which way it's
// travelling) instead of a uniform ring. headColor is the bright leading
// color; the tail fades toward transparent-dim.
function buildGradientOrbitLine(pts, headColor) {
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const colors = new Float32Array(pts.length * 3);
  const c = new THREE.Color(headColor);
  for (let i = 0; i < pts.length; i++) {
    // brightness falls off along the path; head (i=0) is full, tail dims
    const t = i / (pts.length - 1);
    const bright = 1.0 - t * 0.82; // keep a faint tail rather than going fully black
    colors[i * 3] = c.r * bright;
    colors[i * 3 + 1] = c.g * bright;
    colors[i * 3 + 2] = c.b * bright;
  }
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
  return new THREE.Line(geom, mat);
}

function buildRealOrbitPath(satrec, epochMs, periodMin, color) {
  const pts = [];
  const steps = 220;
  for (let i = 0; i <= steps; i++) {
    const tMin = (i / steps) * periodMin;
    const pv = tleStateAtMs(satrec, epochMs + tMin * 60000);
    if (pv && pv.position) pts.push(eciToScene(pv.position));
  }
  return buildGradientOrbitLine(pts, color);
}

function buildStateVectorOrbitPath(track, color) {
  const points = (track && Array.isArray(track.r) ? track.r : [])
    .map(p => eciToScene({ x: p[0], y: p[1], z: p[2] }));
  if (points.length < 2) return null;
  return buildGradientOrbitLine(points, color);
}

// Orbital angular momentum direction (position x velocity) -- stays fixed
// for a given orbital plane, so comparing it at two different times
// directly measures how much the real plane has actually precessed
// between them (verified against this exact approach when diagnosing why
// drawn orbit lines drift out of sync with real satellites: SGP4 correctly
// models J2 nodal precession, e.g. ~12 degrees over 69 days for ISS's
// current TLE, but a orbit line sampled once and never updated doesn't
// track that).
function computeOrbitPlaneNormal(satrec, atMs) {
  const pv = tleStateAtMs(satrec, atMs);
  if (!pv || !pv.position) return null;
  const p = new THREE.Vector3(pv.position.x, pv.position.y, pv.position.z);
  const v = new THREE.Vector3(pv.velocity.x, pv.velocity.y, pv.velocity.z);
  return new THREE.Vector3().crossVectors(p, v).normalize();
}

// Redraws a TLE-driven satellite's orbit-path line once its real orbital
// plane has precessed enough from wherever it was last drawn to visibly
// matter, rather than never updating it (which is what caused the drift
// this exists to fix) or redrawing every frame (needlessly expensive --
// this is a real per-call SGP4 cost x 220 samples). Threshold-triggered
// instead of time-triggered so it self-adapts to whatever sim-time-scale
// is selected: real-time barely ever triggers this, sped-up time triggers
// it more often, automatically, without separate logic for each case.
const ORBIT_LINE_REFRESH_THRESHOLD_DEG = 1.0;

function refreshOrbitLineIfNeeded(key) {
  const inst = activeSatellites.get(key);
  if (!inst || !inst.satrec || !inst.lastOrbitPlaneNormal) return; // only real TLE-driven satellites have a plane that can precess this way
  const nowMs = getCurrentSimMs();
  const currentNormal = computeOrbitPlaneNormal(inst.satrec, nowMs);
  if (!currentNormal) return;
  const driftDeg = inst.lastOrbitPlaneNormal.angleTo(currentNormal) * 180 / Math.PI;
  if (driftDeg < ORBIT_LINE_REFRESH_THRESHOLD_DEG) return;

  if (inst.orbitLine) scene.remove(inst.orbitLine);
  inst.orbitLine = buildRealOrbitPath(inst.satrec, nowMs, inst.periodMin, 0xffffff);
  scene.add(inst.orbitLine);
  inst.lastOrbitDrawMs = nowMs;
  inst.lastOrbitPlaneNormal = currentNormal;
}

// ---------------------------------------------------------------------------
// Ground tracks -- the sub-satellite (nadir) point traced onto the Earth's
// surface. Two pieces, split by what frame each naturally lives in:
//
//   * The painted trail is Earth-FIXED. Each sample is converted ECI->ECEF
//     (Rz(-GMST) at that sample's time) and added as a child of earthMesh, so
//     it rotates with the ground and stays locked to the geography the
//     satellite actually passed over. Because earthMesh's own spin is exactly
//     Ry(GMST), the leading tip (sample at "now") lands directly beneath the
//     satellite -- verified algebraically: Ry(gmst)*eciToScene(Rz(-gmst)*eci)
//     == eciToScene(eci), i.e. the two rotations cancel with no fudge factor.
//
//   * The current sub-point marker is INERTIAL and trivial: the nadir point
//     is just the satellite's own position projected to the surface, so the
//     marker is a child of the scene set to model.position.setLength(R). No
//     GMST needed for it, and it stays frame-exact between trail rebuilds.
//
// ECI->ECEF here is GMST-only (no polar motion / nutation), consistent with
// the scene's GMST-only Earth spin -- same TEME-frame simplification noted in
// updateEarthRotation, arcsecond-to-arcminute scale.

// Raw ECI position (km, Z-up) for either propagation mode at a given sim ms.
// 'fixed' (JWST/L2) has no meaningful ground track and returns null.
function eciAtMs(inst, ms) {
  const entry = inst.entry;
  if (entry.mode === 'state') {
    const pv = stateVectorStateAtMs(inst, ms);
    return pv ? pv.position : null;
  }
  if (entry.mode === 'tle') {
    if (!inst.satrec) return null;
    const pv = tleStateAtMs(inst.satrec, ms);
    return (pv && pv.position) ? pv.position : null;
  }
  return null;
}

// ECI -> Earth-fixed sub-satellite point, expressed in earthMesh-LOCAL scene
// coordinates on a sphere of radius surfaceR.
function subSatellitePointLocal(eci, gmst, surfaceR) {
  const cg = Math.cos(gmst), sg = Math.sin(gmst);
  const ex =  eci.x * cg + eci.y * sg;   // Rz(-gmst) * eci  (ECI -> ECEF)
  const ey = -eci.x * sg + eci.y * cg;
  const ez =  eci.z;
  const inv = surfaceR / Math.sqrt(ex * ex + ey * ey + ez * ez);
  return new THREE.Vector3(ex * inv, ez * inv, -ey * inv); // eciToScene remap of the ECEF vector
}

// Builds the painted trail line (child of earthMesh). Sample 0 = now (bright
// leading tip via buildGradientOrbitLine's head-fade), increasing index =
// further into the past. Span is one orbital period, capped so very long
// orbits don't paint an unreadably long near-stationary smear.
function buildGroundTrackLine(inst) {
  if (!inst || (inst.entry.mode !== 'tle' && inst.entry.mode !== 'state')) return null;
  const nowMs = getCurrentSimMs();
  const spanMin = Math.min(inst.periodMin || 95, GROUND_TRACK_MAX_SPAN_MIN);
  const pts = [];
  for (let i = 0; i <= GROUND_TRACK_SAMPLES; i++) {
    const ms = nowMs - (i / GROUND_TRACK_SAMPLES) * spanMin * 60000;
    const eci = eciAtMs(inst, ms);
    if (!eci) continue;
    const gmst = satellite.gstime(new Date(ms));
    pts.push(subSatellitePointLocal(eci, gmst, GROUND_TRACK_SURFACE_R));
  }
  if (pts.length < 2) return null;
  const line = buildGradientOrbitLine(pts, GROUND_TRACK_COLOR);
  line.material.opacity = 0.9;
  return line;
}

function attachGroundTrack(inst) {
  if (!inst || inst.entry.mode === 'fixed' || inst.entry.mode === 'metadata') return; // no ground track for non-orbital entries
  if (!inst.groundTrackLine) {
    const line = buildGroundTrackLine(inst);
    if (line) { inst.groundTrackLine = line; earthMesh.add(line); }
  }
  if (!inst.subPointMarker) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(60, R_EARTH_KM * 0.012), 12, 12),
      new THREE.MeshBasicMaterial({ color: GROUND_TRACK_COLOR })
    );
    inst.subPointMarker = m;
    scene.add(m);
    if (inst.model) m.position.copy(inst.model.position).setLength(GROUND_TRACK_SURFACE_R);
  }
}

function detachGroundTrack(inst) {
  if (!inst) return;
  if (inst.groundTrackLine) {
    earthMesh.remove(inst.groundTrackLine);
    inst.groundTrackLine.geometry.dispose();
    inst.groundTrackLine = null;
  }
  if (inst.subPointMarker) {
    scene.remove(inst.subPointMarker);
    inst.subPointMarker.geometry.dispose();
    inst.subPointMarker.material.dispose();
    inst.subPointMarker = null;
  }
}

function setGroundTracksEnabled(on) {
  groundTracksEnabled = on;
  for (const inst of activeSatellites.values()) {
    if (on) attachGroundTrack(inst); else detachGroundTrack(inst);
  }
}

function setQuaternionAttitudeEnabled(on) {
  quaternionAttitudeEnabled = Boolean(on);
  updateAllActiveSatellitePositions();
}

function refreshAttitudeControl() {
  const toggle = document.getElementById('attitude-toggle');
  if (!toggle) return;
  const available = [...activeSatellites.values()].some(inst =>
    inst.entry.mode === 'state' && inst.stateTrack &&
    Array.isArray(inst.stateTrack.q) && inst.stateTrack.q.length
  );
  toggle.disabled = !available;
  toggle.title = available
    ? 'Use supplied body-to-TEME quaternions when state-vector data includes q=[w,x,y,z].'
    : 'Body-to-TEME quaternion attitude is available when state-vector data includes q=[w,x,y,z].';
  if (!available) {
    toggle.checked = false;
    quaternionAttitudeEnabled = false;
  }
}

// The painted trail's tail retreats and its tip advances as sim time moves, so
// it's rebuilt on a wall-clock cadence (the marker stays frame-exact in
// between, so this only refreshes the historical trail, not the live nadir).
let lastGroundTrackRebuildMs = 0;
function refreshGroundTracks() {
  if (!groundTracksEnabled) return;
  const nowPerf = performance.now();
  if (nowPerf - lastGroundTrackRebuildMs < 900) return;
  lastGroundTrackRebuildMs = nowPerf;
  for (const inst of activeSatellites.values()) {
    if (!inst.groundTrackLine) continue; // fixed-mode sats never get one
    const fresh = buildGroundTrackLine(inst);
    if (!fresh) continue;
    earthMesh.remove(inst.groundTrackLine);
    inst.groundTrackLine.geometry.dispose();
    inst.groundTrackLine = fresh;
    earthMesh.add(fresh);
  }
}

// Current sub-point marker: satellite position projected straight down to the
// surface, in the inertial scene frame. Cheap and exact every frame.
function updateSubPointMarkers() {
  if (!groundTracksEnabled) return;
  for (const inst of activeSatellites.values()) {
    if (!inst.subPointMarker || !inst.model || inst.entry.mode === 'fixed') continue;
    inst.subPointMarker.position.copy(inst.model.position).setLength(GROUND_TRACK_SURFACE_R);
  }
}

// Separation (km) between two satellites at sim-time t. Distances are computed
// in raw ECI, but note eciToScene is a pure rotation, so the ECI separation
// equals the scene separation exactly -- no need to remap first.
function pairSeparationKm(instA, instB, t) {
  const a = eciAtMs(instA, t), b = eciAtMs(instB, t);
  if (!a || !b) return Infinity;
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Ternary search for the true time of closest approach in a coarse bracket.
// Separation-vs-time near an approach is smooth and unimodal, so ternary
// search converges to the minimum quickly. Also returns the relative speed at
// TCA (finite difference of the separation vector over 1 s).
function refineConjunction(instA, instB, tCoarseMs, stepMs) {
  let lo = tCoarseMs - stepMs, hi = tCoarseMs + stepMs;
  for (let k = 0; k < 60 && (hi - lo) > 1; k++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (pairSeparationKm(instA, instB, m1) <= pairSeparationKm(instA, instB, m2)) hi = m2; else lo = m1;
  }
  const tca = (lo + hi) / 2;
  const missKm = pairSeparationKm(instA, instB, tca);
  const a0 = eciAtMs(instA, tca - 500), b0 = eciAtMs(instB, tca - 500);
  const a1 = eciAtMs(instA, tca + 500), b1 = eciAtMs(instB, tca + 500);
  let relVelKmS = 0;
  if (a0 && b0 && a1 && b1) {
    relVelKmS = Math.hypot((a1.x - b1.x) - (a0.x - b0.x),
                           (a1.y - b1.y) - (a0.y - b0.y),
                           (a1.z - b1.z) - (a0.z - b0.z)); // over a 1.0 s span -> km/s
  }
  return { tcaMs: tca, missKm, relVelKmS };
}

// Coarse pass caches each satellite's position once per timestep (N props/step,
// not N^2), then all pairwise distances read from the cache -- so cost scales
// with N x steps, not pairs x steps. Every pair whose coarse minimum is within
// a widened gate is then refined to a true TCA; those under threshold are kept.
function screenConjunctions() {
  const entries = [...activeSatellites.entries()].filter(
    ([, i]) => i.entry.mode === 'tle' || i.entry.mode === 'state');
  const n = entries.length;
  conjunctionEvents = [];
  if (n < 2) return { status: 'need_more' };

  const nowMs = getCurrentSimMs();
  const windowMs = conjunctionWindowHours * 3600000;
  const numSteps = Math.max(1, Math.round(windowMs / CONJ_COARSE_STEP_MS));
  const gate = conjunctionThresholdKm * 4; // only refine pairs that got remotely close
  const bestDist = new Float64Array(n * n).fill(Infinity);
  const bestT = new Float64Array(n * n);
  const pos = new Array(n);

  for (let s = 0; s <= numSteps; s++) {
    const t = nowMs + s * CONJ_COARSE_STEP_MS;
    for (let i = 0; i < n; i++) { const e = eciAtMs(entries[i][1], t); pos[i] = e ? [e.x, e.y, e.z] : null; }
    for (let i = 0; i < n; i++) {
      if (!pos[i]) continue;
      for (let j = i + 1; j < n; j++) {
        if (!pos[j]) continue;
        const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1], dz = pos[i][2] - pos[j][2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const idx = i * n + j;
        if (d < bestDist[idx]) { bestDist[idx] = d; bestT[idx] = t; }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const idx = i * n + j;
      if (bestDist[idx] > gate) continue;
      const A = entries[i][1], B = entries[j][1];
      const r = refineConjunction(A, B, bestT[idx], CONJ_COARSE_STEP_MS);
      if (r.missKm <= conjunctionThresholdKm) {
        conjunctionEvents.push({
          keyA: entries[i][0], keyB: entries[j][0],
          nameA: A.entry.name, nameB: B.entry.name,
          tcaMs: r.tcaMs, missKm: r.missKm, relVelKmS: r.relVelKmS,
        });
      }
    }
  }
  conjunctionEvents.sort((a, b) => a.missKm - b.missKm);
  return { status: 'ok', count: conjunctionEvents.length, pairs: n * (n - 1) / 2 };
}

function conjunctionColor(missKm) {
  if (missKm < 1) return 0xff4d4d;   // red -- very close
  if (missKm < 5) return 0xffa53d;   // orange
  return 0xffe066;                    // yellow -- within threshold but comfortable
}

// Live risk-colored connector between the two objects, drawn only when sim
// time is within +/- CONJ_LINE_SHOW_MIN of an event's TCA (so clicking a
// result, which jumps to TCA, makes the line appear at the approach).
function updateConjunctionLines() {
  const nowMs = getCurrentSimMs();
  const showMs = CONJ_LINE_SHOW_MIN * 60000;
  const wanted = new Set();
  for (const ev of conjunctionEvents) {
    if (Math.abs(nowMs - ev.tcaMs) > showMs) continue;
    const a = activeSatellites.get(ev.keyA), b = activeSatellites.get(ev.keyB);
    if (!a || !b || !a.model || !b.model) continue;
    const id = ev.keyA + '|' + ev.keyB;
    wanted.add(id);
    const pts = [a.model.position.clone(), b.model.position.clone()];
    let line = conjunctionLines.get(id);
    if (!line) {
      const mat = new THREE.LineBasicMaterial({ color: conjunctionColor(ev.missKm), transparent: true, opacity: 0.95 });
      line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
      conjunctionLines.set(id, line);
      scene.add(line);
    } else {
      line.geometry.setFromPoints(pts);
      line.geometry.attributes.position.needsUpdate = true;
    }
  }
  for (const [id, line] of conjunctionLines) {
    if (wanted.has(id)) continue;
    scene.remove(line); line.geometry.dispose(); line.material.dispose();
    conjunctionLines.delete(id);
  }
}

function clearConjunctionsForKey(key) {
  if (conjunctionEvents.length) {
    conjunctionEvents = conjunctionEvents.filter(ev => ev.keyA !== key && ev.keyB !== key);
    renderConjunctionResults();
  }
  for (const [id, line] of conjunctionLines) {
    if (id.split('|').indexOf(key) === -1) continue;
    scene.remove(line); line.geometry.dispose(); line.material.dispose();
    conjunctionLines.delete(id);
  }
}

function fmtUtcShort(ms) {
  return new Date(ms).toISOString().replace('T', ' ').slice(0, 16) + 'Z';
}

function renderConjunctionResults(res) {
  const box = document.getElementById('conj-results');
  if (!box) return;
  if (res && res.status === 'need_more') {
    box.innerHTML = '<div class="conj-empty">Add at least 2 propagatable satellites to screen.</div>';
    return;
  }
  if (!conjunctionEvents.length) {
    box.innerHTML = (res && res.status === 'ok')
      ? `<div class="conj-empty">No approaches under ${conjunctionThresholdKm} km in the next ${conjunctionWindowHours} h (screened ${res.pairs} pair${res.pairs === 1 ? '' : 's'}).</div>`
      : '<div class="conj-empty">Screen to compute closest approaches.</div>';
    return;
  }
  box.innerHTML = conjunctionEvents.map(ev => {
    const col = '#' + conjunctionColor(ev.missKm).toString(16).padStart(6, '0');
    return `<div class="conj-event" data-tca="${ev.tcaMs}" style="border-left-color:${col}">` +
      `<div class="conj-pair">${escapeHtml(ev.nameA)} &harr; ${escapeHtml(ev.nameB)}</div>` +
      `<div class="conj-meta">miss ${ev.missKm.toFixed(ev.missKm < 10 ? 2 : 1)} km &middot; ` +
      `${fmtUtcShort(ev.tcaMs)} &middot; ${ev.relVelKmS.toFixed(2)} km/s rel</div></div>`;
  }).join('');
  box.querySelectorAll('.conj-event').forEach(el => {
    el.addEventListener('click', () => setSimTime(Number(el.getAttribute('data-tca'))));
  });
  const _cxb = document.getElementById('conj-export-btn'); if (_cxb) _cxb.disabled = !conjunctionEvents.length;
}

// ---------------------------------------------------------------------------
// Pass predictions -- when a chosen satellite rises above a ground site, its
// culmination (max elevation) and set, computed with satellite.js's own
// look-angle math (eciToEcf + ecfToLookAngles) against a WGS84 observer. This
// is the third SSA tool alongside ground tracks and conjunction screening, and
// it reuses the same propagation primitives.
let observerGd = null;                 // {longitude, latitude, height} rad/km
let observerPin = null;                // marker on the globe at the site
let passTargetKey = null;              // which satellite passes were computed for
let passEvents = [];
let passMinElevationDeg = 10;
let inViewLine = null;                 // observer->satellite connector while above the horizon
const PASS_COARSE_STEP_MS = 30000;     // 30 s coarse grid; rise/set/culmination refined

function radToDeg360(r) { return ((r * 180 / Math.PI) % 360 + 360) % 360; }

// Look angles (azimuth/elevation/range) of a satellite from the observer at t.
function lookAnglesAt(inst, t) {
  const eci = eciAtMs(inst, t);
  if (!eci || !observerGd) return null;
  const ecf = satellite.eciToEcf(eci, satellite.gstime(new Date(t)));
  return satellite.ecfToLookAngles(observerGd, ecf); // {azimuth, elevation, rangeSat}
}

function setObserver(latDeg, lonDeg, altKm) {
  observerGd = {
    latitude: latDeg * Math.PI / 180,
    longitude: lonDeg * Math.PI / 180,
    height: altKm || 0,
  };
  // Place / move the site marker (Earth-fixed, child of earthMesh, same
  // geodetic->ECEF->scene-local path the ground track uses).
  const ecf = satellite.geodeticToEcf(observerGd); // {x,y,z} ECEF km
  const local = eciToScene(ecf).setLength(GROUND_TRACK_SURFACE_R);
  if (!observerPin) {
    observerPin = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(70, R_EARTH_KM * 0.014), 14, 14),
      new THREE.MeshBasicMaterial({ color: 0x66ff99 })
    );
    earthMesh.add(observerPin);
  }
  observerPin.position.copy(local);
}

// Marches elevation over the window on a coarse grid, bisects each horizon
// crossing for precise rise/set, and ternary-searches each pass for its
// culmination. Handles the two edge cases honestly: continuously visible
// (e.g. a GEO satellite over its sub-point) and never visible.
// --- Optical visibility (illumination geometry) ----------------------------
// A satellite is naked-eye/optically visible only when the OBSERVER is in
// darkness AND the SATELLITE is sunlit. This is the dominant "can I see it"
// filter -- most numerically-above-the-horizon passes are NOT visible because
// they happen in daylight or with the satellite in Earth's shadow. Both parts
// are exact from the ephemeris we already have.
const TWILIGHT_SUN_ELEV_DEG = -6; // civil twilight: sky dark enough to see a sunlit satellite

// Sun elevation at the observer (radians). Places a point far along the sun
// direction and reuses satellite.js's own topocentric look-angle transform.
function sunElevationRad(t) {
  if (!observerGd) return 0;
  const s = computeSunDirectionEci(new Date(t)); // ECI unit vector, Z-up
  const gmst = satellite.gstime(new Date(t));
  const sEcf = satellite.eciToEcf({ x: s.x, y: s.y, z: s.z }, gmst);
  const o = satellite.geodeticToEcf(observerGd);
  const far = { x: o.x + sEcf.x * 1e9, y: o.y + sEcf.y * 1e9, z: o.z + sEcf.z * 1e9 };
  return satellite.ecfToLookAngles(observerGd, far).elevation;
}

// Is a satellite (ECI position) sunlit, i.e. NOT in Earth's shadow? Standard
// cylindrical-umbra approximation: if it's on the anti-sun side and its
// perpendicular distance from the Earth-sun line is under Earth's radius, it's
// eclipsed. (Cylinder rather than the umbra cone -- at LEO/MEO the difference
// is negligible; a conservative, standard visible-pass approximation.)
function satelliteSunlit(eci, t) {
  const s = computeSunDirectionEci(new Date(t));
  const rDotS = eci.x * s.x + eci.y * s.y + eci.z * s.z;
  if (rDotS >= 0) return true; // sun-facing side of Earth center: can't be shadowed
  const rMag2 = eci.x * eci.x + eci.y * eci.y + eci.z * eci.z;
  const perp = Math.sqrt(Math.max(0, rMag2 - rDotS * rDotS));
  return perp > R_EARTH_KM;
}

// Full visibility state of the target from the site at time t.
function visibilityAt(inst, t, mask) {
  const eci = eciAtMs(inst, t);
  const la = eci ? lookAnglesAt(inst, t) : null;
  const up = !!(la && la.elevation > mask);
  const dark = sunElevationRad(t) < TWILIGHT_SUN_ELEV_DEG * Math.PI / 180;
  const sunlit = eci ? satelliteSunlit(eci, t) : false;
  return { up, dark, sunlit, visible: up && dark && sunlit };
}

function computePasses(targetKey, windowHours) {
  const inst = activeSatellites.get(targetKey);
  if (!inst || (inst.entry.mode !== 'tle' && inst.entry.mode !== 'state')) return { status: 'bad_target' };
  if (!observerGd) return { status: 'no_site' };
  passTargetKey = targetKey;
  passEvents = [];

  const nowMs = getCurrentSimMs();
  const mask = passMinElevationDeg * Math.PI / 180;
  const steps = Math.max(1, Math.round(windowHours * 3600000 / PASS_COARSE_STEP_MS));
  const endMs = nowMs + steps * PASS_COARSE_STEP_MS;
  const elev = t => { const la = lookAnglesAt(inst, t); return la ? la.elevation : -Math.PI / 2; };

  // Bisect the horizon crossing (elev == mask) within a bracket that contains it.
  const bisect = (t0, t1) => {
    const s0 = elev(t0) > mask;
    for (let k = 0; k < 45 && (t1 - t0) > 250; k++) {
      const tm = (t0 + t1) / 2;
      if ((elev(tm) > mask) === s0) t0 = tm; else t1 = tm;
    }
    return (t0 + t1) / 2;
  };

  const buildPass = (riseMs, setMs, continuous) => {
    let lo = riseMs, hi = setMs; // ternary-search the max elevation
    for (let k = 0; k < 60 && (hi - lo) > 1000; k++) {
      const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
      if (elev(m1) < elev(m2)) lo = m1; else hi = m2;
    }
    const culMs = (lo + hi) / 2;
    const cul = lookAnglesAt(inst, culMs), rise = lookAnglesAt(inst, riseMs), set = lookAnglesAt(inst, setMs);
    // Optical visibility: scan the pass for any moment the target is up, the
    // site is dark, and the satellite is sunlit -- and bound that visible window.
    let visRiseMs = null, visSetMs = null;
    const SAMPLES = 60;
    for (let i = 0; i <= SAMPLES; i++) {
      const t = riseMs + (setMs - riseMs) * (i / SAMPLES);
      if (visibilityAt(inst, t, mask).visible) { if (visRiseMs == null) visRiseMs = t; visSetMs = t; }
    }
    let visibility;
    if (visRiseMs != null) visibility = 'visible';
    else visibility = visibilityAt(inst, culMs, mask).dark ? 'eclipsed' : 'daylight'; // dark-but-shadowed vs sunlit-sky
    return {
      riseMs, riseAzDeg: radToDeg360(rise.azimuth),
      culMs, maxElDeg: cul.elevation * 180 / Math.PI, culAzDeg: radToDeg360(cul.azimuth), rangeKm: cul.rangeSat,
      setMs, setAzDeg: radToDeg360(set.azimuth),
      durationMin: (setMs - riseMs) / 60000,
      continuous: !!continuous,
      visibility, visRiseMs, visSetMs,
    };
  };

  let prevUp = elev(nowMs) > mask;
  let riseMs = prevUp ? nowMs : null; // already above the horizon at window start
  for (let s = 1; s <= steps; s++) {
    const t = nowMs + s * PASS_COARSE_STEP_MS;
    const up = elev(t) > mask;
    if (up && !prevUp) {
      riseMs = bisect(t - PASS_COARSE_STEP_MS, t);
    } else if (!up && prevUp) {
      const setMs = bisect(t - PASS_COARSE_STEP_MS, t);
      if (riseMs != null) passEvents.push(buildPass(riseMs, setMs, false));
      riseMs = null;
    }
    prevUp = up;
  }
  if (riseMs != null) {
    // Still above the horizon at window end: continuous only if it never set
    // AND was already up at the very start (GEO-like); otherwise it's a pass
    // that runs past the window edge.
    passEvents.push(buildPass(riseMs, endMs, riseMs === nowMs));
  }
  passEvents.sort((a, b) => a.riseMs - b.riseMs);
  return { status: 'ok', count: passEvents.length };
}

function removeInViewLine() {
  if (!inViewLine) return;
  scene.remove(inViewLine); inViewLine.geometry.dispose(); inViewLine.material.dispose(); inViewLine = null;
}

// Live green connector from the site to the target while it's above the mask --
// the "it's overhead now" visual, drawn only when there's a valid target+site.
function updateInViewLine() {
  if (!observerGd || !observerPin || !passTargetKey) { removeInViewLine(); return; }
  const inst = activeSatellites.get(passTargetKey);
  if (!inst || !inst.model) { removeInViewLine(); return; }
  const nowMs = getCurrentSimMs();
  const la = lookAnglesAt(inst, nowMs);
  if (!la || la.elevation <= passMinElevationDeg * Math.PI / 180) { removeInViewLine(); return; }
  // Bright green only when it's genuinely visible right now (site dark + sat
  // sunlit); muted when it's up but in daylight or in Earth's shadow.
  const vis = visibilityAt(inst, nowMs, passMinElevationDeg * Math.PI / 180);
  const color = vis.visible ? 0x66ff99 : 0x4a6a5c;
  const obsWorld = new THREE.Vector3();
  observerPin.getWorldPosition(obsWorld);
  const pts = [obsWorld, inst.model.position.clone()];
  if (!inViewLine) {
    inViewLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: vis.visible ? 0.95 : 0.5 })
    );
    scene.add(inViewLine);
  } else {
    inViewLine.geometry.setFromPoints(pts);
    inViewLine.geometry.attributes.position.needsUpdate = true;
    inViewLine.material.color.setHex(color);
    inViewLine.material.opacity = vis.visible ? 0.95 : 0.5;
  }
}

function refreshPassTargetOptions() {
  const sel = document.getElementById('pass-target');
  if (!sel) return;
  const prev = sel.value;
  const opts = [...activeSatellites.entries()]
    .filter(([, i]) => i.entry.mode === 'tle' || i.entry.mode === 'state')
    .map(([k, i]) => `<option value="${k}">${escapeHtml(i.entry.name)}</option>`);
  sel.innerHTML = opts.join('') || '<option value="">(no propagatable satellites)</option>';
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
}

function fmtHM(ms) { return new Date(ms).toISOString().slice(11, 16); }

function passElevationColor(el) {
  if (el >= 60) return '#6fd68a';   // excellent overhead pass
  if (el >= 30) return '#c7e06a';
  if (el >= 15) return '#ffe066';
  return '#e0a840';                  // low, grazing
}

// --- Optional cloud-cover layer (opt-in, external Open-Meteo request) -------
// Off by default. When enabled, annotates near-term passes with forecast cloud
// cover at culmination. This is the only part of the tool that reaches the
// network and the only part whose data is a forecast (reliable a few days out),
// so it degrades gracefully and is clearly labeled as such.
let cloudEnabled = false;
let cloudData = null; // { times:[ms...], clouds:[%...] }

async function fetchCloudCover(latDeg, lonDeg) {
  // Layered cloud, not just the total. For seeing a satellite this matters a
  // lot: low/mid deck (stratus, altostratus) is effectively opaque, while high
  // cirrus is thin enough that a bright target is often still visible through
  // it. Using total cloud cover alone treats those as equivalent, which is
  // misleading for observation planning.
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${latDeg.toFixed(4)}` +
    `&longitude=${lonDeg.toFixed(4)}` +
    `&hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high` +
    `&forecast_days=7&timezone=UTC`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
  const j = await resp.json();
  const h = j.hourly;
  if (!h || !h.time || !h.cloud_cover) throw new Error('unexpected response');
  return {
    times: h.time.map(s => Date.parse(s + ':00Z')),
    total: h.cloud_cover,
    low: h.cloud_cover_low || h.cloud_cover.map(() => null),
    mid: h.cloud_cover_mid || h.cloud_cover.map(() => null),
    high: h.cloud_cover_high || h.cloud_cover.map(() => null),
    fetchedMs: Date.now(),
  };
}

// Linear interpolation between the bracketing hourly samples, rather than
// snapping to the nearest hour -- a pass at :30 sits between two very
// different hours during a frontal passage.
function lerpSeries(series, i0, i1, f) {
  const a = series[i0], b = series[i1];
  if (a == null || b == null) return (a == null ? b : a);
  return a + (b - a) * f;
}

// Sky conditions at a time: layered cloud + an opacity-weighted obstruction
// estimate + forecast lead time (confidence degrades with lead).
function skyAt(ms) {
  if (!cloudData || !cloudData.times.length) return null;
  const t = cloudData.times;
  if (ms < t[0] - 3600e3 || ms > t[t.length - 1] + 3600e3) return null; // outside forecast horizon
  let i1 = 0;
  while (i1 < t.length - 1 && t[i1] < ms) i1++;
  const i0 = Math.max(0, i1 - 1);
  const span = (t[i1] - t[i0]) || 1;
  const f = Math.max(0, Math.min(1, (ms - t[i0]) / span));
  const total = lerpSeries(cloudData.total, i0, i1, f);
  const low = lerpSeries(cloudData.low, i0, i1, f);
  const mid = lerpSeries(cloudData.mid, i0, i1, f);
  const high = lerpSeries(cloudData.high, i0, i1, f);
  // Opacity weighting: low/mid decks block a point source almost completely;
  // cirrus attenuates but a bright satellite often still shows through, so it
  // contributes at a reduced weight. This is a documented heuristic for
  // observability, not a radiative-transfer result.
  const HIGH_TRANSMISSIVITY = 0.45; // fraction of high cloud that still permits a sighting
  let obstruction;
  if (low == null && mid == null && high == null) {
    obstruction = total; // layered fields unavailable -- fall back to total
  } else {
    const opaque = Math.max(low || 0, mid || 0);
    const thin = (high || 0) * (1 - HIGH_TRANSMISSIVITY);
    obstruction = Math.min(100, Math.max(opaque, thin));
  }
  const leadH = (ms - (cloudData.fetchedMs || Date.now())) / 3600e3;
  return {
    total: Math.round(total), low: low == null ? null : Math.round(low),
    mid: mid == null ? null : Math.round(mid), high: high == null ? null : Math.round(high),
    obstruction: Math.round(obstruction), leadH,
    confidence: leadH <= 48 ? 'good' : (leadH <= 96 ? 'fair' : 'low'),
  };
}


function cloudColor(pct) {
  if (pct <= 25) return '#6fd68a';   // clear-ish
  if (pct <= 60) return '#ffe066';
  return '#e0a840';                   // mostly cloudy
}

async function updateCloudAnnotations() {
  const statusEl = document.getElementById('cloud-status');
  if (!cloudEnabled || !observerGd) {
    cloudData = null;
    if (statusEl) statusEl.textContent = '';
    renderPassResults({ status: 'ok' });
    return;
  }
  if (statusEl) statusEl.textContent = 'sky: loading forecast...';
  try {
    cloudData = await fetchCloudCover(observerGd.latitude * 180 / Math.PI, observerGd.longitude * 180 / Math.PI);
    if (statusEl) statusEl.textContent = 'sky: Open-Meteo layered cloud \u2014 % of sky blocking the target at culmination (high cirrus weighted lighter); forecast, near-term most reliable';
  } catch (e) {
    cloudData = null;
    if (statusEl) statusEl.textContent = 'sky: forecast unavailable (offline or blocked)';
  }
  renderPassResults({ status: 'ok' });
}

const VIS_BADGE = {
  visible:  { dot: '#6fd68a', label: 'visible' },
  daylight: { dot: '#7d8796', label: 'daylight' },
  eclipsed: { dot: '#4a5568', label: 'in shadow' },
};

function renderPassResults(res) {
  const box = document.getElementById('pass-results');
  if (!box) return;
  if (res && res.status === 'no_site') { box.innerHTML = '<div class="conj-empty">Enter a site latitude/longitude first.</div>'; return; }
  if (res && res.status === 'bad_target') { box.innerHTML = '<div class="conj-empty">Pick a propagatable target satellite.</div>'; return; }
  const inst = passTargetKey ? activeSatellites.get(passTargetKey) : null;
  const tName = inst ? escapeHtml(inst.entry.name) : 'target';
  if (!passEvents.length) {
    box.innerHTML = (res && res.status === 'ok')
      ? `<div class="conj-empty">${tName} does not rise above ${passMinElevationDeg}&deg; over this site in the next window.</div>`
      : '<div class="conj-empty">Set a site and target, then Compute.</div>';
    return;
  }
  box.innerHTML = passEvents.map(p => {
    // Border reflects the thing the user actually asked: can I SEE it. Green =
    // optically visible; muted = up but daylight/shadowed.
    const vis = VIS_BADGE[p.visibility] || VIS_BADGE.daylight;
    const border = p.visibility === 'visible' ? passElevationColor(p.maxElDeg) : '#3a4657';
    const badge = `<span class="vis-badge"><span class="vis-dot" style="background:${vis.dot}"></span>${vis.label}` +
      (p.visibility === 'visible' && p.visRiseMs != null ? ` ${fmtHM(p.visRiseMs)}\u2013${fmtHM(p.visSetMs)}` : '') + `</span>`;
    const sky = skyAt(p.culMs);
    let cloud = '';
    if (sky) {
      const parts = [];
      if (sky.low != null) parts.push('low ' + sky.low + '%');
      if (sky.mid != null) parts.push('mid ' + sky.mid + '%');
      if (sky.high != null) parts.push('high ' + sky.high + '%');
      const tip = parts.length ? parts.join(' \u00B7 ') + ` (total ${sky.total}%)` : `total ${sky.total}%`;
      const conf = sky.confidence === 'good' ? '' : ` <span style="opacity:0.6">(${sky.confidence} conf)</span>`;
      cloud = ` &middot; <span style="color:${cloudColor(sky.obstruction)}" title="${tip}">sky ${sky.obstruction}% blocked</span>${conf}`;
    }
    if (p.continuous) {
      return `<div class="conj-event" data-tca="${p.culMs}" style="border-left-color:${border}">` +
        `<div class="conj-pair">${tName} &middot; continuously above horizon ${badge}</div>` +
        `<div class="conj-meta">max el ${p.maxElDeg.toFixed(0)}&deg; &middot; range ${p.rangeKm.toFixed(0)} km${cloud}</div></div>`;
    }
    return `<div class="conj-event" data-tca="${p.culMs}" style="border-left-color:${border}">` +
      `<div class="conj-pair">${fmtHM(p.riseMs)} &uarr; &rarr; ${fmtHM(p.setMs)} &darr; &middot; ${p.durationMin.toFixed(0)} min ${badge}</div>` +
      `<div class="conj-meta">max el ${p.maxElDeg.toFixed(0)}&deg; at ${fmtHM(p.culMs)} (az ${p.culAzDeg.toFixed(0)}&deg;) &middot; ${p.rangeKm.toFixed(0)} km${cloud}</div></div>`;
  }).join('');
  box.querySelectorAll('.conj-event').forEach(el => {
    el.addEventListener('click', () => setSimTime(Number(el.getAttribute('data-tca'))));
  });
  const _pxb = document.getElementById('pass-export-btn'); if (_pxb) _pxb.disabled = !passEvents.length;
}

// --- CSV export (manual, on button click) ---------------------------------
// Exports the computed analysis tables. Uses a Blob + object URL so it works
// fully offline; never triggered automatically -- only on an explicit click.
function csvCell(v) {
  const s = (v == null) ? '' : String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function csvStamp() { return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19); }
function csvSafe(s) { return String(s).replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 40); }

function downloadCsv(filename, rows) {
  const text = rows.map(r => r.map(csvCell).join(',')).join('\r\n');
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportConjunctionsCsv() {
  if (!conjunctionEvents.length) return;
  const rows = [['object_a', 'object_b', 'tca_utc', 'miss_km', 'rel_vel_km_s']];
  for (const e of conjunctionEvents) {
    rows.push([e.nameA, e.nameB, new Date(e.tcaMs).toISOString(), e.missKm.toFixed(3), e.relVelKmS.toFixed(3)]);
  }
  downloadCsv(`ssapy_conjunctions_${csvStamp()}.csv`, rows);
}

function exportPassesCsv() {
  if (!passEvents.length) return;
  const inst = passTargetKey ? activeSatellites.get(passTargetKey) : null;
  const tName = inst ? inst.entry.name : 'target';
  const latDeg = observerGd ? (observerGd.latitude * 180 / Math.PI).toFixed(4) : '';
  const lonDeg = observerGd ? (observerGd.longitude * 180 / Math.PI).toFixed(4) : '';
  const rows = [['target', 'site_lat_deg', 'site_lon_deg', 'min_el_deg',
    'rise_utc', 'rise_az_deg', 'culmination_utc', 'max_el_deg', 'cul_az_deg', 'range_km',
    'set_utc', 'set_az_deg', 'duration_min', 'visibility',
    'cloud_total_pct', 'cloud_low_pct', 'cloud_mid_pct', 'cloud_high_pct',
    'sky_blocked_pct', 'forecast_confidence']];
  for (const p of passEvents) {
    const s = skyAt(p.culMs);
    rows.push([tName, latDeg, lonDeg, passMinElevationDeg,
      new Date(p.riseMs).toISOString(), p.riseAzDeg.toFixed(1),
      new Date(p.culMs).toISOString(), p.maxElDeg.toFixed(1), p.culAzDeg.toFixed(1), p.rangeKm.toFixed(1),
      new Date(p.setMs).toISOString(), p.setAzDeg.toFixed(1), p.durationMin.toFixed(1),
      p.visibility,
      s ? s.total : '', s && s.low != null ? s.low : '', s && s.mid != null ? s.mid : '',
      s && s.high != null ? s.high : '', s ? s.obstruction : '', s ? s.confidence : '']);
  }
  downloadCsv(`ssapy_passes_${csvSafe(tName)}_${csvStamp()}.csv`, rows);
}

// --- Satellite shading: eclipse dimming + earthshine ------------------------
// The scene's directional light follows the real sun, but Three.js won't cast
// Earth's shadow onto a satellite without an expensive shadow-map setup. We
// already compute the umbra analytically for pass visibility, so reuse it:
// a satellite inside Earth's shadow is darkened, which is both physically
// correct and the visual cue that explains why it can't be seen right then.
// Eclipsed craft aren't pure black in reality -- they're weakly lit by
// earthshine -- so the floor is a dim blue-grey rather than zero.
const ECLIPSE_DARK = 0.28;      // earthshine floor
const ECLIPSE_TINT = new THREE.Color(0x9fb6d4); // cool cast of light reflected off Earth

function applyEclipseShading(inst, sunlit) {
  if (inst._eclipseState === sunlit) return; // only touch materials on transition
  inst._eclipseState = sunlit;
  // NOTE: several models deliberately share one material across multiple meshes
  // (e.g. Hubble/GPS/JWST use a single panel or mirror material for both wings).
  // The baseline colour is therefore cached on the MATERIAL, not the mesh, and
  // each material is processed once per transition -- keying it per-mesh would
  // let the second mesh capture an already-darkened colour as its baseline and
  // leave the material permanently dim after one eclipse cycle.
  const seen = new Set();
  inst.model.traverse(obj => {
    const mats = Array.isArray(obj.material) ? obj.material : (obj.material ? [obj.material] : []);
    for (const mat of mats) {
      if (!mat || !mat.color || seen.has(mat)) continue;
      seen.add(mat);
      if (!mat.userData._eclipseBaseColor) mat.userData._eclipseBaseColor = mat.color.clone();
      const base = mat.userData._eclipseBaseColor;
      // LOD glyphs are emissive so they read clearly in a bulk load; that has
      // to dim in shadow too, or an eclipsed glyph would keep glowing.
      if (mat.emissive && mat.userData._eclipseBaseEmissive === undefined) {
        mat.userData._eclipseBaseEmissive = (mat.emissiveIntensity === undefined) ? 1 : mat.emissiveIntensity;
      }
      if (sunlit) {
        mat.color.copy(base);
        if (mat.emissive && mat.userData._eclipseBaseEmissive !== undefined) {
          mat.emissiveIntensity = mat.userData._eclipseBaseEmissive;
        }
      } else {
        mat.color.copy(base).multiplyScalar(ECLIPSE_DARK).lerp(ECLIPSE_TINT, 0.18);
        if (mat.emissive && mat.userData._eclipseBaseEmissive !== undefined) {
          mat.emissiveIntensity = mat.userData._eclipseBaseEmissive * ECLIPSE_DARK;
        }
      }
    }
  });
}

// Per-frame pass over active satellites, throttled -- the umbra test is cheap
// but there's no need to re-evaluate every satellite every single frame.
let _lastEclipseCheckMs = 0;
function updateEclipseShading() {
  const nowPerf = performance.now();
  if (nowPerf - _lastEclipseCheckMs < 250) return;
  _lastEclipseCheckMs = nowPerf;
  const t = getCurrentSimMs();
  for (const inst of activeSatellites.values()) {
    if (!inst.model || inst.entry.mode === 'fixed') continue;
    const eci = eciAtMs(inst, t);
    if (!eci) continue;
    applyEclipseShading(inst, satelliteSunlit(eci, t));
  }
}

// Single-mesh stand-in used once the active count passes SIMPLE_MODEL_ABOVE.
// One octahedron, one material, no greebles or panels -- ~1 draw call instead
// of ~4-21, which is what makes a 1000-satellite load practical. Colour still
// distinguishes object type so a bulk load stays readable at a glance.
function buildSimpleGlyph(type, size) {
  const color = type === 'debris' ? 0xc98a6b : (type === 'rocket' ? 0x9aa4b2 : 0xffe066);
  const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.22, shininess: 20 });
  const mesh = new THREE.Mesh(new THREE.OctahedronGeometry(size * 0.5, 0), mat);
  mesh.userData.simpleGlyph = true;
  return mesh;
}

function addSatellite(key, explicitEntry) {
  if (activeSatellites.has(key)) return; // already active, nothing to do
  const entry = explicitEntry || SATELLITE_CATALOG[key];
  if (!entry) return;
  if (entry.mode === 'metadata') return 'metadata';
  if (activeSatellites.size >= MAX_ACTIVE_SATELLITES) {
    console.warn(`Not adding ${entry.name}: at the ${MAX_ACTIVE_SATELLITES}-satellite limit. Remove one first.`);
    return false;
  }

  const inst = { entry };
  let r0, framingR;
  // LOD decision, made once up front from the current active count. `simple`
  // satellites skip both the detailed mesh and the orbit line.
  const simple = activeSatellites.size >= SIMPLE_MODEL_ABOVE;
  const wantOrbitLine = activeSatellites.size < ORBIT_LINE_ABOVE;
  inst.simple = simple;

  if (entry.mode === 'fixed') {
    // JWST special case -- see the long comment on SATELLITE_CATALOG for
    // why this doesn't go through SGP4 propagation at all. Shown
    // at a fixed, real-distance illustrative position along the
    // anti-sunward direction (JWST orbits L2, on the far side of Earth
    // from the Sun) rather than a fabricated orbit.
    const illustrativeDistKm = 1500000; // real Sun-Earth L2 distance scale; camera uses logarithmic depth so no compression is needed
    inst.fixedDistKm = illustrativeDistKm;
    r0 = illustrativeDistKm;
    framingR = illustrativeDistKm;
    inst.periodMin = null;
  } else if (entry.mode === 'state') {
    inst.stateTrack = entry.stateVector;
    const pv0 = stateVectorStateAtMs(inst, getCurrentSimMs());
    if (!pv0) return false;
    inst.periodMin = inst.stateTrack.t.length > 1
      ? (inst.stateTrack.t[inst.stateTrack.t.length - 1] - inst.stateTrack.t[0]) / 60000
      : null;
    if (wantOrbitLine) inst.orbitLine = buildStateVectorOrbitPath(inst.stateTrack, 0xffffff);
    r0 = Math.hypot(pv0.position.x, pv0.position.y, pv0.position.z);
    framingR = Math.max(...inst.stateTrack.r.map(p => Math.hypot(p[0], p[1], p[2])));
  } else if (entry.mode === 'tle') {
    const satrec = satellite.twoline2satrec(entry.tle1, entry.tle2);
    inst.satrec = satrec;
    inst.periodMin = computeOrbitPeriodMinutes(satrec);
    // Sample from the CURRENT sim time, not the TLE's own epoch -- SGP4
    // models real perturbations (J2 nodal precession chief among them),
    // so a path sampled from a stale epoch drifts out of sync with the
    // satellite's actual current position as the real orbital plane
    // precesses. See refreshOrbitLineIfNeeded() below, which keeps this
    // in sync as sim time keeps advancing after this initial draw.
    const drawMs = getCurrentSimMs();
    if (wantOrbitLine) {
      inst.orbitLine = buildRealOrbitPath(satrec, drawMs, inst.periodMin, 0xffffff);
      inst.lastOrbitDrawMs = drawMs;
      inst.lastOrbitPlaneNormal = computeOrbitPlaneNormal(satrec, drawMs);
    }
    const pv0 = tleStateAtMs(satrec, drawMs);
    if (!pv0) return false;
    r0 = Math.sqrt(pv0.position.x ** 2 + pv0.position.y ** 2 + pv0.position.z ** 2);
    framingR = r0; // near-circular in practice for every TLE-driven entry here, so r0 alone frames it fine
  } else {
    return false;
  }

  if (inst.orbitLine) scene.add(inst.orbitLine);

  // Keep display glyphs a fixed size independent of orbit radius. The
  // cislunar orbit should look like a distant orbit with a small spacecraft,
  // not a huge model scaled up by its apogee distance.
  const modelSize = SATELLITE_GLYPH_SIZE_KM;
  inst.model = simple
    ? buildSimpleGlyph(entry.type, modelSize)
    : buildModelForType(entry.type, modelSize, entry.name);
  scene.add(inst.model);

  // Floating HTML label -- projected from the satellite's 3D position each
  // frame (see updateLabels). HTML overlay rather than an in-scene sprite:
  // crisper text at any zoom and trivially styleable for the leadership-
  // facing look.
  const labelEl = document.createElement('div');
  labelEl.className = 'sat-label';
  const nameSpan = document.createElement('span');
  nameSpan.textContent = entry.name;
  const subSpan = document.createElement('span');
  subSpan.className = 'sat-label-sub';
  labelEl.appendChild(nameSpan);
  labelEl.appendChild(subSpan);
  const labelLayer = document.getElementById('label-layer');
  if (labelLayer) labelLayer.appendChild(labelEl);
  inst.labelEl = labelEl;
  inst.labelSubEl = subSpan; // updated each frame with live altitude/speed

  inst.r0 = r0;
  inst.framingR = framingR;

  activeSatellites.set(key, inst);
  updateOneSatellitePosition(key); // position immediately, don't wait for the next animate() frame
  refreshAttitudeControl();
  if (groundTracksEnabled) attachGroundTrack(inst); // honor the toggle for satellites added while it's on
  return true;
}

function removeSatellite(key) {
  const inst = activeSatellites.get(key);
  if (!inst) return;
  detachGroundTrack(inst);
  clearConjunctionsForKey(key);
  if (key === passTargetKey) { passEvents = []; passTargetKey = null; removeInViewLine(); renderPassResults(); }
  if (inst.model) scene.remove(inst.model);
  if (inst.orbitLine) scene.remove(inst.orbitLine);
  if (inst.labelEl && inst.labelEl.parentNode) inst.labelEl.parentNode.removeChild(inst.labelEl);
  activeSatellites.delete(key);
  refreshAttitudeControl();
}

// Frames the camera to fit whichever active orbit is largest, so adding a
// far-out satellite (e.g. TESS) alongside a LEO one doesn't leave it
// outside the view -- an inherent tradeoff of real orbital scale
// differences, not a bug: the LEO satellite will look tiny by comparison,
// same as it would in reality from that vantage point.
function reframeCamera() {
  if (activeSatellites.size === 0) return; // keep whatever framing was last set
  let maxFramingR = 0;
  for (const inst of activeSatellites.values()) maxFramingR = Math.max(maxFramingR, inst.framingR);
  camDist = Math.max(R_EARTH_KM * 1.3, maxFramingR * 2.1);
}

// WGS84 geodetic readout (lat/lon/height above the ellipsoid) via satellite.js.
// This is the trusted, correct sub-satellite / altitude computation; the code
// only remaps to a sphere for the *rendered* globe, but the numbers shown to
// the user should be the true geodetic ones.
function geodeticReadout(eci, dateMs) {
  const gmst = satellite.gstime(new Date(dateMs));
  const gd = satellite.eciToGeodetic(eci, gmst); // {longitude, latitude, height} in rad / km
  return {
    latDeg: satellite.degreesLat(gd.latitude),
    lonDeg: satellite.degreesLong(gd.longitude),
    altKm: gd.height,
  };
}

function fmtLat(d) { return Math.abs(d).toFixed(1) + '\u00B0' + (d >= 0 ? 'N' : 'S'); }
function fmtLon(d) { return Math.abs(d).toFixed(1) + '\u00B0' + (d >= 0 ? 'E' : 'W'); }

function updateOneSatellitePosition(key) {
  const inst = activeSatellites.get(key);
  if (!inst || !inst.model) return;
  const entry = inst.entry;

  if (entry.mode === 'fixed') {
    // Position is recomputed each frame from the current sun direction
    // (not frozen at add-time) -- the real Sun's direction slowly changes
    // over time, and now that sunDirection is computed from the actual
    // date rather than a fixed hardcoded vector, JWST's anti-sunward
    // position should track that rather than staying stuck wherever the
    // sun happened to be when you added it.
    inst.model.position.copy(sunDirection).multiplyScalar(-inst.fixedDistKm);
    // No meaningful "velocity" for a fixed illustrative point; orient using
    // the sun direction instead (sunshield-facing-Sun is the one real,
    // stable fact about JWST's attitude worth showing).
    const zHat = sunDirection.clone(); // "nadir"-equivalent axis points sunward
    let seed = new THREE.Vector3(0, 1, 0);
    if (Math.abs(seed.dot(zHat)) > 0.9) seed = new THREE.Vector3(1, 0, 0);
    const xHat = new THREE.Vector3().crossVectors(seed, zHat).normalize();
    const yHat = new THREE.Vector3().crossVectors(zHat, xHat);
    inst.model.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(xHat, yHat, zHat));
    return;
  }

  if (entry.mode === 'tle' || entry.mode === 'state') {
    if (entry.mode === 'tle' && !inst.satrec) return;
    const nowMs = getCurrentSimMs();
    const pv = entry.mode === 'state'
      ? stateVectorStateAtMs(inst, nowMs)
      : tleStateAtMs(inst.satrec, nowMs);
    if (!pv || !pv.position) {
      inst.model.visible = false;
      if (inst.labelEl) inst.labelEl.style.display = 'none';
      return;
    }
    inst.model.visible = true;
    const pos = eciToScene(pv.position);
    const vel = eciToScene(pv.velocity);
    // Live readouts. Altitude/lat/lon come from satellite.js's WGS84 geodetic
    // conversion (eciToGeodetic), NOT geocentric radius minus a spherical mean
    // radius -- the ellipsoid's ~21 km equator-to-pole difference makes the
    // naive version wrong by up to that much near the poles. Speed is the ECI
    // speed magnitude (the eciToScene remap is a pure rotation, so |vel| is
    // unchanged).
    const g = geodeticReadout(pv.position, nowMs);
    inst.liveAltKm = g.altKm;
    inst.liveLatDeg = g.latDeg;
    inst.liveLonDeg = g.lonDeg;
    inst.liveSpeedKmS = vel.length();
    if (quaternionAttitudeEnabled && pv.quaternion) {
      applyQuaternionOrientation(inst.model, pos, pv.quaternion);
    } else {
      applyOrbitOrientation(inst.model, pos, vel);
    }
  }
}

function updateAllActiveSatellitePositions() {
  for (const key of activeSatellites.keys()) updateOneSatellitePosition(key);
}

// Projects each active satellite's 3D world position to 2D screen space and
// positions its floating HTML label there. Hides the label when the
// satellite is off-screen or occluded behind Earth (a naive projection
// would otherwise float "ISS" over the wrong side of the globe when it's
// on the far side). Occlusion test: if the satellite is farther from the
// camera than the point where the camera's ray to it grazes Earth, it's
// behind the planet.
const _labelProjV = new THREE.Vector3();
function updateLabels() {
  if (!labelsEnabled) return;
  const w = renderer.domElement.clientWidth;
  const h = renderer.domElement.clientHeight;
  const camPos = camera.position;
  const _nowLabelText = performance.now();
  // With many satellites active, floating labels overlap into an unreadable mass
  // and the per-frame DOM writes dominate the frame. Above the threshold, hide
  // them all (once) and skip the work; the info panel still lists everything.
  if (activeSatellites.size > LABEL_DECLUTTER_ABOVE) {
    if (!_labelsDecluttered) {
      for (const inst of activeSatellites.values()) if (inst.labelEl) inst.labelEl.style.display = 'none';
      _labelsDecluttered = true;
    }
    return;
  }
  _labelsDecluttered = false;
  for (const inst of activeSatellites.values()) {
    const el = inst.labelEl;
    if (!el) continue;
    if (!inst.model.visible) { el.style.display = 'none'; continue; }
    const worldPos = inst.model.position;

    // Occlusion via proper ray-sphere intersection: the satellite is hidden
    // only when Earth lies fully between it and the camera -- i.e. the
    // segment from camera to satellite enters the Earth sphere AND the
    // satellite is beyond where it exits. (A naive "does the ray pass near
    // Earth's center" test wrongly hides low satellites like ISS, whose
    // line of sight always grazes within Earth's radius near the surface.)
    const dir = _labelProjV.copy(worldPos).sub(camPos);
    const segLen = dir.length();
    dir.multiplyScalar(1 / segLen); // normalize in place
    const b = 2 * camPos.dot(dir);
    const c = camPos.dot(camPos) - R_EARTH_KM * R_EARTH_KM;
    const disc = b * b - 4 * c;
    let occluded = false;
    if (disc > 0) {
      const sq = Math.sqrt(disc);
      const tNear = (-b - sq) / 2; // where the camera->sat ray first enters Earth's sphere
      // Occluded only if Earth is actually BETWEEN camera and satellite:
      // the ray must enter the sphere ahead of the camera (tNear > 0) and
      // before reaching the satellite (tNear < segLen). A near-side
      // satellite has its position short of tNear, so it's correctly not
      // hidden -- which is the case the previous tFar-only test got wrong.
      if (tNear > 0 && tNear < segLen) occluded = true;
    }

    const proj = _labelProjV.copy(worldPos).project(camera);
    const onScreen = proj.z < 1 && proj.x >= -1 && proj.x <= 1 && proj.y >= -1 && proj.y <= 1;
    if (occluded || !onScreen) {
      el.style.display = 'none';
      continue;
    }
    el.style.display = 'block';
    el.style.left = ((proj.x * 0.5 + 0.5) * w) + 'px';
    el.style.top = ((1 - (proj.y * 0.5 + 0.5)) * h) + 'px';

    // Live readout, throttled -- updating text every frame is wasteful and
    // makes the numbers jitter unreadably; ~4/sec still reads as live.
    if (inst.labelSubEl && _nowLabelText - (inst._lastLabelTextMs || 0) > 250) {
      if (inst.liveAltKm != null) {
        inst.labelSubEl.textContent = inst.liveAltKm.toFixed(0) + ' km  \u00B7  ' +
          fmtLat(inst.liveLatDeg) + ' ' + fmtLon(inst.liveLonDeg) + '  \u00B7  ' +
          inst.liveSpeedKmS.toFixed(2) + ' km/s';
      } else if (inst.entry.mode === 'fixed') {
        inst.labelSubEl.textContent = 'L2 · illustrative';
      }
      inst._lastLabelTextMs = _nowLabelText;
    }
  }
}

function setLabelsEnabled(on) {
  labelsEnabled = Boolean(on);
  const layer = document.getElementById('label-layer');
  const toggle = document.getElementById('labels-toggle');
  if (layer) layer.style.display = labelsEnabled ? '' : 'none';
  if (toggle) {
    toggle.textContent = labelsEnabled ? 'Hide labels' : 'Show labels';
    toggle.setAttribute('aria-pressed', String(labelsEnabled));
  }
  if (labelsEnabled) {
    _labelsDecluttered = false;
    updateLabels();
  }
}

let lastOrbitRefreshCheckMs = 0;
function refreshAllOrbitLinesIfNeeded() {
  const nowPerf = performance.now();
  if (nowPerf - lastOrbitRefreshCheckMs < 1000) return; // the check itself has a small per-satellite SGP4 cost -- no need more than ~1/sec
  lastOrbitRefreshCheckMs = nowPerf;
  for (const key of activeSatellites.keys()) refreshOrbitLineIfNeeded(key);
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

function updateInfoPanel() {
  const panel = document.getElementById('sat-info');
  if (!panel) return;
  if (activeSatellites.size === 0) {
    panel.innerHTML = `<span style="opacity:0.7">No satellites selected -- pick one or more above.</span>`;
    return;
  }

  // Full descriptive blocks read fine for a handful of satellites, but not
  // for dozens -- past a small threshold, switch to one compact line each.
  const compact = activeSatellites.size > 5;
  const blocks = [];
  for (const inst of activeSatellites.values()) {
    const entry = inst.entry;
    const name = escapeHtml(entry.name);
    if (entry.mode === 'fixed') {
      blocks.push(compact
        ? `<div class="sat-info-line"><b>${name}</b> -- ~1.5M km (illustrative)</div>`
        : `<div class="sat-info-block"><b>${name}</b><br>distance ~1.5 million km (real) &middot; illustrative position<br>` +
          `<span style="opacity:0.7">${escapeHtml(entry.note)}</span></div>`);
    } else if (entry.mode === 'metadata') {
      blocks.push(compact
        ? `<div class="sat-info-line"><b>${name}</b> -- DISCOS metadata only</div>`
        : `<div class="sat-info-block"><b>${name}</b><br>DISCOS metadata only; no TLE or state vector is available for plotting.<br>` +
          `<span style="opacity:0.7">${escapeHtml(entry.note)}</span></div>`);
    } else if (entry.mode === 'state') {
      const altKm = inst.liveAltKm == null ? 'n/a' : inst.liveAltKm.toFixed(0);
      const sampleCount = entry.stateVector && entry.stateVector.t ? entry.stateVector.t.length : 0;
      blocks.push(compact
        ? `<div class="sat-info-line"><b>${name}</b> -- alt ${altKm} km &middot; state vectors</div>`
        : `<div class="sat-info-block"><b>${name}</b><br>alt ${altKm} km &middot; ${sampleCount} supplied state-vector samples<br>` +
          `<span style="opacity:0.7">${escapeHtml(entry.note)}</span></div>`);
    } else {
      const altKm = (inst.r0 - R_EARTH_KM).toFixed(0);
      // TLE staleness: a TLE describes one epoch, and propagation error grows
      // with age (we measured ~800 km of orbit-plane drift on a 69-day-old ISS
      // TLE). Surfacing the epoch age tells the user how much to trust the
      // shown position.
      let provenance = '';
      if (entry.mode === 'tle' && inst.satrec) {
        const ageDays = (Date.now() - tleEpochToMs(inst.satrec)) / 86400000;
        const epochStr = new Date(tleEpochToMs(inst.satrec)).toISOString().slice(0, 10);
        const stale = ageDays > 14;
        const ageStr = `TLE epoch ${epochStr} (${ageDays.toFixed(0)} d old${stale ? ', aging' : ''})`;
        provenance = `<br><span style="opacity:0.7;${stale ? 'color:#e0a840;' : ''}">${ageStr}</span>`;
      }
      blocks.push(compact
        ? `<div class="sat-info-line"><b>${name}</b> -- alt ${altKm} km &middot; ${inst.periodMin.toFixed(0)} min</div>`
        : `<div class="sat-info-block"><b>${name}</b><br>alt ${altKm} km &middot; period ${inst.periodMin.toFixed(1)} min<br>` +
          `<span style="opacity:0.7">${escapeHtml(entry.note)}</span>${provenance}</div>`);
    }
  }
  panel.innerHTML = compact ? blocks.join('') : blocks.join('<hr class="sat-info-sep">');
  refreshPassTargetOptions(); // keep the pass-prediction target list in sync with the active set
}

// SSA analysis toggles (ground tracks now; conjunction/pass-prediction later
// will slot in alongside). Kept separate from the time/search setup so the
// analysis feature set can grow without tangling the core viewer controls.
function setupAnalysisControls() {
  const gt = document.getElementById('ground-track-toggle');
  if (gt) gt.addEventListener('change', () => setGroundTracksEnabled(gt.checked));
  const attitude = document.getElementById('attitude-toggle');
  if (attitude) attitude.addEventListener('change', () => setQuaternionAttitudeEnabled(attitude.checked));
  refreshAttitudeControl();
  const labels = document.getElementById('labels-toggle');
  if (labels) {
    labels.addEventListener('click', () => setLabelsEnabled(!labelsEnabled));
    setLabelsEnabled(true);
  }

  const screenBtn = document.getElementById('conj-screen-btn');
  const winInput = document.getElementById('conj-window');
  const thrInput = document.getElementById('conj-threshold');
  if (screenBtn) {
    screenBtn.addEventListener('click', () => {
      conjunctionWindowHours = Math.min(168, Math.max(1, Number(winInput && winInput.value) || 24));
      conjunctionThresholdKm = Math.max(0.1, Number(thrInput && thrInput.value) || 10);
      const orig = screenBtn.textContent;
      screenBtn.disabled = true;
      screenBtn.textContent = 'Screening...';
      // Defer so the "Screening..." state paints before the synchronous search runs.
      setTimeout(() => {
        const res = screenConjunctions();
        renderConjunctionResults(res);
        screenBtn.disabled = false;
        screenBtn.textContent = orig;
      }, 30);
    });
  }

  const passBtn = document.getElementById('pass-compute-btn');
  if (passBtn) {
    refreshPassTargetOptions();
    passBtn.addEventListener('click', () => {
      const lat = Number(document.getElementById('pass-lat').value);
      const lon = Number(document.getElementById('pass-lon').value);
      const minEl = Number(document.getElementById('pass-minel').value);
      const win = Math.min(168, Math.max(1, Number(document.getElementById('pass-window').value) || 24));
      const target = document.getElementById('pass-target').value;
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) { renderPassResults({ status: 'no_site' }); return; }
      passMinElevationDeg = Number.isFinite(minEl) ? Math.max(0, Math.min(89, minEl)) : 10;
      setObserver(lat, lon, 0);
      const orig = passBtn.textContent;
      passBtn.disabled = true; passBtn.textContent = 'Computing...';
      setTimeout(() => {
        const res = computePasses(target, win);
        renderPassResults(res);
        passBtn.disabled = false; passBtn.textContent = orig;
        if (cloudEnabled && res.status === 'ok') updateCloudAnnotations(); // async cloud overlay, non-blocking
      }, 30);
    });
  }

  const cloudToggle = document.getElementById('cloud-toggle');
  if (cloudToggle) {
    cloudToggle.addEventListener('change', () => {
      cloudEnabled = cloudToggle.checked;
      updateCloudAnnotations(); // fetch (or clear) and re-render whatever passes are shown
    });
  }

  const conjExport = document.getElementById('conj-export-btn');
  if (conjExport) conjExport.addEventListener('click', exportConjunctionsCsv);
  const passExport = document.getElementById('pass-export-btn');
  if (passExport) passExport.addEventListener('click', exportPassesCsv);
}

function setupControls() {
  const dom = renderer.domElement;
  dom.addEventListener('mousedown', e => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    camTheta -= dx * 0.005;
    camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, camPhi - dy * 0.005));
  });
  dom.addEventListener('wheel', e => {
    e.preventDefault();
    camDist *= (1 + e.deltaY * 0.001);
    camDist = Math.max(R_EARTH_KM * 1.15, Math.min(R_EARTH_KM * 150, camDist));
  }, { passive: false });

  let touchLastX = 0, touchLastY = 0, touchDist = 0;
  dom.addEventListener('touchstart', e => {
    if (e.touches.length === 1) { touchLastX = e.touches[0].clientX; touchLastY = e.touches[0].clientY; }
    else if (e.touches.length === 2) { touchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }
  });
  dom.addEventListener('touchmove', e => {
    e.preventDefault();
    if (e.touches.length === 1) {
      const dx = e.touches[0].clientX - touchLastX, dy = e.touches[0].clientY - touchLastY;
      touchLastX = e.touches[0].clientX; touchLastY = e.touches[0].clientY;
      camTheta -= dx * 0.005;
      camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, camPhi - dy * 0.005));
    } else if (e.touches.length === 2) {
      const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
      camDist *= (1 + (touchDist - d) * 0.003);
      camDist = Math.max(R_EARTH_KM * 1.15, Math.min(R_EARTH_KM * 150, camDist));
      touchDist = d;
    }
  }, { passive: false });
}

function onResize() {
  const container = document.getElementById('scene-container');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  camera.position.set(
    camTarget.x + camDist * Math.sin(camPhi) * Math.cos(camTheta),
    camTarget.y + camDist * Math.cos(camPhi),
    camTarget.z + camDist * Math.sin(camPhi) * Math.sin(camTheta)
  );
  camera.lookAt(camTarget);
  if (starfield) starfield.position.copy(camera.position);

  const simDate = new Date(getCurrentSimMs());
  sunDirection.copy(eciToScene(computeSunDirectionEci(simDate)));
  sunLight.position.copy(sunDirection).multiplyScalar(100000);
  updateCelestialMarkers(simDate);
  updateEarthRotation(simDate);
  cloudDriftAccumulator += 0.00006;

  if (performance.now() - lastClockDisplayUpdateMs > 250) { // no need to touch the DOM every single frame for a text readout
    const clockEl = document.getElementById('time-clock');
    if (clockEl) clockEl.textContent = simDate.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    lastClockDisplayUpdateMs = performance.now();
  }
  updateAllActiveSatellitePositions();
  updateSubPointMarkers();       // frame-exact nadir dots (cheap)
  updateEclipseShading();        // darken satellites inside Earth's shadow (throttled)
  refreshAllOrbitLinesIfNeeded();
  refreshGroundTracks();         // rebuild painted trails on a throttled cadence
  updateConjunctionLines();      // live risk-colored connectors near each TCA
  updateInViewLine();            // live site->satellite connector while above the horizon
  updateLabels();
  renderer.render(scene, camera);
}

init();
