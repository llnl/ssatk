// Headless smoke test for the built viewer.
//
//   cd src/ && python build_satellite_viewer.py && node final_check.js
//
// Needs Node + Playwright + a headless Chromium (`npx playwright install
// chromium`). Loads the built HTML, exercises the core invariants and both SSA
// features, and asserts zero console errors. Prints "errorCount: 0" and exits 0
// on success; exits 1 on any failure.
const { chromium } = require('playwright');
const path = require('path');

// The built HTML sits next to this script (assemble.py writes it there); fall
// back to one directory up so either layout works.
const fs = require('fs');
const CANDIDATES = [
  process.env.SSATK_SATELLITE_VIEWER_HTML,
  path.resolve(__dirname, 'satellite_3d_scene_threejs.html'),
  path.resolve(__dirname, '..', 'satellite_3d_scene_threejs.html'),
].filter(Boolean);
const HTML = CANDIDATES.find(p => fs.existsSync(p));
if (!HTML) {
  console.error('ERROR: satellite_3d_scene_threejs.html not found. Run "python build_satellite_viewer.py" first.');
  console.error('Looked in:\n  ' + CANDIDATES.join('\n  '));
  process.exit(1);
}

function assert(cond, msg) { if (!cond) { console.error('FAIL:', msg); process.exitCode = 1; } }

(async () => {
  const browser = await chromium.launch({
    args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

  await page.goto('file://' + HTML, { waitUntil: 'load' });
  await page.waitForTimeout(3000);

  // --- Baseline -------------------------------------------------------------
  const base = await page.evaluate(() => ({
    catalog: Object.keys(SATELLITE_CATALOG).length,
    active: activeSatellites.size,
    defaultName: [...activeSatellites.values()][0]?.entry.name || null,
    hasEarth: !!earthMesh,
  }));
  assert(base.active === 0, `no external TLE should be active by default, got ${base.active}`);
  assert(base.hasEarth, 'earthMesh missing');

  await page.evaluate(() => addSatellite('__ISS', {
    name: 'ISS (test)', type: 'ISS', mode: 'tle',
    tle1: '1 25544U 98067A   26133.42450843  .00004829  00000+0  95080-4 0  9993',
    tle2: '2 25544  51.6310 112.1825 0007522  54.1994 305.9693 15.49203550566361',
  }));

  // --- Database formats -----------------------------------------------------
  const formats = await page.evaluate(() => {
    const iss = activeSatellites.get('__ISS').entry;
    const csv = parseCsvDatabase(
      `name,line1,line2\n"ISS, ZARYA","${iss.tle1}","${iss.tle2}"\n`
    );
    return {
      csv: normalizeSatelliteRecord(csv[0]),
      hdf5ReaderLoaded: typeof hdf5 !== 'undefined',
    };
  });
  assert(formats.csv && formats.csv.name === 'ISS, ZARYA', 'quoted CSV record did not normalize');
  assert(!formats.hdf5ReaderLoaded, 'browser HDF5 reader should not be embedded');

  // --- Ground tracks --------------------------------------------------------
  await page.click('#ground-track-toggle');
  await page.waitForTimeout(400);
  const gt = await page.evaluate(() => {
    const inst = [...activeSatellites.values()][0];
    const satDir = inst.model.position.clone().normalize();
    const mDir = inst.subPointMarker.position.clone().normalize();
    return {
      line: !!inst.groundTrackLine,
      marker: !!inst.subPointMarker,
      markerAngleDeg: Math.acos(Math.min(1, satDir.dot(mDir))) * 180 / Math.PI,
    };
  });
  assert(gt.line && gt.marker, 'ground track line/marker not created');
  assert(gt.markerAngleDeg < 0.01, `nadir marker off by ${gt.markerAngleDeg.toFixed(4)} deg`);

  // --- State-vector Hermite interpolation ----------------------------------
  const hermite = await page.evaluate(() => {
    const radius = 7000;
    const omega = Math.sqrt(SSAPY_CONSTANTS.MU_EARTH_KM3_S2 / (radius ** 3));
    const dt = 60;
    const angle = omega * dt;
    const track = {
      t: [0, dt * 1000],
      r: [[radius, 0, 0], [radius * Math.cos(angle), radius * Math.sin(angle), 0]],
      v: [[0, radius * omega, 0], [-radius * omega * Math.sin(angle), radius * omega * Math.cos(angle), 0]],
    };
    const state = stateVectorStateAtMs({ stateTrack: track }, dt * 500);
    const radiusErrorM = Math.abs(Math.hypot(state.position.x, state.position.y) - radius) * 1000;
    return { radiusErrorM, outside: stateVectorStateAtMs({ stateTrack: track }, -1) };
  });
  assert(hermite.radiusErrorM < 1, `Hermite midpoint error ${hermite.radiusErrorM} m`);
  assert(hermite.outside === null, 'state track outside its span should be hidden');

  // --- Conjunction screening (verified vs brute force) ----------------------
  const conj = await page.evaluate(() => {
    const iss = activeSatellites.get('__ISS').entry;
    addSatellite('__A', { ...iss, name: 'A' });
    addSatellite('__B', { ...iss, name: 'B' });
    conjunctionWindowHours = 3;
    conjunctionThresholdKm = 5000;
    screenConjunctions();
    const ev = conjunctionEvents.find(e =>
      (e.keyA === '__A' && e.keyB === '__B') || (e.keyA === '__B' && e.keyB === '__A'));
    // brute-force ground truth at 1 s resolution
    const now = getCurrentSimMs(), A = activeSatellites.get('__A'), B = activeSatellites.get('__B');
    let bf = Infinity;
    for (let t = now; t <= now + 3 * 3600e3; t += 1000) {
      const a = eciAtMs(A, t), b = eciAtMs(B, t);
      bf = Math.min(bf, Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z));
    }
    return { found: !!ev, miss: ev ? ev.missKm : null, bruteForce: bf };
  });
  assert(conj.found, 'engineered conjunction not found');
  assert(conj.found && Math.abs(conj.miss - conj.bruteForce) < 0.5,
    `miss distance ${conj.miss} vs brute force ${conj.bruteForce}`);

  // --- Geodetic readout (WGS84 via satellite.js) ----------------------------
  const geo = await page.evaluate(() => {
    const inst = [...activeSatellites.values()].find(i => i.entry.mode === 'tle');
    const now = getCurrentSimMs();
    const g = geodeticReadout(satellite.propagate(inst.satrec, new Date(now)).position, now);
    const gd = satellite.eciToGeodetic(satellite.propagate(inst.satrec, new Date(now)).position, satellite.gstime(new Date(now)));
    return { alt: g.altKm, libHeight: gd.height, latOk: Math.abs(g.latDeg) <= 90, lonOk: Math.abs(g.lonDeg) <= 180 };
  });
  assert(Math.abs(geo.alt - geo.libHeight) < 1e-6, `geodetic altitude ${geo.alt} != lib ${geo.libHeight}`);
  assert(geo.latOk && geo.lonOk, 'geodetic lat/lon out of range');

  // --- Pass predictions vs brute-force elevation ----------------------------
  const pass = await page.evaluate(() => {
    const key = [...activeSatellites.keys()].find(k => activeSatellites.get(k).entry.mode === 'tle');
    const inst = activeSatellites.get(key);
    passMinElevationDeg = 10;
    setObserver(37.68, -121.77, 0);
    computePasses(key, 24);
    const mask = 10 * Math.PI / 180;
    const elev = t => { const la = lookAnglesAt(inst, t); return la ? la.elevation : -Math.PI / 2; };
    const now = getCurrentSimMs();
    let bf = 0, inP = false;
    for (let t = now; t <= now + 24 * 3600e3; t += 5000) {
      const up = elev(t) > mask;
      if (up && !inP) bf++;
      inP = up;
    }
    return { computed: passEvents.length, bruteForce: bf };
  });
  assert(pass.computed === pass.bruteForce, `pass count ${pass.computed} != brute force ${pass.bruteForce}`);

  // --- Optical visibility (illumination geometry) ---------------------------
  const visi = await page.evaluate(() => {
    const now = getCurrentSimMs();
    const s = computeSunDirectionEci(new Date(now));
    const along = { x: s.x * 7000, y: s.y * 7000, z: s.z * 7000 };
    const behind = { x: -s.x * 7000, y: -s.y * 7000, z: -s.z * 7000 };
    const labels = passEvents.map(p => p.visibility);
    return {
      sunlitAlong: satelliteSunlit(along, now),   // toward sun -> lit
      eclipsedBehind: !satelliteSunlit(behind, now), // anti-sun low -> shadowed
      allLabelsValid: labels.every(l => l === 'visible' || l === 'daylight' || l === 'eclipsed'),
    };
  });
  assert(visi.sunlitAlong, 'satellite toward the sun should be sunlit');
  assert(visi.eclipsedBehind, 'satellite directly behind Earth should be eclipsed');
  assert(visi.allLabelsValid, 'every pass should have a valid visibility label');
  console.log(`visibility: umbra test OK, labels valid (${pass.computed} passes)`);

  // --- Eclipse shading round-trip (regression guard) ------------------------
  // Several models share one material across multiple meshes (Hubble/GPS/JWST
  // panels and mirrors). The eclipse baseline must be cached per-MATERIAL, not
  // per-mesh, or the second mesh captures an already-darkened colour and the
  // material stays permanently dim. This asserts colours restore exactly after
  // repeated dark/lit cycles.
  const eclipse = await page.evaluate(() => {
    let worst = 0, shared = 0;
    for (const key of ['__ISS', 'jwst']) {
      removeSatellite(key); addSatellite(key);
      const inst = activeSatellites.get(key);
      const counts = new Map(), mats = [];
      inst.model.traverse(o => {
        if (o.material && o.material.color) { mats.push(o.material); counts.set(o.material, (counts.get(o.material) || 0) + 1); }
      });
      for (const [, c] of counts) if (c > 1) shared++;
      const before = mats.map(m => m.color.clone());
      for (let c = 0; c < 5; c++) { inst._eclipseState = undefined; applyEclipseShading(inst, false); applyEclipseShading(inst, true); }
      mats.forEach((m, i) => {
        worst = Math.max(worst, Math.abs(m.color.r - before[i].r) + Math.abs(m.color.g - before[i].g) + Math.abs(m.color.b - before[i].b));
      });
      removeSatellite(key);
    }
    return { worstDrift: worst, sharedMaterialsSeen: shared };
  });
  assert(eclipse.sharedMaterialsSeen > 0, 'expected at least one shared material to exercise the guard');
  assert(eclipse.worstDrift < 1e-6, `eclipse shading did not restore colours (drift ${eclipse.worstDrift})`);
  console.log(`eclipse shading: restores exactly after 5 cycles (drift ${eclipse.worstDrift}, ${eclipse.sharedMaterialsSeen} shared materials)`);

  console.log(`baseline: ${JSON.stringify(base)}`);
  console.log(`groundTrack markerAngleDeg: ${gt.markerAngleDeg.toExponential(2)}`);
  console.log(`conjunction miss ${conj.miss && conj.miss.toFixed(4)} km vs brute force ${conj.bruteForce.toFixed(4)} km`);
  console.log(`geodetic altitude ${geo.alt.toFixed(4)} km (matches lib)`);
  console.log(`pass predictions: ${pass.computed} passes (brute force ${pass.bruteForce})`);
  console.log('errorCount:', errors.length);
  if (errors.length) { errors.slice(0, 10).forEach(e => console.log('  ' + e)); process.exitCode = 1; }

  await browser.close();
  if (process.exitCode) console.error('\nSMOKE TEST FAILED'); else console.log('\nSMOKE TEST PASSED');
})();
