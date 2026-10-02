// ClearVieques — application: boot, frame loop, adaptive resolution, sites and modes, controls, debug/capture API.
(function () {
  'use strict';
  const CV = window.CV, $ = id => document.getElementById(id);
  const canvas = $('gl');
  const ui = new CV.UI();

  CV.log.onError = (msg) => { const e = $('errs'); e.style.display = 'block'; e.textContent += msg + '\n\n'; e.scrollTop = e.scrollHeight; };
  const fatal = (err) => {
    console.error(err);
    $('loader').style.display = 'none';
    const msg = (err && err.message) || String(err);
    // only a missing WebGPU gets the "WebGPU is required" heading; anything else is a load or runtime failure, which a reload often fixes
    if (!/WebGPU|adapter/i.test(msg)) $('fatalTitle').textContent = 'The demo could not start';
    $('fatal').style.display = 'flex'; $('fatalMsg').textContent = /WebGPU|adapter/i.test(msg) ? msg : msg + ' — try reloading the page.';
  };

  const app = CV.app = {
    p: {}, paused: false, time: 0, siteId: 'caracas', night: false, fixedTime: null,
    quality: 1, stats: { fps: 0, ms: 0 }, exposureBase: 1.7, airExt: 1.7e-4, deepGain: 1.0, vignette: 0.2,
    paddle: { on: false, x: 0, z: 0, active: 0, last: null },
  };

  // Slider defaults for a site in a mode. In night mode the "sun" slider is the moon's elevation.
  function defaultsFor(site, night) {
    return { sunElev: night ? site.moonElev : site.sunElev, exposure: 0, seaOffset: 0, energy: site.energy, wind: site.wind, turbidity: site.turbidity, moon: site.id === 'mosquito' ? 0.25 : 0.5 };
  }
  const moonFmt = v => (v < 0.06 || v > 0.94) ? 'new' : (Math.abs(v - 0.5) < 0.06 ? 'full' : (v < 0.5 ? 'waxing ' : 'waning ') + Math.round(100 * (1 - Math.cos(CV.Night.phaseAngle(v))) / 2) + '%');

  async function boot() {
    const params = new URLSearchParams(location.search);
    ui.loadMsg('Requesting a WebGPU device…');
    const gpu = app.gpu = await CV.initGPU(canvas);
    CV.log.info('GPU', gpu.info.vendor, gpu.info.architecture, 'f32filter', gpu.f32filter, 'timestamps', gpu.timestamps);

    if (params.has('site') && CV.SITES[params.get('site')]) app.siteId = params.get('site');
    const site = app.site = CV.SITES[app.siteId];
    app.night = params.has('mode') ? params.get('mode') === 'night' : site.defaultMode === 'night';
    app.p = defaultsFor(site, app.night);
    ui.loadMsg('Decoding NOAA topo-bathy…');
    await new Promise(r => setTimeout(r, 30));
    const terrain = app.terrain = new CV.Terrain(gpu, app.siteId, site);
    app.meta = terrain.meta;

    ui.loadMsg('Building the atmosphere…');
    const sky = app.sky = new CV.Sky(gpu);
    sky.update(app.p.sunElev * CV.D2R);

    ui.loadMsg('Compiling water kernels…');
    const mip = app.mip = new CV.MipGen(gpu);
    const waves = app.waves = new CV.Waves(gpu, mip);
    const ripples = app.ripples = new CV.Ripples(gpu, terrain);
    const sceneLayout = CV.makeSceneLayout(gpu);
    const caustics = app.caustics = new CV.Caustics(gpu, sceneLayout);
    await Promise.all([mip.ready, waves.ready, ripples.ready, caustics.ready]);
    caustics.attachMip(mip);

    ui.loadMsg('Compiling optics…');
    const renderer = app.renderer = new CV.Renderer(gpu, terrain, sky, waves, ripples, caustics, mip, sceneLayout);
    const canopy = app.canopy = new CV.Canopy(gpu, sceneLayout);
    await Promise.all([renderer.ready, canopy.ready]);
    ui.loadMsg('Growing the forest…');
    canopy.bake(terrain, bakeRes());
    if (params.has('quality')) app.quality = CV.clamp(parseInt(params.get('quality')) || 0, 0, 2);
    applyTier(app.quality);
    if (params.has('msaa')) renderer.setSamples(parseInt(params.get('msaa')) || 1);
    if (params.has('scale')) { renderer.scale = parseFloat(params.get('scale')); app.fixedScale = true; }
    else setRung((window.devicePixelRatio || 1) > 1.5 ? 2 : 4);
    if (params.has('debug')) renderer.debugView = parseInt(params.get('debug')) || 0;
    if (params.get('taa') === '0') renderer.taa.enabled = false;     // temporal anti-aliasing off (diagnostics / comparison)
    if (params.has('bloom')) renderer.bloomStrength = parseFloat(params.get('bloom')) || 0;   // bloom mix (0 = off)
    if (params.get('nocrown') === '1') renderer.noCrown = true;      // diagnostics: flat-shaded terrain without the canopy displacement
    // on a phone the panel would cover most of the scene: start with the controls tucked away (one tap on "Show controls" brings them back)
    const narrow = window.matchMedia && window.matchMedia('(max-width: 560px)').matches;
    if (params.get('ui') === '0' || (narrow && params.get('ui') !== '1')) { document.body.classList.add('hidden'); $('hide').textContent = 'Show controls ↗'; }
    if (params.has('t')) app.fixedTime = parseFloat(params.get('t')) || 0;   // freeze the wave clock (reproducible frames)

    const camera = app.camera = new CV.Camera(canvas, terrain, () => app.terrain.seaLevel + app.p.seaOffset);
    camera.set(app.night ? site.heroNight : site.hero);
    camera.onClick = (x, y, w, h) => {
      const hit = camera.pickWater(x, y, w, h);
      if (hit) ripples.splash(hit[0], hit[2], 0.17, -0.05, 1.0);
    };
    camera.onPaddle = (x, y, w, h, down) => paddleMove(x, y, w, h, down);
    setupUI();
    applyWaveParams();
    syncModeUI();
    ui.loaded();
    app.ready = true;
    if (params.get('hold') !== '1') requestAnimationFrame(loop);      // ?hold=1: boot, then wait — lets a harness capture the very first frame
  }
  const bakeRes = () => ({ samLin: app.renderer.samLin, samRep: app.renderer.samRep, noise: app.renderer.noise });

  // ------------------------------------------------------------------------------------------------ UI wiring
  function setupUI() {
    const d = app.p;
    ui.addSlider('ctlLight', { id: 'sunElev', label: 'Sun elevation', min: 4, max: 88, step: 0.5, value: d.sunElev, fmt: v => v.toFixed(0) + '°' });
    ui.addSlider('ctlLight', { id: 'exposure', label: 'Exposure', min: -2.5, max: 2.5, step: 0.05, value: d.exposure, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) + ' EV' });
    ui.addSlider('ctlLight', { id: 'moon', label: 'Moon phase (night)', min: 0, max: 1, step: 0.01, value: d.moon, fmt: moonFmt });
    ui.addSlider('ctlWater', { id: 'seaOffset', label: 'Depth offset', min: -1.5, max: 2.0, step: 0.01, value: d.seaOffset, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) + ' m' });
    ui.addSlider('ctlWater', { id: 'energy', label: 'Wave energy', min: 0, max: 1, step: 0.01, value: d.energy, fmt: v => Math.round(v * 100) + '%' });
    ui.addSlider('ctlWater', { id: 'wind', label: 'Wind', min: 0, max: 14, step: 0.1, value: d.wind, fmt: v => v.toFixed(1) + ' m/s' });
    ui.addSlider('ctlWater', { id: 'turbidity', label: 'Turbidity · Jerlov', min: 0, max: 1, step: 0.005, value: d.turbidity, fmt: v => CV.jerlov(v).name });
    ui.onInput = (id, v) => {
      app.p[id] = v;
      if (id === 'sunElev') app.sky.update(v * CV.D2R);
      if (id === 'energy' || id === 'wind') applyWaveParams();
    };

    $('btnPause').onclick = () => { app.paused = !app.paused; $('btnPause').classList.toggle('on', app.paused); $('btnPause').textContent = app.paused ? 'Resume' : 'Pause'; syncStatus(); };
    $('btnReset').onclick = () => { resetAll(); };
    $('btnShot').onclick = () => { heroView(); };
    $('btnMode').onclick = () => toggleMode();
    $('btnDay').onclick = () => setNight(false);
    $('btnNight').onclick = () => setNight(true);
    $('btnPaddle').onclick = () => setPaddle(!app.paddle.on);
    document.querySelectorAll('.tab').forEach(t => { t.onclick = () => switchSite(t.dataset.site); });
    $('hide').onclick = () => { document.body.classList.toggle('hidden'); $('hide').textContent = document.body.classList.contains('hidden') ? 'Show controls ↗' : 'Hide controls ↗'; };
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'KeyO') toggleMode();
      else if (e.code === 'KeyH') $('hide').click();
      else if (e.code === 'KeyP') $('btnPause').click();
      else if (e.code === 'KeyR') resetAll();
      else if (e.code === 'KeyV') heroView();
      else if (e.code === 'KeyX') setPaddle(!app.paddle.on);
      else if (e.code === 'KeyN') setNight(!app.night);
    });
    window.addEventListener('resize', resizeCanvas); resizeCanvas();
  }

  // Reflect the current site + mode in the panel: tabs, Day/Night, slider labels and ranges, status line, names, coordinates.
  function syncModeUI() {
    const site = app.site, night = app.night;
    document.body.classList.toggle('night', night);
    document.querySelectorAll('.tab').forEach(t => { t.classList.toggle('on', t.dataset.site === app.siteId); t.disabled = false; t.title = ''; });
    $('btnDay').classList.toggle('on', !night); $('btnNight').classList.toggle('on', night);
    $('btnDay').disabled = false; $('btnNight').disabled = false; $('btnDay').title = 'Sunlit water'; $('btnNight').title = 'Moonlit water (N)';
    ui.reconfigure('sunElev', night ? { label: 'Moon elevation', min: 6, max: 80, value: app.p.sunElev } : { label: 'Sun elevation', min: 4, max: 88, value: app.p.sunElev });
    ui.show('moon', night);
    for (const k of Object.keys(app.p)) ui.set(k, app.p[k]);
    $('siteName').textContent = site.name; $('siteSub').textContent = site.sub;
    const m = app.meta; if (m.lat0 !== undefined) ui.coords(`${CV.fmtLat(m.lat0)} · ${CV.fmtLon(m.lon0)}`);
    $('btnPaddle').classList.toggle('on', app.paddle.on);
    syncStatus();
  }
  function syncStatus() {
    if (app.paused) return ui.status('PAUSED');
    ui.status(app.night ? (app.site.biolum ? 'NIGHT / BIOLUMINESCENT LAGOON' : 'NIGHT / MOONLIT WATER') : 'LIVE / SHALLOW WATER');
  }

  function toggleMode() { const c = app.camera; c.setMode(c.mode === 'fly' ? 'orbit' : 'fly'); $('btnMode').textContent = c.mode === 'fly' ? 'Fly' : 'Orbit'; }
  function heroView() {
    const s = app.site, d = defaultsFor(s, app.night);
    app.camera.set(app.night ? s.heroNight : s.hero);
    app.p.sunElev = d.sunElev; app.p.moon = d.moon; syncModeUI(); app.sky.update(app.p.sunElev * CV.D2R);
  }
  function resetAll() {
    app.p = defaultsFor(app.site, app.night); syncModeUI();
    app.sky.update(app.p.sunElev * CV.D2R); applyWaveParams(); heroView();
  }
  // Day <-> night on the current site: only the light changes (sun -> moon); the sea state and exposure the user set are kept.
  function setNight(on) {
    if (on === app.night) return;
    app.night = on;
    const d = defaultsFor(app.site, on);
    app.p.sunElev = d.sunElev; app.p.moon = d.moon;
    syncModeUI(); app.sky.update(app.p.sunElev * CV.D2R);
  }
  function applyWaveParams() {
    const s = app.site;
    app.waves.set({ wind: app.p.wind, energy: app.p.energy, windDirDeg: s.windFrom, swellDirDeg: s.swellFrom, fetchBase: s.fetchBase, fetchGain: s.fetchGain, swellScale: s.swell });
  }
  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(64, Math.floor(canvas.clientWidth * dpr)), h = Math.max(64, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  // Switch location: decode the new site's terrain, bake its canopy, rebind everything that references the terrain, apply the site's defaults.
  async function switchSite(id, night) {
    if (app.switching || !CV.SITES[id] || (id === app.siteId && night === undefined)) return;
    app.switching = true;
    const site = CV.SITES[id], ld = $('loader');
    ld.style.display = 'flex'; ld.classList.remove('done'); $('loadMsg').textContent = `Reading ${site.name}…`;
    await new Promise(r => setTimeout(r, 60));
    try {
      const old = app.terrain, terrain = new CV.Terrain(app.gpu, id, site);
      app.canopy.bake(terrain, bakeRes());
      app.renderer.setTerrain(terrain); app.ripples.setTerrain(terrain); app.camera.terrain = terrain;
      app.terrain = terrain; app.site = site; app.siteId = id; app.meta = terrain.meta;
      app.night = night === undefined ? site.defaultMode === 'night' : night;
      app.p = defaultsFor(site, app.night);
      app.camera.set(app.night ? site.heroNight : site.hero);
      app.caustics.tick = 0; app.paddle.last = null;
      applyWaveParams(); app.sky.update(app.p.sunElev * CV.D2R); syncModeUI();
      old.destroy();
    } catch (e) { CV.log.error('site switch:', e); }
    ui.loaded(); app.switching = false;
  }
  app.switchSite = switchSite; app.setNight = setNight;

  // Quality tiers shed fixed per-frame cost (the caustic map dominates it). hi = splat into the 2048^2 level and filter down; every = refresh interval.
  const TIERS = [
    { NP: 480, hi: false, every: 2, msaa: 1 },
    { NP: 720, hi: false, every: 1, msaa: 1 },
    { NP: 1120, hi: true, every: 1, msaa: 4 },   // MSAA only pays off when the internal resolution is close to native
  ];
  // Adaptive-resolution ladder: [tier, render scale]. One rung per step keeps texture re-creation rare; the edge-adaptive upscaler hides the low rungs.
  const LADDER = [[0, 0.40], [0, 0.50], [0, 0.60], [0, 0.70], [1, 0.70], [1, 0.80], [1, 0.90], [2, 0.90], [2, 1.00]];
  function setRung(i) {
    i = CV.clamp(i, 0, LADDER.length - 1); app.rung = i;
    const [q, sc] = LADDER[i];
    if (q !== app.quality) { app.quality = q; applyTier(q); }
    app.renderer.scale = sc; hist.length = 0;
    const t = app.renderer.timer; t.samples = 0; t.avg = 0;
  }
  function applyTier(q) {
    const t = TIERS[q], c = app.caustics;
    c.NP = t.NP; c.hi = t.hi; c.every = t.every; c.tick = 0; app.renderer.setSamples(t.msaa); hist.length = 0;
  }
  function setQuality(q) {
    q = CV.clamp(q, 0, TIERS.length - 1); if (q === app.quality) return; app.quality = q; applyTier(q);
  }
  app.setQuality = setQuality;

  // ------------------------------------------------------------------------------------------------ paddle and wake
  function setPaddle(on) {
    app.paddle.on = on; app.paddle.last = null; app.paddle.active = 0;
    app.camera.paddleMode = on; $('btnPaddle').classList.toggle('on', on);
    canvas.style.cursor = on ? 'crosshair' : '';
  }
  // Pointer over the water in paddle mode: place the blade; while the button is down, stroke (interpolated so fast drags stay continuous).
  function paddleMove(px, py, w, h, down) {
    const hit = app.camera.pickWater(px, py, w, h), P = app.paddle;
    if (!hit) { P.active = 0; P.last = null; return; }
    P.x = hit[0]; P.z = hit[2]; P.active = 1;
    if (down) {
      const l = P.last, r = app.ripples;
      if (l) {
        const dx = P.x - l[0], dz = P.z - l[1], d = Math.hypot(dx, dz), n = Math.max(1, Math.min(24, Math.ceil(d / 0.14)));
        for (let i = 1; i <= n; i++) r.splash(l[0] + dx * i / n, l[1] + dz * i / n, 0.24, -0.028, 1.0);
      } else r.splash(P.x, P.z, 0.24, -0.04, 1.0);
      P.last = [P.x, P.z];
    } else P.last = null;
  }
  // The camera is a boat: moving over water pushes a bow wave a few metres ahead and leaves agitation under itself.
  let camPrev = null;
  function cameraWake(dt) {
    const cam = app.camera, p = cam.pos, sea = app.terrain.seaLevel + app.p.seaOffset;
    if (camPrev && dt > 0 && app.night && app.site.biolum > 0 && app.terrain.heightAt(p[0], p[2]) < sea - 0.15 && p[1] < sea + 8) {
      const v = Math.hypot(p[0] - camPrev[0], p[2] - camPrev[1]) / dt;
      if (v > 0.35) {
        const f = cam.fwd, hl = Math.hypot(f[0], f[2]) || 1, k = CV.clamp(v / 4, 0.25, 1), fx = f[0] / hl, fz = f[2] / hl, rx = -fz, rz = fx;
        const R = app.ripples, at = (d, lat) => [p[0] + fx * d + rx * lat, p[2] + fz * d + rz * lat];
        R.splash(...at(5.5, 0), 1.1 + 0.6 * k, -0.012 * k, 0.5 + 0.45 * k);                      // bow wave
        R.splash(...at(3.8, 1.9), 0.8, -0.008 * k, 0.4 + 0.4 * k); R.splash(...at(3.8, -1.9), 0.8, -0.008 * k, 0.4 + 0.4 * k);   // the V of the wedge
        R.splash(p[0], p[2], 1.4, -0.006 * k, 0.5 * k);
      }
    }
    camPrev = [p[0], p[2]];
  }

  // ------------------------------------------------------------------------------------------------ state for one frame
  function buildState(dt, w, h) {
    const cam = app.camera, m = cam.matrices(w / h), p = app.p, tm = app.terrain, site = app.site, night = app.night;
    const f = cam.fwd, hl = Math.hypot(f[0], f[2]) || 1, fx = f[0] / hl, fz = f[2] / hl;
    const sea = tm.seaLevel + p.seaOffset;
    const cc = [cam.pos[0] + fx * 12, cam.pos[2] + fz * 12];
    const groundH = tm.heightAt(cc[0], cc[1]);
    const P = app.paddle;
    return {
      cam: m, time: app.fixedTime !== null ? app.fixedTime : app.time, dt,
      sunAz: night ? site.moonAz : site.sunAz, sunElev: p.sunElev, exposure: p.exposure, exposureLin: app.exposureBase * Math.pow(2, p.exposure),
      seaOffset: p.seaOffset, turbidity: p.turbidity, airExt: app.airExt, deepGain: app.deepGain, vignette: app.vignette,
      focus: [cam.pos[0] + fx * 14, cam.pos[2] + fz * 14], causticCenter: cc, meanDepth: CV.clamp(sea - groundH, 0.6, 14),
      night, moon: p.moon, matSet: site.matSet, biolum: site.biolum, cloudCover: night ? 0.10 : undefined,
      paddle: [P.x, P.z, 0.24, P.on && P.active ? 1 : 0],
    };
  }

  // ------------------------------------------------------------------------------------------------ loop
  let last = performance.now(), frameN = 0, underRun = 0, lastRungChange = 0; const hist = [], bad = {};
  function loop(now) {
    requestAnimationFrame(loop);
    const dtRaw = (now - last) / 1000; last = now;
    const dt = Math.min(dtRaw, 0.05);
    if (app.switching) return;
    if (!app.paused) app.time += dt;
    app.camera.update(dt);
    if (!app.paused) cameraWake(dt);
    resizeCanvas();
    const gpu = app.gpu;
    const state = buildState(app.paused ? 0 : dt, canvas.width, canvas.height);
    try { app.renderer.frame(state, gpu.ctx.getCurrentTexture().createView(), canvas.width, canvas.height); }
    catch (e) { CV.log.error('frame:', e); return; }
    hist.push(dtRaw * 1000); if (hist.length > 60) hist.shift();
    frameN++;
    // adaptive resolution: walk the ladder to keep GPU time under ~13.5 ms (headroom for 60 fps; raw frame time is vsync-quantised and can't show headroom)
    if (frameN % 30 === 0 && !app.fixedScale && frameN > 90) {
      const r = app.renderer, gt = r.timer.enabled && r.timer.samples > 10 ? r.timer.avg : null;
      const s = [...hist].sort((a, b) => a - b), med = s[s.length >> 1] || 16.7, tnow = performance.now();
      const over = (gt !== null && gt > 13.5) || med > 19, under = (gt !== null ? gt < 8.5 : med < 14.6) && med < 15.5;
      if (over) {
        underRun = 0;
        if (tnow - lastRungChange > 700 && app.rung > 0) { bad[app.rung] = tnow; setRung(app.rung - (gt !== null && gt > 24 ? 2 : 1)); lastRungChange = tnow; }
      } else if (under) {
        if (++underRun >= 6 && app.rung < LADDER.length - 1 && !(bad[app.rung + 1] && tnow - bad[app.rung + 1] < 25000)) { setRung(app.rung + 1); lastRungChange = tnow; underRun = 0; }
      } else underRun = 0;
    }
    if (frameN % 15 === 0 && hist.length > 0) {                       // (a rung change just cleared the history: keep the last reading)
      const s = [...hist].sort((a, b) => a - b), med = s[s.length >> 1];
      app.stats.fps = 1000 / med; app.stats.ms = med;
      const c = app.camera.pos, m = app.meta;
      let pos = '';
      if (m.lat0 !== undefined) {
        const lat = m.lat0 - c[2] / 110700, lon = m.lon0 + c[0] / (111320 * Math.cos(m.lat0 * CV.D2R));
        pos = ` · ${CV.fmtLat(lat)} ${CV.fmtLon(lon)} · ${(c[1] - (app.terrain.seaLevel + app.p.seaOffset)).toFixed(1)} m`;
      }
      const gt = app.renderer.timer.enabled && app.renderer.timer.samples > 10 ? ` · gpu ${app.renderer.timer.avg.toFixed(1)} ms` : '';
      ui.hud(`${Math.round(app.stats.fps)} fps${gt} · ${app.renderer.width}×${app.renderer.height} · scale ${app.renderer.scale.toFixed(2)} · tier ${app.quality}${pos}`);
    }
  }

  // ------------------------------------------------------------------------------------------------ debug / capture API (used by the screenshot harness)
  app.setCamera = (pose) => app.camera.set(pose);
  app.setParams = (o) => { for (const [k, v] of Object.entries(o)) { app.p[k] = v; ui.set(k, v); if (k === 'sunElev') app.sky.update(v * CV.D2R); } applyWaveParams(); };
  // Per-pass GPU timings (ms) for one frame at the current size/scale. Note the 'scene' pass is the only one that scales with resolution.
  app.profile = async (frames = 8) => {
    const acc = {}; const w = canvas.width, h = canvas.height;
    for (let i = 0; i < frames; i++) {
      const prof = new CV.Profiler(app.gpu);
      CV.prof = prof;                                   // scoped to this synchronous frame call only (the rAF loop must not see it)
      try { app.renderer.frame(buildState(1 / 60, w, h), app.gpu.ctx.getCurrentTexture().createView(), w, h); } finally { CV.prof = null; }
      const r = await prof.results();
      if (i >= 2) for (const [k, v] of Object.entries(r)) acc[k] = (acc[k] || 0) + v / (frames - 2);
      await new Promise(r => setTimeout(r, 30));
    }
    const total = Object.values(acc).reduce((a, b) => a + b, 0);
    return { ms: Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, +v.toFixed(2)])), total: +total.toFixed(2), scale: app.renderer.scale, size: [app.renderer.width, app.renderer.height] };
  };
  app.shot = async (name = 'shot.png', w = 1920, h = 1080) => {
    const st = buildState(0, w, h);
    const { data } = await app.renderer.capture(st, w, h);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    cv.getContext('2d').putImageData(new ImageData(data, w, h), 0, 0);
    const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
    const res = await fetch('/__save?name=' + encodeURIComponent(name), { method: 'POST', body: blob });
    return { ok: res.ok, bytes: blob.size, name };
  };

  boot().catch(fatal);
})();
