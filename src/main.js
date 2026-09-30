// ClearVieques — application: boot, frame loop, adaptive resolution, controls, debug/capture API.
(function () {
  'use strict';
  const CV = window.CV, $ = id => document.getElementById(id);
  const canvas = $('gl');
  const ui = new CV.UI();

  CV.log.onError = (msg) => { const e = $('errs'); e.style.display = 'block'; e.textContent += msg + '\n\n'; e.scrollTop = e.scrollHeight; };
  const fatal = (err) => {
    console.error(err);
    $('loader').style.display = 'none';
    $('fatal').style.display = 'flex'; $('fatalMsg').textContent = (err && err.message) || String(err);
  };

  const DEFAULTS = { sunElev: 58, exposure: 0, seaOffset: 0, energy: 0.35, wind: 3.6, turbidity: 0.16, moon: 0.5 };
  const app = CV.app = {
    p: { ...DEFAULTS }, paused: false, time: 0, siteId: 'caracas', night: false, fixedTime: null,
    quality: 1, stats: { fps: 0, ms: 0 }, exposureBase: 1.7, airExt: 1.0e-4, deepGain: 1.0, vignette: 0.2,
  };

  async function boot() {
    const params = new URLSearchParams(location.search);
    ui.loadMsg('Requesting a WebGPU device…');
    const gpu = app.gpu = await CV.initGPU(canvas);
    CV.log.info('GPU', gpu.info.vendor, gpu.info.architecture, 'f32filter', gpu.f32filter, 'timestamps', gpu.timestamps);

    const site = app.site = CV.SITES[app.siteId];
    ui.loadMsg('Decoding NOAA topo-bathy…');
    await new Promise(r => setTimeout(r, 30));
    const terrain = app.terrain = new CV.Terrain(gpu, app.siteId);
    app.meta = terrain.meta;
    if (terrain.meta.hero) Object.assign(site, { hero: terrain.meta.hero });

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
    canopy.bake(terrain, { samLin: renderer.samLin, samRep: renderer.samRep, noise: renderer.noise });
    if (params.has('quality')) app.quality = CV.clamp(parseInt(params.get('quality')) || 0, 0, 2);
    applyTier(app.quality);
    if (params.has('msaa')) renderer.setSamples(parseInt(params.get('msaa')) || 1);
    if (params.has('scale')) { renderer.scale = parseFloat(params.get('scale')); app.fixedScale = true; }
    else setRung((window.devicePixelRatio || 1) > 1.5 ? 2 : 4);
    if (params.has('debug')) renderer.debugView = parseInt(params.get('debug')) || 0;
    if (params.get('nocrown') === '1') renderer.noCrown = true;      // diagnostics: flat-shaded terrain without the canopy displacement
    if (params.get('ui') === '0') { document.body.classList.add('hidden'); $('hide').textContent = 'Show controls ↗'; }
    if (params.has('t')) app.fixedTime = parseFloat(params.get('t')) || 0;   // freeze the wave clock (reproducible frames)

    const camera = app.camera = new CV.Camera(canvas, terrain, () => terrain.seaLevel + app.p.seaOffset);
    camera.set(site.hero);
    camera.onClick = (x, y, w, h) => {
      const hit = camera.pickWater(x, y, w, h);
      if (hit) ripples.splash(hit[0], hit[2]);
    };
    setupUI(site);
    applyWaveParams();
    ui.status('LIVE / SHALLOW WATER');
    ui.loaded();
    app.ready = true;
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------------------------------------ UI wiring
  function setupUI(site) {
    ui.addSlider('ctlLight', { id: 'sunElev', label: 'Sun elevation', min: 4, max: 88, step: 0.5, value: DEFAULTS.sunElev, fmt: v => v.toFixed(0) + '°' });
    ui.addSlider('ctlLight', { id: 'exposure', label: 'Exposure', min: -2.5, max: 2.5, step: 0.05, value: DEFAULTS.exposure, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) + ' EV' });
    ui.addSlider('ctlLight', { id: 'moon', label: 'Moon phase (night)', min: 0, max: 1, step: 0.01, value: DEFAULTS.moon, fmt: v => (v < 0.06 || v > 0.94) ? 'new' : (Math.abs(v - 0.5) < 0.06 ? 'full' : v < 0.5 ? 'waxing' : 'waning') });
    ui.show('moon', false);
    ui.addSlider('ctlWater', { id: 'seaOffset', label: 'Depth offset', min: -1.5, max: 2.0, step: 0.01, value: DEFAULTS.seaOffset, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(2) + ' m' });
    ui.addSlider('ctlWater', { id: 'energy', label: 'Wave energy', min: 0, max: 1, step: 0.01, value: DEFAULTS.energy, fmt: v => Math.round(v * 100) + '%' });
    ui.addSlider('ctlWater', { id: 'wind', label: 'Wind', min: 0, max: 14, step: 0.1, value: DEFAULTS.wind, fmt: v => v.toFixed(1) + ' m/s' });
    ui.addSlider('ctlWater', { id: 'turbidity', label: 'Turbidity · Jerlov', min: 0, max: 1, step: 0.005, value: DEFAULTS.turbidity, fmt: v => CV.jerlov(v).name });
    ui.onInput = (id, v) => {
      app.p[id] = v;
      if (id === 'sunElev') app.sky.update(v * CV.D2R);
      if (id === 'energy' || id === 'wind') applyWaveParams();
    };
    $('siteName').textContent = site.name; $('siteSub').textContent = site.sub;
    const m = app.meta; if (m.lat0 !== undefined) ui.coords(`${CV.fmtLat(m.lat0)} · ${CV.fmtLon(m.lon0)}`);

    $('btnPause').onclick = () => { app.paused = !app.paused; $('btnPause').classList.toggle('on', app.paused); $('btnPause').textContent = app.paused ? 'Resume' : 'Pause'; ui.status(app.paused ? 'PAUSED' : 'LIVE / SHALLOW WATER'); };
    $('btnReset').onclick = () => { resetAll(); };
    $('btnShot').onclick = () => { heroView(); };
    $('btnMode').onclick = () => toggleMode();
    $('hide').onclick = () => { document.body.classList.toggle('hidden'); $('hide').textContent = document.body.classList.contains('hidden') ? 'Show controls ↗' : 'Hide controls ↗'; };
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'KeyO') toggleMode();
      else if (e.code === 'KeyH') $('hide').click();
      else if (e.code === 'KeyP') $('btnPause').click();
      else if (e.code === 'KeyR') resetAll();
      else if (e.code === 'KeyV') heroView();
    });
    window.addEventListener('resize', resizeCanvas); resizeCanvas();
  }
  function toggleMode() { const c = app.camera; c.setMode(c.mode === 'fly' ? 'orbit' : 'fly'); $('btnMode').textContent = c.mode === 'fly' ? 'Fly' : 'Orbit'; }
  function heroView() { const s = app.site; app.camera.set(s.hero); app.p.sunElev = s.sunElev; ui.set('sunElev', s.sunElev); app.sky.update(s.sunElev * CV.D2R); }
  function resetAll() {
    Object.assign(app.p, DEFAULTS);
    for (const k of Object.keys(DEFAULTS)) ui.set(k, DEFAULTS[k]);
    app.sky.update(app.p.sunElev * CV.D2R); applyWaveParams(); heroView();
  }
  function applyWaveParams() {
    const s = app.site;
    app.waves.set({ wind: app.p.wind, energy: app.p.energy, windDirDeg: (s.windFrom), swellDirDeg: (s.swellFrom) });
  }
  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(64, Math.floor(canvas.clientWidth * dpr)), h = Math.max(64, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

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

  // ------------------------------------------------------------------------------------------------ state for one frame
  function buildState(dt, w, h) {
    const cam = app.camera, m = cam.matrices(w / h), p = app.p, tm = app.terrain;
    const f = cam.fwd, hl = Math.hypot(f[0], f[2]) || 1, fx = f[0] / hl, fz = f[2] / hl;
    const sea = tm.seaLevel + p.seaOffset;
    const cc = [cam.pos[0] + fx * 12, cam.pos[2] + fz * 12];
    const groundH = tm.heightAt(cc[0], cc[1]);
    return {
      cam: m, time: app.fixedTime !== null ? app.fixedTime : app.time, dt,
      sunAz: app.site.sunAz, sunElev: p.sunElev, exposure: p.exposure, exposureLin: app.exposureBase * Math.pow(2, p.exposure),
      seaOffset: p.seaOffset, turbidity: p.turbidity, airExt: app.airExt, deepGain: app.deepGain, vignette: app.vignette,
      focus: [cam.pos[0] + fx * 14, cam.pos[2] + fz * 14], causticCenter: cc, meanDepth: CV.clamp(sea - groundH, 0.6, 14),
      night: app.night, moon: p.moon,
    };
  }

  // ------------------------------------------------------------------------------------------------ loop
  let last = performance.now(), frameN = 0, underRun = 0, lastRungChange = 0; const hist = [], bad = {};
  function loop(now) {
    requestAnimationFrame(loop);
    const dtRaw = (now - last) / 1000; last = now;
    const dt = Math.min(dtRaw, 0.05);
    if (!app.paused) app.time += dt;
    app.camera.update(dt);
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
    if (frameN % 15 === 0) {
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
