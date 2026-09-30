// ClearVieques — camera: fly (WASD + drag-look) and orbit modes, terrain/water collision, click picking.
// World frame: x east, y up, z south. yaw = compass bearing (deg, clockwise from north), pitch = degrees above horizon.
(function () {
  'use strict';
  const CV = window.CV;
  const clamp = CV.clamp;

  CV.Camera = class Camera {
    constructor(canvas, terrain, getSea) {
      this.canvas = canvas; this.terrain = terrain; this.getSea = getSea;
      this.pos = [0, 3, 0]; this.yaw = 0; this.pitch = 0; this.fov = 62;
      this.mode = 'fly'; this.orbit = { target: [0, 0, -30], dist: 25 };
      this.speed = 6; this.vel = [0, 0, 0]; this.keys = new Set();
      this.drag = null; this.onClick = null; this.onPaddle = null; this.paddleMode = false; this.enabled = true;
      this.minAbove = 0.55;
      this.bind();
    }

    dirFromAngles(yawDeg, pitchDeg) {
      const y = yawDeg * CV.D2R, p = pitchDeg * CV.D2R;
      return [Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)];
    }
    get fwd() { return this.dirFromAngles(this.yaw, this.pitch); }

    set(pose) {
      this.pos = pose.pos.slice(); this.yaw = pose.yaw; this.pitch = pose.pitch;
      if (pose.fov) this.fov = pose.fov;
      this.vel = [0, 0, 0];
      if (this.mode === 'orbit') this.syncOrbitFromView();
    }

    setMode(mode) {
      if (mode === this.mode) return;
      this.mode = mode;
      if (mode === 'orbit') this.syncOrbitFromView();
    }
    syncOrbitFromView() {
      const f = this.fwd, sea = this.getSea();
      let dist = 22;
      if (f[1] < -0.02) dist = clamp((sea - this.pos[1]) / f[1], 6, 400);
      this.orbit.dist = dist;
      this.orbit.target = [this.pos[0] + f[0] * dist, this.pos[1] + f[1] * dist, this.pos[2] + f[2] * dist];
    }

    // Over water: 0.4 m above the surface (so you can skim a shallow bay). Over land: 1.4 m above the ground.
    minHeight(x, z) { const g = this.terrain.heightAt(x, z), sea = this.getSea(); return g > sea - 0.05 ? Math.max(sea + 0.4, g + 1.4) : sea + 0.4; }

    bind() {
      const c = this.canvas;
      const paddle = (e, down) => { const r = c.getBoundingClientRect(); if (this.onPaddle) this.onPaddle(e.clientX - r.left, e.clientY - r.top, r.width, r.height, down); };
      c.addEventListener('pointerdown', e => {
        if (!this.enabled) return;
        c.setPointerCapture(e.pointerId);
        this.drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, t: performance.now(), moved: 0, button: e.button, shift: e.shiftKey, paddle: this.paddleMode && e.button === 0 && !e.shiftKey };
        if (this.drag.paddle) paddle(e, true);
      });
      c.addEventListener('pointermove', e => {
        const d = this.drag;
        if (this.paddleMode && (!d || d.paddle)) { paddle(e, !!d); if (d) return; }   // the blade follows the pointer; dragging strokes the water
        if (!d) return;
        if (d.paddle) return;
        const dx = e.clientX - d.x, dy = e.clientY - d.y; d.x = e.clientX; d.y = e.clientY; d.moved += Math.abs(dx) + Math.abs(dy);
        if (d.button === 2 || d.shift) { if (this.mode === 'orbit') this.pan(dx, dy); return; }
        if (d.moved < 4 && performance.now() - d.t < 250) return;
        const k = 0.16 * (this.fov / 62);
        this.yaw += dx * k; this.pitch = clamp(this.pitch - dy * k, -89, 89);
        if (this.mode === 'orbit') { this.orbitReposition(); }
      });
      const up = e => {
        const d = this.drag; this.drag = null;
        if (d && d.paddle) { paddle(e, false); return; }
        if (d && d.button === 0 && !d.shift && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 6 && performance.now() - d.t < 350 && this.onClick) {
          const r = c.getBoundingClientRect(); this.onClick(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
        }
      };
      c.addEventListener('pointerup', up); c.addEventListener('pointercancel', () => { this.drag = null; });
      c.addEventListener('contextmenu', e => e.preventDefault());
      c.addEventListener('wheel', e => {
        e.preventDefault();
        const s = Math.exp(-e.deltaY * 0.0015);
        if (this.mode === 'orbit') { this.orbit.dist = clamp(this.orbit.dist / s, 2, 900); this.orbitReposition(); }
        else this.speed = clamp(this.speed * s, 0.4, 250);
      }, { passive: false });
      window.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return; this.keys.add(e.code); });
      window.addEventListener('keyup', e => this.keys.delete(e.code));
      window.addEventListener('blur', () => this.keys.clear());
    }

    pan(dx, dy) {
      const o = this.orbit, s = o.dist * 0.0018, y = this.yaw * CV.D2R;
      const rx = Math.cos(y), rz = Math.sin(y); // right vector (horizontal)
      const fx = Math.sin(y), fz = -Math.cos(y);
      o.target[0] += (-dx * rx + dy * fx) * s; o.target[2] += (-dx * rz + dy * fz) * s;
      this.orbitReposition();
    }
    orbitReposition() {
      const o = this.orbit, f = this.fwd;
      this.pos = [o.target[0] - f[0] * o.dist, o.target[1] - f[1] * o.dist, o.target[2] - f[2] * o.dist];
      this.constrain();
    }
    constrain() {
      const h = this.minHeight(this.pos[0], this.pos[2]);
      if (this.pos[1] < h) this.pos[1] = h;
    }

    update(dt) {
      if (!this.enabled) return;
      const k = this.keys;
      let ix = 0, iy = 0, iz = 0;
      if (k.has('KeyW') || k.has('ArrowUp')) iz += 1; if (k.has('KeyS') || k.has('ArrowDown')) iz -= 1;
      if (k.has('KeyD') || k.has('ArrowRight')) ix += 1; if (k.has('KeyA') || k.has('ArrowLeft')) ix -= 1;
      if (k.has('KeyE') || k.has('Space')) iy += 1; if (k.has('KeyQ') || k.has('KeyC')) iy -= 1;
      const boost = (k.has('ShiftLeft') || k.has('ShiftRight')) ? 4 : (k.has('ControlLeft') ? 0.25 : 1);
      if (this.mode === 'fly') {
        const f = this.fwd, y = this.yaw * CV.D2R, r = [Math.cos(y), 0, Math.sin(y)];
        const want = [0, 0, 0];
        for (let i = 0; i < 3; i++) want[i] = (f[i] * iz + r[i] * ix) * this.speed * boost;
        want[1] += iy * this.speed * boost * 0.7;
        const a = 1 - Math.exp(-dt * 7);
        for (let i = 0; i < 3; i++) { this.vel[i] += (want[i] - this.vel[i]) * a; this.pos[i] += this.vel[i] * dt; }
        const h = this.minHeight(this.pos[0], this.pos[2]); if (this.pos[1] < h) { this.pos[1] = h; if (this.vel[1] < 0) this.vel[1] = 0; }
      } else {
        const o = this.orbit, y = this.yaw * CV.D2R;
        const sp = o.dist * 0.6 * boost;
        o.target[0] += (Math.sin(y) * iz + Math.cos(y) * ix) * sp * dt; o.target[2] += (-Math.cos(y) * iz + Math.sin(y) * ix) * sp * dt;
        o.dist = clamp(o.dist * Math.exp(-iy * 0.6 * dt), 2, 900);
        this.orbitReposition();
      }
    }

    matrices(aspect) {
      const f = this.fwd, up = [0, 1, 0];
      const view = CV.m4.view(this.pos, f, up);
      const proj = CV.m4.perspectiveRevZ(this.fov * CV.D2R, aspect, 0.12);
      const r = CV.v3.norm(CV.v3.cross(f, up)), u = CV.v3.cross(r, f), tanY = Math.tan(this.fov * CV.D2R / 2);
      return { view, proj, viewProj: CV.m4.mul(proj, view), pos: this.pos.slice(), fwd: f, right: r, up: u, tanY, tanX: tanY * aspect };
    }

    // Ray through a canvas pixel (CSS px) -> intersection with the water plane y = sea (null if the ray misses / hits land first)
    pickWater(px, py, w, h) {
      const m = this.matrices(w / h), nx = (px / w) * 2 - 1, ny = 1 - (py / h) * 2;
      const d = CV.v3.norm([m.fwd[0] + m.right[0] * nx * m.tanX + m.up[0] * ny * m.tanY, m.fwd[1] + m.right[1] * nx * m.tanX + m.up[1] * ny * m.tanY, m.fwd[2] + m.right[2] * nx * m.tanX + m.up[2] * ny * m.tanY]);
      if (d[1] >= -1e-4) return null;
      const sea = this.getSea(), t = (sea - this.pos[1]) / d[1];
      if (t <= 0 || t > 4000) return null;
      const hit = [this.pos[0] + d[0] * t, sea, this.pos[2] + d[2] * t];
      if (this.terrain.heightAt(hit[0], hit[2]) > sea - 0.03) return null; // land
      return hit;
    }
  };
})();
