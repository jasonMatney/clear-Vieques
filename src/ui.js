// ClearVieques — control panel wiring (dark glass panel in the Clearwater family).
(function () {
  'use strict';
  const CV = window.CV;
  const $ = id => document.getElementById(id);

  CV.UI = class UI {
    constructor() { this.defs = {}; this.onInput = null; }

    addSlider(container, def) {
      const el = document.createElement('label');
      el.className = 'ctl'; el.id = 'ctl-' + def.id;
      el.innerHTML = `<span class="lab">${def.label}</span><output></output><input type="range" min="${def.min}" max="${def.max}" step="${def.step}" value="${def.value}">`;
      $(container).appendChild(el);
      const input = el.querySelector('input'), out = el.querySelector('output');
      const paint = () => {
        const v = parseFloat(input.value);
        out.textContent = def.fmt ? def.fmt(v) : String(v);
        input.style.setProperty('--p', ((v - def.min) / (def.max - def.min) * 100) + '%');
      };
      input.addEventListener('input', () => { paint(); if (this.onInput) this.onInput(def.id, parseFloat(input.value)); });
      paint();
      this.defs[def.id] = { def, input, paint, el };
    }
    set(id, v) { const d = this.defs[id]; if (!d) return; d.input.value = v; d.paint(); }
    get(id) { return parseFloat(this.defs[id].input.value); }
    show(id, on) { const d = this.defs[id]; if (d) d.el.classList.toggle('off', !on); }

    status(text) { $('statusText').textContent = text; }
    coords(text) { $('coords').textContent = text; }
    hud(text) { $('hud').textContent = text; }
    loaded() { $('loader').classList.add('done'); setTimeout(() => { $('loader').style.display = 'none'; }, 900); }
    loadMsg(m) { $('loadMsg').textContent = m; }
  };

  CV.fmtLat = (lat) => Math.abs(lat).toFixed(4) + '° ' + (lat >= 0 ? 'N' : 'S');
  CV.fmtLon = (lon) => Math.abs(lon).toFixed(4) + '° ' + (lon >= 0 ? 'E' : 'W');
})();
