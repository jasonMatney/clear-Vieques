#!/usr/bin/env node
// Headless-Chrome screenshot / smoke-test driver for ClearVieques. No dependencies (Node >= 22: global fetch + WebSocket).
//
//   node tools/shot.mjs <url> <out.png> [--w 1600] [--h 900] [--dpr 1] [--wait 15000] [--eval "js expression"] [--chrome "/path/to/chrome"]
//
// Works for http://localhost:8137/ (node tools/serve.mjs) AND for file:///…/index.html (proving the demo runs straight from disk).
// It waits (real time) until CV.app.ready, optionally evaluates a JS expression (e.g. to move the camera), prints console errors,
// and saves a PNG of the page via Page.captureScreenshot. Uses WebGPU through ANGLE/Metal (macOS) — pass --chrome for other paths.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = process.argv.slice(2);
const url = args[0], out = args[1];
if (!url || !out) { console.error('usage: node tools/shot.mjs <url> <out.png> [--w N --h N --dpr N --wait ms --eval js --chrome path]'); process.exit(2); }
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const W = +opt('w', 1600), H = +opt('h', 900), DPR = +opt('dpr', 1), WAIT = +opt('wait', 15000), EVAL = opt('eval', null);
const chrome = opt('chrome', process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const port = 9300 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cv-chrome-'));

const proc = spawn(chrome, ['--headless=new', `--remote-debugging-port=${port}`, '--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist',
  `--user-data-dir=${profile}`, `--window-size=${W},${H}`, '--hide-scrollbars', '--no-first-run', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cleanup = () => { try { proc.kill('SIGKILL'); } catch {} try { fs.rmSync(profile, { recursive: true, force: true }); } catch {} };
process.on('exit', cleanup);

async function main() {
  let target;
  for (let i = 0; i < 60; i++) { // wait for the debugging endpoint
    try { const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); target = list.find(t => t.type === 'page'); if (target) break; } catch {}
    await sleep(250);
  }
  if (!target) throw new Error('Chrome did not expose a debugging target');
  const ws = new WebSocket(target.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map(); const logs = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    else if (d.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(d.params.type)) logs.push(`[${d.params.type}] ` + d.params.args.map(a => a.value ?? a.description ?? '').join(' '));
    else if (d.method === 'Runtime.exceptionThrown') logs.push('[exception] ' + (d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text));
  };
  const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Page.enable'); await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
  await send('Page.navigate', { url });
  const evalJs = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result;
  const t0 = Date.now(); let ready = false;
  while (Date.now() - t0 < WAIT) { const r = await evalJs('!!(window.CV && CV.app && CV.app.ready)'); if (r.result?.value) { ready = true; break; } await sleep(300); }
  if (ready && EVAL) { const r = await evalJs(EVAL); if (r.exceptionDetails) logs.push('[eval error] ' + JSON.stringify(r.exceptionDetails.exception?.description)); else console.log('eval ->', JSON.stringify(r.result?.value)); }
  if (ready) await sleep(2500); // let waves / caustics settle
  const info = await evalJs('(window.CV && CV.app && CV.app.ready) ? JSON.stringify({ fps: +CV.app.stats.fps.toFixed(1), frames: CV.app.waves.frame, gpu: navigator.gpu ? "webgpu" : "none", url: location.href }) : "not ready"');
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log(ready ? 'ready' : 'NOT READY within ' + WAIT + ' ms', '|', info.result?.value, '| saved', out);
  if (logs.length) console.log('console problems:\n  ' + [...new Set(logs)].slice(0, 12).join('\n  '));
  ws.close();
  return ready ? 0 : 1;
}
main().then(c => { cleanup(); process.exit(c); }).catch(e => { console.error(e); cleanup(); process.exit(1); });
