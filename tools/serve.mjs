// Tiny dev server: static files + POST /__save?name=x.png (screenshot upload used by the test harness).
// Usage: node tools/serve.mjs [port]   ->  http://localhost:8137/
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.argv[2] || process.env.PORT || 8137);
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.wgsl': 'text/plain; charset=utf-8', '.md': 'text/plain; charset=utf-8', '.bin': 'application/octet-stream', '.i16': 'application/octet-stream' };

http.createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  res.setHeader('Cache-Control', 'no-store');
  if (req.method === 'POST' && url.pathname === '/__save') {
    const name = path.basename(url.searchParams.get('name') || 'shot.png');
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      fs.mkdirSync(path.join(root, 'shots'), { recursive: true });
      fs.writeFileSync(path.join(root, 'shots', name), Buffer.concat(chunks));
      res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('ok ' + name);
    });
    return;
  }
  let p = decodeURIComponent(url.pathname);
  if (p.endsWith('/')) p += 'index.html';
  const file = path.join(root, p);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404); return res.end('not found: ' + p); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(buf);
  });
}).listen(port, () => console.log(`ClearVieques dev server on http://localhost:${port}/`));
