#!/usr/bin/env node
// Cache-busting for GitHub Pages (which lets browsers keep every file for 10 minutes): stamps each <script src> in index.html with ?v=<hash of
// that script's contents>. A page always loads the scripts it was published with, so an update can no longer pair a fresh index.html with a
// stale cached script. Run before committing changes to any script:  node tools/stamp.mjs   (--check: exit 1 if index.html is out of date)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const file = path.join(root, 'index.html');
const html = fs.readFileSync(file, 'utf8');
const out = html.replace(/<script src="([^"?]+)(\?v=[0-9a-f]+)?"><\/script>/g, (_, src) => {
  const h = crypto.createHash('sha1').update(fs.readFileSync(path.join(root, src))).digest('hex').slice(0, 8);
  return `<script src="${src}?v=${h}"></script>`;
});
if (process.argv.includes('--check')) {
  if (out !== html) { console.error('index.html script stamps are out of date: run node tools/stamp.mjs'); process.exit(1); }
  console.log('index.html script stamps are up to date');
} else {
  fs.writeFileSync(file, out);
  console.log(out === html ? 'index.html already up to date' : 'index.html script stamps updated');
}
