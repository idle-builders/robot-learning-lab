// Remote read-only dashboard. Pages load this module instead of app.js. It answers the
// dashboard's /api/* calls from the index written by infra/dashboard/web_export.py, maps
// design files to files/ in the bucket (served by the pod), refuses writes, and then
// starts the normal app.
const manifest = await (await fetch('/snapshot/manifest.json', {cache: 'no-cache'})).json();
const nativeFetch = globalThis.fetch.bind(globalThis);
const json = (value, status = 200) => new Response(JSON.stringify(value), {status, headers: {'Content-Type': 'application/json'}});
const missing = () => json({error: 'Only on the lab computer.'}, 404);
const file = name => name ? nativeFetch(`/${name}`) : Promise.resolve(missing());
let currentShas = new Map();

function assetName(path, sha) {
  const version = sha || new URLSearchParams(globalThis.assetVersionQuery?.(path) || '').get('sha');
  return (version && manifest.assets[`${path}@${version}`]) || manifest.assets[path]
    || (currentShas.has(path) && manifest.assets[`${path}@${currentShas.get(path)}`]);
}
globalThis.snapshotAssetUrl = path => {
  const name = assetName(path);
  return name ? `/${name}` : `/snapshot/missing.html?path=${encodeURIComponent(path)}`;
};

function resultId(q) {
  const id = q.get('id'), versionId = q.get('version_id'), skill = q.get('skill') || q.get('skill_id'), version = q.get('version');
  if (versionId) return manifest.version_ids[versionId];
  if (skill) return version ? manifest.skill_versions[`${skill}:${version}`] : manifest.skills[skill];
  if (!id) return null;
  return /^\d+$/.test(id) ? manifest.version_ids[id] ?? id : manifest.aliases[id] ?? id;
}

async function preview(q) {
  const path = q.get('path') || '';
  const name = assetName(path, q.get('sha'));
  if (!name || !/\.(json|jsonl|csv|tsv|md|txt|log)$/i.test(path)) return json({error: 'No text preview for this file'}, 404);
  const text = await (await nativeFetch(`/${name}`)).text();
  return json({text: text.slice(0, 120_000), truncated: text.length > 120_000});
}

globalThis.fetch = async (input, init = {}) => {
  const url = new URL(input instanceof Request ? input.url : input, location.href);
  if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return nativeFetch(input, init);
  if ((init.method || 'GET').toUpperCase() !== 'GET') return json({error: 'This is a read-only web snapshot. Make changes in the local dashboard.'}, 405);
  const q = url.searchParams;
  switch (url.pathname) {
    case '/api/results': return file(manifest.lists[q.get('dashboard') || 'learning']);
    case '/api/result': {
      const response = await file(manifest.results[resultId(q)]);
      if (response.ok) currentShas = new Map(((await response.clone().json()).assets || []).map(a => [a.path, a.sha256]));
      return response;
    }
    case '/api/video': {
      const key = q.get('clip_id') || q.get('id') || '';
      const clip = manifest.clips[manifest.clip_nums[key] ?? key];
      return clip ? json(clip) : json({error: 'Video not uploaded'}, 404);
    }
    case '/api/preview': return preview(q);
    case '/api/design-reviews': return manifest.design_reviews[q.get('result')] ? file(manifest.design_reviews[q.get('result')]) : missing();
    case '/api/design-bom':
      if (q.get('revision')) return json({error: 'Earlier BOM revisions are in the local dashboard'}, 404);
      return file(manifest.design_boms[q.get('result')]);
    case '/api/robot-bom': return q.has('export') ? missing() : file(manifest.robot_bom);
    case '/api/sessions': return json({sessions: []});
    default: return missing();
  }
};

document.documentElement.classList.add('web-snapshot');
await import('/app.js');
