import {esc, evidenceText} from './readable.js?v=studies-final-20261020';
import {assetUrl as url} from './asset-url.js';
function sourceLink(ref) {
  const label = esc(ref.label || ref.path || ref.url || 'Source');
  if (/^https?:\/\//.test(ref.url || '')) return `<a href="${esc(ref.url)}" target="_blank" rel="noopener">${label}</a>`;
  if (ref.path) return `<a href="${url(ref.path)}" download>${label}</a>`;
  return `<span>${label}</span>`;
}
function pictureGroups(groups = []) {
  return groups.map(group => `<details class="evidence-picture-group"><summary>${esc(group.title || 'Pictures')} · ${(group.images || []).length}</summary><div class="evidence-pictures">${(group.images || []).map(item => `<figure><a href="${url(item.path)}" target="_blank" rel="noopener" aria-label="Enlarge ${esc(item.caption || 'Evidence picture')}"><img src="${url(item.path)}" loading="lazy" alt="${esc(item.caption || 'Evidence picture')}"></a>${item.caption ? `<figcaption>${esc(item.caption)}</figcaption>` : ''}</figure>`).join('')}</div></details>`).join('');
}
export function designEvidence(items = []) {
  return `<div class="design-evidence-list">${items.map(item => `<section class="design-evidence-item"><div class="meta"><span>${esc(item.kind || 'Evidence')}</span></div><h5>${esc(item.title)}</h5>${evidenceText(item.observation, 2)}${item.measurements?.length ? `<div class="matrix"><table><thead><tr><th>Measurement</th><th>Value</th><th>Unit</th><th>Condition</th></tr></thead><tbody>${item.measurements.map(m => `<tr><td>${esc(m.name)}</td><td>${esc(m.value)}</td><td>${esc(m.unit)}</td><td>${esc(m.condition || '')}</td></tr>`).join('')}</tbody></table></div>` : ''}${item.conditions ? `<details><summary>Setup & conditions</summary><dl>${Object.entries(item.conditions).map(([k,v])=>`<dt>${esc(k.replaceAll('_',' '))}</dt><dd>${esc(typeof v === 'string' ? v : JSON.stringify(v))}</dd>`).join('')}</dl></details>` : ''}${item.limitations ? `<details><summary>Limits of this evidence</summary>${evidenceText(item.limitations)}</details>` : ''}${pictureGroups(item.image_groups)}<ul class="source-links">${(item.source_refs || []).map(ref=>`<li>${sourceLink(ref)}</li>`).join('')}</ul>${(item.clip_ids || []).map(id => `<button type="button" data-design-video="${esc(id)}">Watch ${esc(id.split(':').at(-1))}</button>`).join('')}</section>`).join('')}</div>`;
}
