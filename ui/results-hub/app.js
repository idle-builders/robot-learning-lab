import {evidenceText, briefText, shortText} from './readable.js?v=studies-final-20261020';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';
const isDesign = location.pathname.startsWith('/design');
const dashboard = isDesign ? 'design' : 'learning';
const home = isDesign ? '/design/' : '/';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const assetUrl = path => `/asset?path=${encodeURIComponent(path)}`;
let results = [], sessions = [], mode = 'published', workspace = 'All', selected = null;
let modelViewer = null;
let catalogView = null, viewGeneration = 0;
let preferences = [];
const versionOrder = (a,b) => String(b.study_version || '').localeCompare(String(a.study_version || ''), undefined, {numeric:true});
function latestStudies() {
  const map = new Map();
  for (const row of [...results].sort(versionOrder)) if (!map.has(row.study_id || row.id)) map.set(row.study_id || row.id, row);
  return [...map.values()].sort((a,b)=>(b.model_count>0)-(a.model_count>0) || designStudyName(a).localeCompare(designStudyName(b)));
}
const belongs = (row, name) => name === 'All' || (isDesign ? (row.design_sections || ['Whole robot']).includes(name) : row.workspace === name);

function records() {
  if (mode === 'archive') return sessions.map(row => ({...row, workspace: row.kind || 'Other'}));
  return isDesign ? latestStudies() : results;
}
function renderList() {
  const all = records();
  const spaces = isDesign ? [...new Set(all.flatMap(row => row.design_sections || ['Whole robot']))] : [...new Set([...(mode === 'published' ? ['Control','Perception','Planning','References','Operations'] : []), ...all.map(row => row.workspace || 'Other')])].sort();
  $('workspaces').innerHTML = ['All', ...spaces].map(name => `<button type="button" data-workspace="${escapeHtml(name)}" class="${workspace === name ? 'active' : ''}" aria-pressed="${workspace === name}">${escapeHtml(name === 'All' && isDesign ? 'All sections' : name)} <small>${all.filter(row => belongs(row, name)).length}</small></button>`).join('');
  $('workspaces').querySelectorAll('button').forEach(button => button.onclick = () => {
    workspace = button.dataset.workspace;
    $('body-map')?.querySelectorAll('[data-section]').forEach(marker => marker.classList.toggle('active', marker.dataset.section === workspace));
    const preferred = preferences.find(p => p.section === workspace);
    const current = records().find(row => row.id === selected && belongs(row, workspace));
    const first = (preferred && records().find(row => row.study_id === preferred.study_id)) || current || records().find(row => belongs(row, workspace));
    if (first) mode === 'archive' ? showSession(first.id) : showResult(first.id);
    else { selected = null; renderList(); $('detail').textContent = 'No current studies in this section. Earlier work may be retained in Archive.'; }
  });
  const query = $('search').value.trim().toLowerCase();
  const filtered = all.filter(row => belongs(row, workspace) &&
    `${row.title} ${row.area || ''} ${row.observation || ''} ${row.summary || ''} ${row.id}`.toLowerCase().includes(query));
  $('count').textContent = `${filtered.length} ${mode === 'archive' ? (filtered.length === 1 ? 'retained source' : 'retained sources') : isDesign ? (filtered.length === 1 ? 'variant study' : 'variant studies') : (filtered.length === 1 ? 'published record' : 'published records')}`;
  $('results').innerHTML = filtered.slice(0, 120).map(row => {
    const name = isDesign ? designStudyName(row) : row.title;
    const version = isDesign && row.study_version ? ` · ${row.study_version}` : '';
    return `<button type="button" title="${escapeHtml(name)}" data-id="${escapeHtml(row.id)}" class="${row.id === selected || isDesign && row.study_id && row.study_id === results.find(r=>r.id===selected)?.study_id ? 'active' : ''}">${escapeHtml(shortText(name, 105))}${isDesign && preferences.some(p=>p.study_id===row.study_id && (workspace==='All'||p.section===workspace)) ? '<b class="preferred-mark">Preferred</b>' : ''}<span>${escapeHtml(isDesign ? row.design_sections?.join(' · ') : row.workspace)}${escapeHtml(version)}${mode === 'published' && row.clip_count ? ` · ${row.uploaded_clips || 0}/${row.clip_count} videos` : ''}${mode === 'published' && row.asset_count ? ` · ${row.asset_count} files` : ''}</span></button>`;
  }).join('');
  if (filtered.length > 120) $('results').insertAdjacentHTML('beforeend', '<p class="muted">Showing 120. Search to narrow the list.</p>');
  if (isDesign) $('body-map')?.querySelectorAll('[data-section]').forEach(marker => {marker.classList.toggle('active',marker.dataset.section===workspace);marker.setAttribute('aria-pressed',String(marker.dataset.section===workspace));marker.onclick = () => {
    const button = [...$('workspaces').querySelectorAll('button')].find(b=>b.dataset.workspace===marker.dataset.section);
    if (button) button.click(); else {workspace=marker.dataset.section;renderList();$('detail').innerHTML='<h2>'+escapeHtml(workspace)+'</h2><p>No variant studies registered for this section yet.</p>'; }
  };});
  $('results').querySelectorAll('button').forEach(button => button.onclick = () => mode === 'archive' ? showSession(button.dataset.id) : showResult(button.dataset.id));
}
function renderGrades(grades) {
  // Graded assessment, 0 (unknown) to 5 (viable): never pass/fail (docs/design/evaluation-principles.md)
  if (!grades || !(grades.dimensions || []).length && !grades.overall) return '';
  const rows = (grades.dimensions || []).map(d => `<div class="grade-row"><span class="grade-dim">${escapeHtml(d.dimension)}</span><span class="grade-bar" title="${escapeHtml(d.grade)} of 5"><i style="width:${Math.max(0, Math.min(5, Number(d.grade))) * 20}%"></i></span><span class="grade-num">${escapeHtml(d.grade)}${d.grade === 0 ? ' · unknown' : ''}</span><small>${escapeHtml([d.confidence && `${d.confidence} confidence`, d.trend].filter(Boolean).join(' · '))}</small><div class="grade-reason">${escapeHtml(shortText(d.reason || '', 210))}${(d.reason || '').length > 210 ? `<details class="text-more"><summary>Full reasoning</summary>${evidenceText(d.reason)}</details>` : ''}</div></div>`).join('');
  const challenges = (grades.requirement_challenges || []).map(c => `<li>${escapeHtml(typeof c === 'string' ? c : JSON.stringify(c))}</li>`).join('');
  return `<details><summary>Graded assessment · distance to viable (0–5)</summary><div class="inside">${grades.overall ? `${briefText(grades.overall, 'Full assessment')}` : ''}<div class="grades">${rows}</div>${challenges ? `<strong>Requirement challenges</strong><ul>${challenges}</ul>` : ''}</div></details>`;
}
function renderMatrix(matrix) {
  if (!matrix) return '<p class="muted">No compact case table was published.</p>';
  const data = Array.isArray(matrix) ? matrix : Array.isArray(matrix.rows) ? matrix.rows : null;
  if (data?.length && data.every(row => row && typeof row === 'object' && !Array.isArray(row))) {
    const columns = [...new Set(data.flatMap(row => Object.keys(row)))].slice(0, 24);
    return `<div class="matrix"><table><thead><tr>${columns.map(k => `<th>${escapeHtml(k)}</th>`).join('')}</tr></thead><tbody>${data.map(row => `<tr>${columns.map(k => `<td>${escapeHtml(typeof row[k] === 'object' ? JSON.stringify(row[k]) : row[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  return `<div class="matrix"><pre>${escapeHtml(JSON.stringify(matrix, null, 2))}</pre></div>`;
}
function renderSection(payload) { return typeof payload === 'string' ? `<p>${escapeHtml(payload)}</p>` : renderMatrix(payload); }
function fileWidget(title, path) {
  return `<div class="clip"><strong>${escapeHtml(title)}</strong><button type="button" data-preview="${escapeHtml(path)}">Preview</button><a href="${assetUrl(path)}" download>Save source</a><pre class="preview" hidden></pre></div>`;
}
function imageWidget(title, path) {
  return `<figure class="image"><img src="${assetUrl(path)}" loading="lazy" alt="${escapeHtml(title)}"><figcaption>${escapeHtml(title)}</figcaption></figure>`;
}
function modelWidget(title, url) {
  return `<div class="clip"><strong>${escapeHtml(title)}</strong><button type="button" data-model="${escapeHtml(url)}">Inspect 3D</button><a href="${escapeHtml(url)}" download>Save model</a></div>`;
}
function bindCommon() {
  $('detail').querySelectorAll('[data-preview]').forEach(button => button.onclick = async () => {
    const pre = button.parentElement.querySelector('.preview');
    if (!pre.hidden) { pre.hidden = true; return; }
    const response = await fetch(`/api/preview?path=${encodeURIComponent(button.dataset.preview)}`);
    const data = await response.json();
    pre.textContent = response.ok ? data.text + (data.truncated ? '\n… preview truncated' : '') : data.error;
    pre.hidden = false;
  });
  $('detail').querySelectorAll('[data-model]').forEach(button => button.onclick = async () => {
    const host = $('model-host');
    host.hidden = false;
    if (!modelViewer) {
      const {createViewer} = await import('/ui/session-inspector/viewer3d.js');
      modelViewer = await createViewer(host.querySelector('.model-canvas'), host.querySelector('.model-label'));
    }
    try { await modelViewer.load(button.dataset.model); host.scrollIntoView({block:'center'}); }
    catch (error) { host.querySelector('.model-label').textContent = `Could not load model: ${error.message}`; }
  });
}
function modelHost() { return '<div id="model-host" class="model-host" hidden><div class="model-canvas"></div><p class="model-label">Loading 3D model…</p></div>'; }
function videoHost() { return '<div id="video-wrap" class="video-wrap" hidden><video id="player" controls playsinline preload="metadata"></video><p id="playing" class="muted"></p></div>'; }
function playVideo(url, title, fallback) {
  $('video-wrap').hidden = false;
  const player = $('player');
  player.onerror = fallback ? () => { player.onerror = null; player.src = fallback; player.load(); } : null;
  player.src = url;
  $('playing').textContent = title;
  $('video-wrap').scrollIntoView({behavior:'smooth',block:'center'});
  player.play().catch(() => {});
}
async function showResult(id) {
  const generation = ++viewGeneration;
  const requestedHash = location.hash;
  const requestedGroup = new URLSearchParams(location.search).get('group') || (requestedHash.startsWith('#catalog-group-') ? requestedHash.slice('#catalog-group-'.length) : null);
  catalogView?.dispose(); catalogView = null;
  const requestedClip = new URLSearchParams(location.search).get('clip');
  selected = id; modelViewer?.dispose(); modelViewer = null;
  history.replaceState(null, '', `${home}?result=${encodeURIComponent(id)}${requestedHash}`);
  renderList(); $('detail').innerHTML = '<p>Loading result…</p>';
  const response = await fetch(`/api/result?id=${encodeURIComponent(id)}`);
  if (generation !== viewGeneration) return;
  if (!response.ok) {
    if (isDesign && response.status === 410) {
      $('detail').innerHTML = '<h2>This hip study was removed</h2><p>The obsolete study was deleted during the body-first design cleanup.</p><a href="/design/?section=Hip">View current hip studies</a>';
    } else $('detail').textContent = 'Result unavailable.';
    return;
  }
  const row = await response.json();
  if (row.dashboard !== dashboard) { location.replace(`${row.dashboard === 'design' ? '/design/' : '/'}?result=${encodeURIComponent(id)}${location.hash}`); return; }
  if (generation !== viewGeneration) return;
  if (isDesign && row.record_kind === 'collection') {
    if (row.variants?.length) {
      const group = row.sections.flatMap(s=>s.payload?.groups || []).find(g=>g.id===requestedGroup);
      const memberIds = new Set((group?.options || []).flatMap(option=>[option.id, `${row.id}:${option.id}`]));
      const memberStudies = new Set(row.variants.filter(v=>!memberIds.size || memberIds.has(v.id)).map(v=>v.study_id));
      // Follow a retained option's study to its newest published iteration even
      // when that iteration was registered outside the legacy collection.
      const currentVariants = results.filter(v=>memberStudies.has(v.study_id));
      const groupVariants = currentVariants.length ? currentVariants : row.variants;
      const groupSection = group?.design_sections?.[0] || results.find(r=>memberIds.has(r.id))?.design_sections?.[0];
      if (groupSection) workspace=groupSection;
      const contextual = groupVariants.filter(v => belongs(results.find(r=>r.id===v.id) || v, workspace));
      const candidates = contextual.length ? contextual : groupVariants;
      const variant = [...candidates].sort((a,b)=>a.study_id.localeCompare(b.study_id) || versionOrder(a,b))[0];
      return showResult(variant.id);
    }
    mode='archive'; await loadArchive(); return showSession(row.id);
  }
  if (isDesign) {
    const {mountDesignDetail} = await import('./design-detail.js?v=mechanism-names-20261021');
    if (generation !== viewGeneration) return;
    const mounted = await mountDesignDetail($('detail'), row, workspace, {evidenceText, briefText, renderMatrix, renderGrades, imageWidget, fileWidget, modelWidget, modelHost, bindCommon, videoHost, playVideo, showResult, preferences, async prefer(section,study_id) {
      const response = await fetch('/api/design-preferences',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({section,study_id})});
      if(!response.ok) throw new Error((await response.json()).error);
      preferences=await (await fetch('/api/design-preferences')).json();renderList();
    }}, () => generation === viewGeneration);
    if(generation !== viewGeneration) {mounted.dispose();return;}
    catalogView=mounted;
    return;
  }
  const catalogs = row.sections.filter(section => section.payload?.kind === 'design_catalog');
  $('detail').classList.toggle('catalog-page', catalogs.length > 0);
  const catalogImages = new Set(catalogs.flatMap(section => section.payload.groups.flatMap(group => group.options.map(option => option.image_path))));
  const clips = row.clips.map(clip => `<div class="clip"><strong>${escapeHtml(clip.title)}</strong><small>${escapeHtml(clip.role)}</small><details class="text-more"><summary>Body and source context</summary>${evidenceText(clip.body)}</details>${briefText(clip.observation, 'Episode observation')}${clip.verified_at ? `<button type="button" data-video="${escapeHtml(clip.id)}">Watch here</button>` : '<small>Bucket upload pending</small>'}</div>`).join('');
  const images = row.assets.filter(item => item.kind === 'image' && !catalogImages.has(item.path)).map(item => imageWidget(item.title,item.path)).join('');
  const files = row.assets.filter(item => item.kind === 'data').map(item => fileWidget(item.title,item.path)).join('');
  const models = row.assets.filter(item => item.kind === 'model').map(item => modelWidget(item.title,assetUrl(item.path))).join('');
  const sections = row.sections.filter(section => section.payload?.kind !== 'design_catalog').map(section => `<details><summary>${escapeHtml(section.title)}</summary><div class="inside">${evidenceText(section.note)}${renderSection(section.payload)}</div></details>`).join('');
  $('detail').innerHTML = `<div class="meta"><span>${escapeHtml(row.workspace)}</span><span>${escapeHtml(row.area)}</span><span>${escapeHtml((row.status || 'Saved result').replaceAll('_',' '))}</span></div><h2>${escapeHtml(row.title)}</h2>${briefText(row.observation || row.summary)}<div class="cards"><div class="card"><strong>Measured evidence</strong>${evidenceText(row.measurements)}</div><div class="card"><strong>Open issues & limits</strong>${evidenceText(row.limitations, 2)}</div></div>${renderGrades(row.grades)}${videoHost()}${modelHost()}<details open><summary>Recordings · ${row.clips.length}</summary><div class="inside clip-list">${clips || '<p>No video was published for this record.</p>'}</div></details>${images ? `<details open><summary>Images · ${row.assets.filter(item => item.kind === 'image' && !catalogImages.has(item.path)).length}</summary><div class="inside gallery">${images}</div></details>` : ''}${models ? `<details><summary>3D models</summary><div class="inside clip-list">${models}</div></details>` : ''}${files ? `<details><summary>Source files</summary><div class="inside clip-list">${files}</div></details>` : ''}<details><summary>Case measurements</summary><div class="inside">${renderMatrix(row.matrix)}</div></details>${sections}<details><summary>Evidence references</summary><div class="inside"><p>${escapeHtml(row.summary)}</p><ul>${row.provenance.map(path => `<li>${escapeHtml(path)}</li>`).join('')}</ul></div></details>`;
  $('detail').querySelectorAll('[data-video]').forEach(button => button.onclick = async () => {
    const response = await fetch(`/api/video?id=${encodeURIComponent(button.dataset.video)}`);
    if (!response.ok) { alert('Video unavailable'); return; }
    const clip = row.clips.find(item => item.id === button.dataset.video);
    playVideo((await response.json()).url, clip.title);
  });
  bindCommon();
  if (catalogs.length) {
    const host = document.createElement('div');
    $('detail').querySelector('.cards').before(host);
    const {mountCatalog} = await import('./catalog.js?v=hip-body-first-20261020');
    if (generation !== viewGeneration) return;
    const mounted = await mountCatalog(host, id, catalogs[0].payload);
    if (generation !== viewGeneration) { mounted.dispose(); return; }
    catalogView = mounted;
    if (requestedHash) document.getElementById(requestedHash.slice(1))?.scrollIntoView();
  }
  if (requestedClip) {
    const button = [...$('detail').querySelectorAll('[data-video]')].find(item => item.dataset.video === requestedClip || item.dataset.video === `${id}:${requestedClip}`);
    if (button) button.closest('.clip').scrollIntoView({block:'center'});
  }
}
async function loadArchive() {
  if (!sessions.length) sessions = (await (await fetch(`/api/sessions?dashboard=${dashboard}`)).json()).sessions;
}
async function showSession(id) {
  ++viewGeneration; catalogView?.dispose(); catalogView = null;
  $('detail').classList.remove('catalog-page');
  selected = id; modelViewer?.dispose(); modelViewer = null;
  const row = sessions.find(item => item.id === id);
  if (!row) return;
  if (row.result_id) return showResult(row.result_id);
  history.replaceState(null, '', `${home}?session=${encodeURIComponent(id)}`);
  renderList();
  const images = row.images.map(item => imageWidget(item.name, item.url.startsWith('/asset?') ? new URL(item.url,location.origin).searchParams.get('path') : decodeURIComponent(item.url.slice(1)))).join('');
  const files = row.data.map(item => fileWidget(item.name, item.url.startsWith('/asset?') ? new URL(item.url,location.origin).searchParams.get('path') : decodeURIComponent(item.url.slice(1)))).join('');
  const models = row.models.map(item => modelWidget(item.name,item.url)).join('');
  const videos = row.videos.map(item => `<div class="clip"><strong>${escapeHtml(item.name)}</strong><button type="button" ${item.clip_id ? `data-archive-clip="${escapeHtml(item.clip_id)}"` : `data-local-video="${escapeHtml(item.url)}"`}>Watch here</button></div>`).join('');
  $('detail').innerHTML = `<div class="meta"><span>${escapeHtml(row.kind)}</span><span>Retained source run</span></div><h2>${escapeHtml(row.title)}</h2>${briefText(row.summary || row.id)}${isDesign ? modelHost() + videoHost() : videoHost() + modelHost()}${videos ? `<details open><summary>Recordings · ${row.videos.length}</summary><div class="inside clip-list">${videos}</div></details>` : ''}${images ? `<details open><summary>Images · ${row.images.length}</summary><div class="inside gallery">${images}</div></details>` : ''}${models ? `<details open><summary>3D models · ${row.models.length}</summary><div class="inside clip-list">${models}</div></details>` : ''}${files ? `<details><summary>Source files · ${row.data.length}</summary><div class="inside clip-list">${files}</div></details>` : ''}${row.landing ? `<details><summary>Historical interactive view</summary><div class="inside"><button type="button" id="load-legacy">Load inside dashboard</button><div id="legacy-frame"></div></div></details>` : ''}`;
  $('detail').querySelectorAll('[data-local-video]').forEach(button => button.onclick = () => {
    const source = decodeURIComponent(button.dataset.localVideo.slice(1));
    playVideo(button.dataset.localVideo, source.split('/').pop(), `/video?path=${encodeURIComponent(source)}`);
  });
  $('detail').querySelectorAll('[data-archive-clip]').forEach(button=>button.onclick=async()=>{const response=await fetch(`/api/video?id=${encodeURIComponent(button.dataset.archiveClip)}`);if(response.ok)playVideo((await response.json()).url,'Retained simulation recording');});
  if (row.landing) $('load-legacy').onclick = () => { $('legacy-frame').innerHTML = `<iframe title="Historical interactive view" sandbox="allow-scripts allow-same-origin" src="${escapeHtml(row.landing)}"></iframe>`; };
  bindCommon();
  if (isDesign && row.models.length) $('detail').querySelector('[data-model]')?.click();
}
async function setMode(next) {
  ++viewGeneration; catalogView?.dispose(); catalogView = null;
  $('detail').classList.remove('catalog-page');
  mode = next; selected = null; workspace = 'All'; modelViewer?.dispose(); modelViewer = null;
  $('published-mode').classList.toggle('active', next === 'published');
  $('archive-mode').classList.toggle('active', next === 'archive');
  $('detail').innerHTML = '<p>Loading evidence…</p>';
  if (next === 'archive') await loadArchive();
  const requestedSection = new URLSearchParams(location.search).get('section');
  if (isDesign && requestedSection) workspace = requestedSection;
  renderList();
  const params = new URLSearchParams(location.search);
  if (isDesign && next === 'published' && params.get('view') === 'bom') {await showRobotBom(); return;}
  const requested = next === 'archive' ? params.get('session') : params.get('result');
  if (requested && next === 'published') {await showResult(requested); return;}
  if (isDesign && next==='published' && !requested && requestedSection) {
    const first=records().find(row=>belongs(row,workspace));
    if(first) return showResult(first.id);
    $('detail').innerHTML='<h2>'+escapeHtml(workspace)+'</h2><p>No current studies in this section. Earlier work is retained in Archive.</p>';
    return;
  }
  if (isDesign && next==='published' && !requested) {
    $('results').innerHTML='<p class="muted">Choose a body section to see its design studies.</p>';
    $('detail').innerHTML='<div class="design-welcome"><span class="eyebrow">DESIGN EXPLORER</span><h2>Start with a part of the robot.</h2><p>Select a joint or body part on the drawing, or choose its name in the section list.</p><div class="cards"><div class="card"><strong>Explore variants</strong>Each study is one design variant, with its own CAD and evidence.</div><div class="card"><strong>Inspect an iteration</strong>The latest version opens first. Earlier CAD and results stay available.</div><div class="card"><strong>Keep a preference</strong>Mark the study you currently prefer for that body section.</div></div><img src="/body-map.png" alt="Cove-50 aesthetic reference, showing front, side and articulated views"></div>';
    return;
  }
  const initial = records().find(row => row.id === requested) || (isDesign && workspace !== 'All' ? records().find(row => belongs(row, workspace)) : null) || (next === 'published' ? results.find(row => row.id === (isDesign ? 'cove50-cad-competition-20261016' : 'basic-skill-sweep-20261016')) : null) || records()[0];
  if (initial) next === 'archive' ? showSession(initial.id) : showResult(initial.id);
  else $('detail').textContent = 'No evidence records yet.';
}
async function start() {
  if (isDesign) {
    preferences = await (await fetch('/api/design-preferences')).json();
    const design = await (await fetch('/api/design')).json();
    $('design-status').textContent = design.selected_design_id ? `${design.name} · selected candidate` : 'No mechanical design selected';
  }
  results = await (await fetch(`/api/results?dashboard=${dashboard}`)).json();
  await setMode(new URLSearchParams(location.search).has('session') ? 'archive' : 'published');
}
async function showRobotBom() {
  const generation = ++viewGeneration;
  catalogView?.dispose(); catalogView = null;
  modelViewer?.dispose(); modelViewer = null;
  mode = 'published'; selected = null;
  history.replaceState(null, '', '/design/?view=bom');
  $('published-mode').classList.add('active'); $('archive-mode').classList.remove('active');
  renderList();
  $('detail').classList.remove('single-study');
  $('detail').classList.add('catalog-page');
  const {mountRobotBom} = await import('./design-bom.js?v=bom-20261020');
  if (generation !== viewGeneration) return;
  const view = await mountRobotBom($('detail'), results, () => generation === viewGeneration);
  if (generation !== viewGeneration) {view.dispose(); return;}
  catalogView = view;
}
if (isDesign) $('robot-bom').onclick = showRobotBom;
$('search').addEventListener('input', renderList);
$('published-mode').onclick = () => setMode('published');
$('archive-mode').onclick = () => setMode('archive');
$('refresh').onclick = async () => { results = await (await fetch(`/api/results?dashboard=${dashboard}`)).json(); sessions = []; if (isDesign && new URLSearchParams(location.search).get('view') === 'bom') {await showRobotBom(); return;} if (mode === 'archive') await loadArchive(); renderList(); if (selected) mode === 'archive' ? showSession(selected) : showResult(selected); };
start().catch(error => { $('detail').textContent = `Could not load dashboard: ${error.message}`; });
