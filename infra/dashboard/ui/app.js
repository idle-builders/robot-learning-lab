import {evidenceText, briefText, shortText} from './readable.js?v=studies-final-20261020';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';
const isDesign = location.pathname.startsWith('/design');
const dashboard = isDesign ? 'design' : 'learning';
// The design navigation (body map, sections, study list) folds away while a study is open, so the
// study and its CAD view take the width; the landing page always shows it. The choice is remembered.
if (isDesign) {
  const NAV_KEY = 'ape-design-nav-collapsed', toggle = document.getElementById('toggle-nav');
  const navCollapsed = () => { try { return localStorage.getItem(NAV_KEY) !== '0'; } catch { return true; } };
  const showNav = collapsed => {
    document.body.classList.toggle('nav-collapsed', collapsed);
    if (toggle) { toggle.textContent = collapsed ? 'Show sections & studies' : 'Hide sections & studies'; toggle.setAttribute('aria-expanded', String(!collapsed)); }
  };
  showNav(navCollapsed());
  if (toggle) toggle.onclick = () => { const collapsed = !document.body.classList.contains('nav-collapsed'); showNav(collapsed); try { localStorage.setItem(NAV_KEY, collapsed ? '1' : '0'); } catch {} if (!collapsed) document.getElementById('design-nav')?.scrollIntoView({block: 'nearest'}); };
}
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
// Git holds versions: a version is named by its date and the commit of its sources.
const commitOf = v => String(v?.source_commit || '').replace('+dirty', '').slice(0, 7);
const versionLabel = v => `${String(v?.version_at || v?.updated_at || '').slice(0, 16)}${commitOf(v) ? ` · commit ${commitOf(v)}` : ''}${String(v?.source_commit || '').endsWith('+dirty') ? ' · uncommitted code' : ''}`;
import {assetUrl} from './asset-url.js';
let results = [], sessions = [], mode = 'published', page = 'policies', workspace = 'All', selected = null;
let skillSort = 'id', qualityTier = 'all';
let modelViewer = null;
let modelViewerKind = null;
let catalogView = null, viewGeneration = 0;
let curveViews = [];
function disposeCurveViews() { curveViews.forEach(view => view.dispose()); curveViews = []; }
let selection = {};  // robot section -> the one selected study (robot/design.json)
const selectedFor = studyId => Object.keys(selection).filter(section => selection[section] === studyId);
const versionOrder = (a,b) => {
  const byDate = String(b.version_at || b.updated_at || '').localeCompare(String(a.version_at || a.updated_at || ''));
  if (byDate) return byDate;
  const aVer = Number(a.version_num ?? NaN), bVer = Number(b.version_num ?? NaN);
  if (!Number.isNaN(aVer) && !Number.isNaN(bVer) && aVer !== bVer) return bVer - aVer;
  const aId = Number(a.version_id ?? NaN), bId = Number(b.version_id ?? NaN);
  if (!Number.isNaN(aId) && !Number.isNaN(bId) && aId !== bId) return bId - aId;
  return String(b.study_version || '').localeCompare(String(a.study_version || ''), undefined, {numeric:true});
};

function initialLearningPage() {
  const path = location.pathname.replace(/\/+$/, '') || '/';
  if (path === '/perception') return 'perception';
  if (path === '/planning') return 'planning';
  if (path === '/archive' || new URLSearchParams(location.search).has('session')) return 'archive';
  return 'policies';
}

function pagePath(targetPage = page) {
  if (isDesign) return '/design/';
  if (targetPage === 'perception') return '/perception';
  if (targetPage === 'planning') return '/planning';
  if (targetPage === 'archive') return '/archive';
  return '/';
}

const PAGE_META = {
  policies: {
    title: 'Policies',
    subtitle: 'The control system: the whole-body controller and the skill policies until the controller supersedes them. Each opens its newest run; earlier runs are in the version menu, named by date and commit.',
    searchLabel: 'Find a controller or skill',
    placeholder: 'Whole-body controller, walk, crouch, arms…',
    allLabel: 'All current',
    itemSingular: 'current line',
    itemPlural: 'current lines'
  },
  perception: {
    title: 'Perception',
    subtitle: '3D room reconstruction, camera memory, sensor capture, and onboard perception compute.',
    searchLabel: 'Find a perception study',
    placeholder: 'Room scan, Gaussian splat, camera…',
    allLabel: 'All areas',
    itemSingular: 'perception study',
    itemPlural: 'perception studies'
  },
  planning: {
    title: 'Planning',
    subtitle: 'High-level task planning, skill sequencing, and planner-to-policy interfaces.',
    searchLabel: 'Find a planning study',
    placeholder: 'Task planner, sequence, handoff…',
    allLabel: 'All areas',
    itemSingular: 'planning study',
    itemPlural: 'planning studies'
  },
  archive: {
    title: 'Archive',
    subtitle: 'Historical source runs, non-policy diagnostics, setup probes, and intermediate training studies.',
    searchLabel: 'Find an archived run',
    placeholder: 'Diagnostic, setup, PPO pilot, transfer…',
    allLabel: 'All archive',
    itemSingular: 'retained record',
    itemPlural: 'retained records'
  }
};

function syncPageChrome() {
  if (isDesign) return;
  const meta = PAGE_META[page] || PAGE_META.policies;
  if ($('page-title')) $('page-title').textContent = meta.title;
  if ($('page-subtitle')) $('page-subtitle').textContent = meta.subtitle;
  if ($('search-label')) $('search-label').textContent = meta.searchLabel;
  if ($('search')) $('search').placeholder = meta.placeholder;
  document.querySelectorAll('#top-nav [data-page]').forEach(link => {
    const active = link.dataset.page === page;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function shortBody(bodyText) {
  const text = String(bodyText || '');
  if (!text) return 'Simulator body';
  if (/r3-20261026|cove50_description|r3-urdf|APE0 R3|Cove-50 R3|Selected R3|cove50-rigid\.xml/i.test(text)) return 'APE0 R3';
  if (/APE0|Cove-50/i.test(text)) return 'APE0';
  if (/ape_sourced_v3|35 cm|v3/i.test(text)) return '35 cm shelf body';
  return shortText(text, 28);
}

function qualityInfo(row) {
  const g = row?.grades;
  if (!g) return null;
  let grade = typeof g.quality_grade === 'number' ? g.quality_grade : null;
  if (grade === null && Array.isArray(g.dimensions) && g.dimensions.length) {
    const nums = g.dimensions.map(d => Number(d.grade)).filter(n => !Number.isNaN(n));
    if (nums.length) grade = Math.round((nums.reduce((a, b) => a + b, 0) / nums.length) * 10) / 10;
  }
  if (grade === null) return null;
  const stage = g.quality_stage || (grade >= 3.8 ? 'Close to viable' : grade >= 3.0 ? 'Workable' : grade >= 2.0 ? 'Major gaps' : 'Far / 35 cm only');
  const tier = grade >= 3.8 ? 'close' : grade >= 3.0 ? 'workable' : grade >= 2.0 ? 'gaps' : 'early';
  const shortStage = /35 cm/i.test(stage) ? stage : (grade >= 3.8 ? 'Close' : grade >= 3.0 ? 'Workable' : grade >= 2.0 ? 'Major gaps' : 'Far');
  const focus = g.quality_focus || g.overall || '';
  const pct = Math.max(6, Math.min(100, Math.round((grade / 5) * 100)));
  return {grade: grade.toFixed(1), numeric: grade, stage, shortStage, focus, tier, pct, dimensions: g.dimensions || []};
}

function latestStudies() {
  const map = new Map();
  for (const row of [...results].sort(versionOrder)) if (!map.has(row.study_id || row.id)) map.set(row.study_id || row.id, row);
  return [...map.values()].sort((a,b)=>selectedFor(b.study_id).length-selectedFor(a.study_id).length || (b.model_count>0)-(a.model_count>0) || designStudyName(a).localeCompare(designStudyName(b)));
}

function latestSkills() {
  const map = new Map();
  const controlRows = results.filter(r => r.workspace === 'Control');
  for (const row of [...controlRows].sort(versionOrder)) {
    const key = String(row.skill_id || row.study_id || row.id);
    if (!map.has(key)) map.set(key, row);
  }
  return [...map.values()].sort((a,b) => (a.skill_id ?? a.skill_order ?? 9999) - (b.skill_id ?? b.skill_order ?? 9999) || (a.skill_title || a.title).localeCompare(b.skill_title || b.title));
}

const SKILL_CATEGORIES = ['Locomotion', 'Posture & Balance', 'Arm Control', 'Skill Composition', 'Manipulation'];

const belongs = (row, name) => {
  if (name === 'All') return true;
  if (isDesign) return (row.design_sections || ['Whole robot']).includes(name);
  if (mode === 'archive' || page === 'archive') return (row.workspace || row.kind || 'Other') === name;
  return (row.area || 'Other') === name;
};

function records() {
  if (mode === 'archive' || (!isDesign && page === 'archive')) {
    return sessions.map(row => ({...row, workspace: row.kind || 'Other'}));
  }
  if (isDesign) return latestStudies();
  if (page === 'policies') return latestSkills();
  if (page === 'perception') return results.filter(row => row.workspace === 'Perception');
  if (page === 'planning') return results.filter(row => row.workspace === 'Planning');
  return results;
}

function renderList() {
  syncPageChrome();
  const all = records();
  let spaces;
  if (isDesign) {
    spaces = [...new Set(all.flatMap(row => row.design_sections || ['Whole robot']))];
  } else if (page === 'policies') {
    spaces = SKILL_CATEGORIES.filter(cat => all.some(row => row.area === cat));
  } else if (page === 'archive') {
    spaces = [...new Set(all.map(row => row.workspace || 'Other'))].sort();
  } else {
    spaces = [...new Set(all.map(row => row.area || 'Other'))].sort();
  }
  const allLabel = isDesign ? 'All sections' : (PAGE_META[page]?.allLabel || 'All');
  $('workspaces').innerHTML = ['All', ...spaces].map(name => `<button type="button" data-workspace="${escapeHtml(name)}" class="${workspace === name ? 'active' : ''}" aria-pressed="${workspace === name}">${escapeHtml(name === 'All' ? allLabel : name)} <small>${all.filter(row => belongs(row, name)).length}</small></button>`).join('');
  $('workspaces').querySelectorAll('button').forEach(button => button.onclick = () => {
    workspace = button.dataset.workspace;
    $('body-map')?.querySelectorAll('[data-section]').forEach(marker => marker.classList.toggle('active', marker.dataset.section === workspace));
    const selectedRow = results.find(r => r.id === selected);
    const current = records().find(row => (row.id === selected || (row.study_id && row.study_id === selectedRow?.study_id)) && belongs(row, workspace));
    const chosen = records().find(row => row.study_id === selection[workspace] && belongs(row, workspace));
    const first = current || chosen || records().find(row => belongs(row, workspace));
    if (first) (mode === 'archive' || page === 'archive') ? showSession(first.id) : showResult(first.id);
    else { selected = null; renderList(); renderEmptyState(); }
  });
  const query = $('search').value.trim().toLowerCase();
  const categoryFiltered = all.filter(row => belongs(row, workspace) &&
    `skill #${row.skill_id || ''} id #${row.version_id || ''} ${row.version_at || ''} ${row.skill_title || ''} ${row.title} ${row.area || ''} ${row.grades?.quality_stage || ''} ${row.grades?.quality_focus || ''} ${row.observation || ''} ${row.summary || ''} ${row.id}`.toLowerCase().includes(query));
  let filtered = categoryFiltered;
  if (!isDesign && page === 'policies') {
    if (qualityTier !== 'all') {
      filtered = filtered.filter(row => qualityInfo(row)?.tier === qualityTier);
    }
    if (skillSort === 'quality_desc') {
      filtered = [...filtered].sort((a, b) => (qualityInfo(b)?.numeric ?? -1) - (qualityInfo(a)?.numeric ?? -1) || (a.skill_id ?? 99) - (b.skill_id ?? 99));
    } else if (skillSort === 'quality_asc') {
      filtered = [...filtered].sort((a, b) => (qualityInfo(a)?.numeric ?? 99) - (qualityInfo(b)?.numeric ?? 99) || (a.skill_id ?? 99) - (b.skill_id ?? 99));
    }
  }
  if (isDesign) {
    $('count').textContent = `${filtered.length} ${mode === 'archive' ? (filtered.length === 1 ? 'retained source' : 'retained sources') : (filtered.length === 1 ? 'variant study' : 'variant studies')}`;
  } else if (page === 'policies') {
    const tierCounts = {close: 0, workable: 0, gaps: 0, early: 0};
    for (const r of categoryFiltered) {
      const q = qualityInfo(r);
      if (q) tierCounts[q.tier] = (tierCounts[q.tier] || 0) + 1;
    }
    $('count').innerHTML = `
      <div class="portfolio-readiness">
        <div class="portfolio-readiness-head">
          <span>Readiness (0–5 scale) · ${filtered.length}/${categoryFiltered.length} skills</span>
          <select id="skill-sort" aria-label="Sort skills">
            <option value="id" ${skillSort === 'id' ? 'selected' : ''}>Sort: Skill #1–15</option>
            <option value="quality_desc" ${skillSort === 'quality_desc' ? 'selected' : ''}>Sort: Highest quality ↓</option>
            <option value="quality_asc" ${skillSort === 'quality_asc' ? 'selected' : ''}>Sort: Needs work first ↑</option>
          </select>
        </div>
        <div class="tier-filter-row" role="group" aria-label="Filter by readiness tier">
          <button type="button" data-tier="all" class="${qualityTier === 'all' ? 'active' : ''}">All (${categoryFiltered.length})</button>
          <button type="button" data-tier="close" class="${qualityTier === 'close' ? 'active' : ''}" title="3.8–5.0 / 5: Close to viable on APE0 R3">Close ≥3.8 (${tierCounts.close})</button>
          <button type="button" data-tier="workable" class="${qualityTier === 'workable' ? 'active' : ''}" title="3.0–3.7 / 5: Workable on APE0 R3">Workable 3.0–3.7 (${tierCounts.workable})</button>
          <button type="button" data-tier="gaps" class="${qualityTier === 'gaps' ? 'active' : ''}" title="2.0–2.9 / 5: Major gaps on APE0 R3">Gaps 2.0–2.9 (${tierCounts.gaps})</button>
          <button type="button" data-tier="early" class="${qualityTier === 'early' ? 'active' : ''}" title="<2.0 / 5: Still on 35 cm body only">35 cm only (${tierCounts.early})</button>
        </div>
      </div>`;
    $('skill-sort').onchange = e => { skillSort = e.target.value; renderList(); };
    $('count').querySelectorAll('[data-tier]').forEach(btn => btn.onclick = () => {
      qualityTier = btn.dataset.tier;
      renderList();
    });
  } else {
    const meta = PAGE_META[page] || PAGE_META.policies;
    $('count').textContent = `${filtered.length} ${filtered.length === 1 ? meta.itemSingular : meta.itemPlural}`;
  }
  const selectedRow = results.find(r => r.id === selected || String(r.version_id) === String(selected));
  const selectedStudyId = selectedRow?.study_id;
  $('results').innerHTML = filtered.slice(0, 120).map(row => {
    const isPolicySkill = !isDesign && page === 'policies';
    const name = isDesign ? designStudyName(row) : isPolicySkill ? (row.skill_title || row.title) : row.title;
    const isActive = row.id === selected || String(row.version_id) === String(selected) || (row.study_id && row.study_id === selectedStudyId);
    if (isPolicySkill) {
      const vCount = row.version_count || 1;
      const skillNum = row.skill_id || row.study_id;
      const verNum = row.version_num || row.study_version || 1;
      const verId = row.version_id;
      const ts = row.version_at ? `${row.version_at.slice(0, 16)} UTC` : '';
      const q = qualityInfo(row);
      const qBadge = q ? `<b class="quality-pill tier-${escapeHtml(q.tier)}" title="Quality ${escapeHtml(q.grade)}/5 (${escapeHtml(q.stage)})">${escapeHtml(q.grade)}/5 · ${escapeHtml(q.shortStage)}</b>` : '';
      const qBar = q ? `<div class="skill-quality-bar tier-${escapeHtml(q.tier)}" aria-hidden="true"><i style="width:${q.pct}%"></i></div>` : '';
      const qFocus = q && q.focus ? `<span class="skill-focus-note">${escapeHtml(shortText(q.focus, 96))}</span>` : '';
      return `<button type="button" title="${escapeHtml(name)} · ${escapeHtml(versionLabel(row))}" data-id="${escapeHtml(verId || row.id)}" class="${isActive ? 'active' : ''}"><div class="skill-card-top"><strong>${escapeHtml(shortText(name, 68))}</strong>${qBadge}</div>${qBar}${qFocus}<span>${escapeHtml(versionLabel(row))} · ${escapeHtml(row.area)} · ${vCount} ${vCount === 1 ? 'version' : 'versions'}</span></button>`;
    }
    const version = (row.version_at || row.source_commit) ? ` · ${versionLabel(row).slice(0, isDesign ? 10 : 16)}` : '';
    return `<button type="button" title="${escapeHtml(name)}" data-id="${escapeHtml(!isDesign && row.version_id ? row.version_id : row.id)}" class="${isActive ? 'active' : ''}">${escapeHtml(shortText(name, 105))}${isDesign && (workspace === 'All' ? selectedFor(row.study_id).length : selection[workspace] === row.study_id) ? `<b class="preferred-mark" title="Selected for ${escapeHtml(selectedFor(row.study_id).join(', '))}">Selected</b>` : ''}<span>${escapeHtml(isDesign ? row.design_sections?.join(' · ') : (row.area || row.workspace))}${escapeHtml(version)}${mode === 'published' && page !== 'archive' && row.clip_count ? ` · ${row.uploaded_clips || 0}/${row.clip_count} videos` : ''}${mode === 'published' && page !== 'archive' && row.asset_count ? ` · ${row.asset_count} files` : ''}</span></button>`;
  }).join('');
  if (filtered.length > 120) $('results').insertAdjacentHTML('beforeend', '<p class="muted">Showing 120. Search to narrow the list.</p>');
  if (isDesign) $('body-map')?.querySelectorAll('[data-section]').forEach(marker => {marker.classList.toggle('active',marker.dataset.section===workspace);marker.setAttribute('aria-pressed',String(marker.dataset.section===workspace));marker.onclick = () => {
    const button = [...$('workspaces').querySelectorAll('button')].find(b=>b.dataset.workspace===marker.dataset.section);
    if (button) button.click(); else {workspace=marker.dataset.section;renderList();$('detail').innerHTML='<h2>'+escapeHtml(workspace)+'</h2><p>No variant studies registered for this section yet.</p>'; }
  };});
  $('results').querySelectorAll('button').forEach(button => button.onclick = () => (mode === 'archive' || page === 'archive') ? showSession(button.dataset.id) : showResult(button.dataset.id));
}

function renderEmptyState() {
  if (!isDesign && page === 'planning') {
    $('detail').innerHTML = `<div class="meta"><span>Planning</span><span>High-level control & sequencing</span></div><h2>Task planning & skill sequencing</h2><p class="lede">This page is dedicated to high-level task planning, behavior trees, and multi-skill planner-to-policy studies.</p><div class="cards"><div class="card"><strong>Low-level multi-skill transitions</strong><p>Continuous 34 s closed-loop transitions across quiet stance, shallow crouch, bilateral arms, and independent arms are versioned under <a href="/?skill=8&version=5">Policies → Skill #8 · Supported stance, posture &amp; arm transitions (Version 5 · ID #37)</a>.</p></div><div class="card"><strong>Publish a planning study</strong><p>Publish new task-planning and perception-to-control sequencing runs with <code>"workspace": "Planning"</code> via <code>infra/dashboard/results_hub.py publish</code>.</p></div></div>`;
    return;
  }
  $('detail').textContent = 'No records in this view yet. Earlier work is retained in Archive.';
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
function renderSection(payload) {
  if (payload?.kind === 'command_error_curves') return '<div data-command-error-curves><p>Native error curves and requested-horizon coverage load when this section is opened.</p><button type="button" data-load-command-error-curves>Load native curves</button></div>';
  return typeof payload === 'string' ? `<p>${escapeHtml(payload)}</p>` : renderMatrix(payload);
}
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
    const response = await fetch(`/api/preview?path=${encodeURIComponent(button.dataset.preview)}${globalThis.assetVersionQuery?.(button.dataset.preview) ?? ''}`);
    const data = await response.json();
    pre.textContent = response.ok ? data.text + (data.truncated ? '\n… preview truncated' : '') : data.error;
    pre.hidden = false;
  });
  $('detail').querySelectorAll('[data-model]').forEach(button => button.onclick = async () => {
    const host = $('model-host');
    host.hidden = false;
    try {
      const kind = /\.(ply|splat|ksplat)(?:[?#]|$)/i.test(button.dataset.model) ? 'splat' : 'mesh';
      if (modelViewer && modelViewerKind !== kind) {
        await modelViewer.dispose(); modelViewer = null;
      }
      if (!modelViewer) {
        const {createViewer} = await import(kind === 'splat' ? '/splat-viewer.js' : '/ui/session-inspector/viewer3d.js');
        modelViewer = await createViewer(host.querySelector('.model-canvas'), host.querySelector('.model-label'));
        modelViewerKind = kind;
      }
      await modelViewer.load(button.dataset.model); host.scrollIntoView({block:'center'});
    }
    catch (error) { host.querySelector('.model-label').textContent = `Could not load model: ${error.message}`; }
  });
}
function modelHost() { return '<div id="model-host" class="model-host" hidden><div class="model-canvas"></div><p class="model-label">Loading 3D model…</p></div>'; }
function videoHost() { return '<div id="video-wrap" class="video-wrap" hidden><video id="player" controls playsinline preload="metadata"></video><p id="playing" class="muted"></p></div>'; }
function playVideo(url, title, fallback, autoPlay = true) {
  const wrap = $('video-wrap');
  if (wrap) wrap.hidden = false;
  const player = $('player');
  if (!player) return;
  player.onerror = fallback ? () => { player.onerror = null; player.src = fallback; player.load(); } : null;
  player.src = url;
  if ($('playing')) $('playing').textContent = title;
  if (autoPlay) {
    wrap?.scrollIntoView({behavior:'smooth',block:'center'});
    player.play().catch(() => {});
  }
}

async function loadHeroClip(clip, generation, autoPlay = false) {
  if (!clip || !clip.verified_at) return;
  const query = clip.clip_num ? `clip_id=${encodeURIComponent(clip.clip_num)}` : `id=${encodeURIComponent(clip.id)}`;
  const response = await fetch(`/api/video?${query}`);
  if (generation !== viewGeneration || !response.ok) return;
  const data = await response.json();
  if (generation !== viewGeneration) return;
  const player = $('player');
  if (!player) return;
  player.poster = data.poster || '';
  player.src = data.url;
  if ($('playing')) $('playing').textContent = clip.clip_num ? `Clip #${clip.clip_num} · ${clip.title}` : clip.title;
  if ($('playing-role')) $('playing-role').textContent = clip.role || '';
  if ($('playing-obs')) $('playing-obs').textContent = clip.observation || '';
  const clipKey = String(clip.clip_num || clip.id);
  document.querySelectorAll('[data-hero-clip]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.heroClip === clipKey);
  });
  if (autoPlay) player.play().catch(() => {});
}

async function showResult(target, options = {}) {
  disposeCurveViews();
  const generation = ++viewGeneration;
  const hashId = location.hash && !location.hash.startsWith('#catalog-group-') && location.hash !== '#design-catalog'
    ? decodeURIComponent(location.hash.slice(1))
    : null;
  const requestedHash = hashId === target ? '' : location.hash;
  const requestedGroup = new URLSearchParams(location.search).get('group') || (requestedHash.startsWith('#catalog-group-') ? requestedHash.slice('#catalog-group-'.length) : null);
  catalogView?.dispose(); catalogView = null;
  const requestedClip = new URLSearchParams(location.search).get('clip');
  let apiQuery;
  if (target && typeof target === 'object') {
    const qs = new URLSearchParams();
    if (target.skill) qs.set('skill', target.skill);
    if (target.version) qs.set('version', target.version);
    if (target.version_id) qs.set('version_id', target.version_id);
    apiQuery = qs.toString();
    selected = target.version_id || target.skill;
  } else {
    selected = target;
    apiQuery = /^\d+$/.test(String(target)) ? `version_id=${encodeURIComponent(target)}` : `id=${encodeURIComponent(target)}`;
  }
  modelViewer?.dispose(); modelViewer = null;
  renderList(); $('detail').innerHTML = '<p>Loading result…</p>';
  const response = await fetch(`/api/result?${apiQuery}`);
  if (generation !== viewGeneration) return;
  if (!response.ok) {
    if (isDesign && response.status === 410) {
      $('detail').innerHTML = '<h2>This design study was deleted</h2><p>It was removed at the user’s request.</p><a href="/design/">View the robot design bench</a>';
    } else $('detail').textContent = 'Result unavailable.';
    return;
  }
  const row = await response.json();
  const id = row.id;
  selected = row.id;
  // Earlier versions show their own registered bytes; the current version shows the working files.
  const newest = [...(row.versions || [])].sort((a, b) => String(b.version_at || '').localeCompare(String(a.version_at || '')))[0];
  const earlier = isDesign ? Boolean(row.study_id && row.id !== row.study_id) : Boolean(newest && newest.id !== row.id);
  const shas = new Map(earlier ? (row.assets || []).map(a => [a.path, a.sha256]) : []);
  globalThis.assetVersionQuery = path => shas.has(path) ? `&sha=${shas.get(path)}` : '';
  if (row.dashboard !== dashboard) { location.replace(`${row.dashboard === 'design' ? '/design/' : '/'}?result=${encodeURIComponent(id)}${location.hash}`); return; }
  if (generation !== viewGeneration) return;

  if (!isDesign && !options.keepPage) {
    const targetPage = row.archived_at ? 'archive' : row.workspace === 'Perception' ? 'perception' : row.workspace === 'Planning' ? 'planning' : 'policies';
    if (targetPage !== page) {
      page = targetPage;
      mode = targetPage === 'archive' ? 'archive' : 'published';
      if (mode === 'archive') await loadArchive();
    }
  }
  const isPolicySkill = row.workspace === 'Control' && row.study_id && !row.archived_at;
  if (isPolicySkill && row.skill_id && row.version_num) {
    history.replaceState(null, '', `${pagePath(page)}?skill=${encodeURIComponent(row.skill_id)}&version=${encodeURIComponent(row.version_num)}${requestedHash}`);
  } else if (!isDesign && row.version_id && !row.archived_at) {
    history.replaceState(null, '', `${pagePath(page)}?version_id=${encodeURIComponent(row.version_id)}${requestedHash}`);
  } else {
    history.replaceState(null, '', `${pagePath(page)}?result=${encodeURIComponent(id)}${requestedHash}`);
  }
  renderList();

  if (isDesign && row.record_kind === 'collection') {
    if (row.variants?.length) {
      const group = row.sections.flatMap(s=>s.payload?.groups || []).find(g=>g.id===requestedGroup);
      const memberIds = new Set((group?.options || []).flatMap(option=>[option.id, `${row.id}:${option.id}`]));
      const memberStudies = new Set(row.variants.filter(v=>!memberIds.size || memberIds.has(v.id)).map(v=>v.study_id));
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
    const {mountDesignDetail} = await import('./design-detail.js?v=bom-ranges-20261123');
    if (generation !== viewGeneration) return;
    const mounted = await mountDesignDetail($('detail'), row, workspace, {evidenceText, briefText, renderMatrix, renderGrades, imageWidget, fileWidget, modelWidget, modelHost, bindCommon, videoHost, playVideo, showResult, selection, async select(section,study_id) {
      const response = await fetch('/api/design-selection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({section,study_id})});
      if(!response.ok) throw new Error((await response.json()).error);
      selection=await response.json();renderList();
    }}, () => generation === viewGeneration);
    if(generation !== viewGeneration) {mounted.dispose();return;}
    catalogView=mounted;
    return;
  }

  const catalogs = row.sections.filter(section => section.payload?.kind === 'design_catalog');
  $('detail').classList.toggle('catalog-page', catalogs.length > 0);
  const catalogImages = new Set(catalogs.flatMap(section => section.payload.groups.flatMap(group => group.options.map(option => option.image_path))));
  const clips = row.clips.map(clip => {
    const origin = clip.result_id && clip.result_id !== row.id
      ? `<small>Earlier recording · <a href="?result=${encodeURIComponent(clip.result_id)}">View source result</a></small>` : '';
    const clipLabel = clip.clip_num ? `Clip #${clip.clip_num} · ${clip.title}` : clip.title;
    const clipKey = clip.clip_num || clip.id;
    return `<div class="clip"><strong>${escapeHtml(clipLabel)}</strong><small>${escapeHtml(clip.role)}</small>${origin}<details class="text-more"><summary>Body and source context</summary>${evidenceText(clip.body)}</details>${briefText(clip.observation, 'Episode observation')}${clip.verified_at ? `<button type="button" data-video="${escapeHtml(clipKey)}">Watch in player</button>` : '<small>Bucket upload pending</small>'}</div>`;
  }).join('');
  const images = row.assets.filter(item => item.kind === 'image' && !catalogImages.has(item.path)).map(item => imageWidget(item.title,item.path)).join('');
  const files = row.assets.filter(item => item.kind === 'data').map(item => fileWidget(item.title,item.path)).join('');
  const worldInspection = row.sections.find(section => section.payload?.kind === 'world_inspection')?.payload;
  const primaryModel = worldInspection?.primary_model_path;
  const modelAssets = row.assets.filter(item => item.kind === 'model');
  const primaryAsset = modelAssets.find(item => item.path === primaryModel);
  const olderModels = modelAssets.filter(item => item !== primaryAsset).map(item => modelWidget(item.title,assetUrl(item.path))).join('');
  const recommendation = worldInspection?.recommendation ? `<p>${escapeHtml(worldInspection.recommendation)}</p>` : '';
  const recommendedId = worldInspection?.recommended_result_id;
  const recommendedLink = recommendedId && recommendedId !== row.id ? `<p><a href="/perception?result=${encodeURIComponent(recommendedId)}">Open the recommended world</a></p>` : '';
  const models = primaryAsset ? `${recommendation}${recommendedLink}${modelWidget(primaryAsset.title,assetUrl(primaryAsset.path))}${olderModels ? `<details><summary>Older attempts · ${modelAssets.length - 1}</summary><div class="inside clip-list">${olderModels}</div></details>` : ''}` : olderModels;
  const modelSection = models ? `<details${primaryAsset ? ' open' : ''}><summary>${primaryAsset ? 'Interactive 3D world' : '3D models'}</summary><div class="inside clip-list">${models}</div></details>` : '';
  const sections = row.sections.filter(section => section.payload?.kind !== 'design_catalog' && section.payload?.kind !== 'world_inspection').map(section => `<details><summary>${escapeHtml(section.title)}</summary><div class="inside">${evidenceText(section.note)}${renderSection(section.payload)}</div></details>`).join('');

  if (isPolicySkill) {
    const versions = [...(row.versions || [])].sort(versionOrder);
    const latestVer = versions[0];
    const isLatest = !latestVer || latestVer.id === row.id;
    const primaryClip = (requestedClip && row.clips.find(c => String(c.clip_num) === String(requestedClip) || c.id === requestedClip || c.id === `${id}:${requestedClip}`)) || row.clips[0];
    const switcher = row.clips.length > 1
      ? `<div class="clip-switcher" role="group" aria-label="Recordings for this policy version">${row.clips.map((c, idx) => `<button type="button" data-hero-clip="${escapeHtml(c.clip_num || c.id)}" class="${c.id === primaryClip?.id ? 'active' : ''}">▶ ${escapeHtml(c.clip_num ? `Clip #${c.clip_num} · ${c.title}` : c.title)} <small>${idx === 0 ? '· primary policy' : '· preserved / baseline'}</small></button>`).join('')}</div>`
      : '';
    const heroVideo = primaryClip
      ? `<section class="policy-video-hero" aria-label="Policy video player"><video id="player" controls playsinline preload="metadata"></video><div class="policy-video-caption"><strong id="playing">${escapeHtml(primaryClip.clip_num ? `Clip #${primaryClip.clip_num} · ${primaryClip.title}` : primaryClip.title)}</strong><span id="playing-role">${escapeHtml(primaryClip.role || '')}</span></div><p id="playing-obs" class="policy-video-obs">${escapeHtml(shortText(primaryClip.observation || '', 220))}</p>${switcher}</section>`
      : videoHost();
    const skillNum = row.skill_id || row.study_id;
    const verNum = row.version_num || row.study_version || 1;
    const verId = row.version_id;
    const latestVerNum = latestVer ? (latestVer.version_num || latestVer.study_version) : verNum;
    const latestVerId = latestVer ? latestVer.version_id : verId;
    const q = qualityInfo(row);
    const dimChips = q && q.dimensions.length
      ? `<div class="dim-chips">${q.dimensions.map(d => `<span class="dim-chip" title="${escapeHtml(d.reason || '')}">${escapeHtml(d.dimension)}: <b>${escapeHtml(d.grade)}/5</b></span>`).join('')}</div>`
      : '';
    const standingsCard = q
      ? `<section class="standings-card tier-${escapeHtml(q.tier)}" aria-label="Skill readiness and next focus">
           <div class="standings-head">
             <strong>Where we stand &amp; what to look at</strong>
             <span class="quality-pill tier-${escapeHtml(q.tier)}">Quality ${escapeHtml(q.grade)} / 5 · ${escapeHtml(q.stage)}</span>
           </div>
           <p class="standings-focus">${escapeHtml(q.focus)}</p>
           ${dimChips}
         </section>`
      : '';

    $('detail').innerHTML = `
      <div class="meta">
        ${q ? `<span class="tier-${escapeHtml(q.tier)}"><strong>Quality ${escapeHtml(q.grade)} / 5 · ${escapeHtml(q.stage)}</strong></span>` : ''}
        <span>${escapeHtml(row.area)}</span>
        <span class="${isLatest ? 'badge-latest' : 'badge-historical'}">${isLatest ? 'Current' : `Earlier version (current: ${escapeHtml(versionLabel(latestVer))})`}</span>
        <span>${escapeHtml(versionLabel(row))}</span>
        <span>${escapeHtml(shortBody(row.body))}</span>
      </div>
      <h2>${escapeHtml(row.skill_title || row.title)}</h2>
      <p class="version-subtitle"><strong>${escapeHtml(versionLabel(row))}${isLatest ? ' · current' : ''}:</strong> ${escapeHtml(row.title)}</p>
      <div class="exploration-bar">
        <label for="policy-version">Version (${versions.length} retained, newest first)
          <select id="policy-version">${versions.map((v, i) => {
            const vq = qualityInfo(v);
            const vqLabel = vq ? ` · ${vq.grade}/5 ${vq.shortStage}` : '';
            return `<option value="${escapeHtml(v.version_id || v.id)}" ${v.id === row.id ? 'selected' : ''}>${escapeHtml(versionLabel(v))}${escapeHtml(vqLabel)}${i === 0 ? ' · current' : ''} — ${escapeHtml(v.title)}</option>`;
          }).join('')}</select>
        </label>
        ${!isLatest && latestVer ? `<div class="preference-actions"><button type="button" id="jump-latest-version">Switch to the current version</button></div>` : ''}
      </div>
      ${standingsCard}
      ${heroVideo}
      ${briefText(row.observation || row.summary)}
      <div class="cards"><div class="card"><strong>Measured evidence</strong>${evidenceText(row.measurements)}</div><div class="card"><strong>Open issues & limits</strong>${evidenceText(row.limitations, 2)}</div></div>
      ${renderGrades(row.grades)}
      ${modelHost()}
      ${modelSection}
      <details><summary>Recording details &amp; source context · ${row.clips.length}</summary><div class="inside clip-list">${clips || '<p>No video was published for this record.</p>'}</div></details>
      ${images ? `<details><summary>Images · ${row.assets.filter(item => item.kind === 'image' && !catalogImages.has(item.path)).length}</summary><div class="inside gallery">${images}</div></details>` : ''}
      ${files ? `<details><summary>Source files</summary><div class="inside clip-list">${files}</div></details>` : ''}
      <details><summary>Case measurements</summary><div class="inside">${renderMatrix(row.matrix)}</div></details>
      ${sections}
      <details><summary>Evidence references</summary><div class="inside"><p>${escapeHtml(row.summary)}</p><ul>${row.provenance.map(path => `<li>${escapeHtml(path)}</li>`).join('')}</ul></div></details>`;

    $('policy-version').onchange = event => showResult(event.target.value);
    if ($('jump-latest-version')) $('jump-latest-version').onclick = () => showResult(latestVer.version_id || latestVer.id);
    $('detail').querySelectorAll('[data-hero-clip]').forEach(btn => {
      btn.onclick = () => {
        const clip = row.clips.find(c => String(c.clip_num || c.id) === btn.dataset.heroClip);
        if (clip) loadHeroClip(clip, generation, true);
      };
    });
    if (primaryClip) loadHeroClip(primaryClip, generation, Boolean(requestedClip));
  } else {
    const archiveNotice = row.archived_at ? `<div class="archive-banner"><strong>Archived diagnostic / setup record:</strong> ${escapeHtml(row.archive_reason || 'Retained in Archive with full native assets and provenance.')}</div>` : '';
    const metaVer = (row.version_at || row.source_commit) ? `<span>${escapeHtml(versionLabel(row))}</span>` : '';
    const metaTs = '';
    $('detail').innerHTML = `${archiveNotice}<div class="meta"><span>${escapeHtml(row.workspace)}</span><span>${escapeHtml(row.area)}</span>${metaVer}${metaTs}<span>${escapeHtml((row.status || 'Saved result').replaceAll('_',' '))}</span></div><h2>${escapeHtml(row.title)}</h2>${briefText(row.observation || row.summary)}<div class="cards"><div class="card"><strong>Measured evidence</strong>${evidenceText(row.measurements)}</div><div class="card"><strong>Open issues & limits</strong>${evidenceText(row.limitations, 2)}</div></div>${renderGrades(row.grades)}${videoHost()}${modelHost()}${primaryModel ? modelSection : ''}<details open><summary>Recordings · ${row.clips.length}</summary><div class="inside clip-list">${clips || '<p>No video was published for this record.</p>'}</div></details>${images ? `<details open><summary>Images · ${row.assets.filter(item => item.kind === 'image' && !catalogImages.has(item.path)).length}</summary><div class="inside gallery">${images}</div></details>` : ''}${primaryModel ? '' : modelSection}${files ? `<details><summary>Source files</summary><div class="inside clip-list">${files}</div></details>` : ''}<details><summary>Case measurements</summary><div class="inside">${renderMatrix(row.matrix)}</div></details>${sections}<details><summary>Evidence references</summary><div class="inside"><p>${escapeHtml(row.summary)}</p><ul>${row.provenance.map(path => `<li>${escapeHtml(path)}</li>`).join('')}</ul></div></details>`;
  }

  $('detail').querySelectorAll('[data-video]').forEach(button => button.onclick = async () => {
    const clip = row.clips.find(item => String(item.clip_num || item.id) === button.dataset.video || item.id === button.dataset.video);
    if (isPolicySkill && clip) {
      await loadHeroClip(clip, generation, true);
      $('player')?.scrollIntoView({behavior:'smooth', block:'center'});
      return;
    }
    const query = /^\d+$/.test(button.dataset.video) ? `clip_id=${encodeURIComponent(button.dataset.video)}` : `id=${encodeURIComponent(button.dataset.video)}`;
    const response = await fetch(`/api/video?${query}`);
    if (!response.ok) { alert('Video unavailable'); return; }
    playVideo((await response.json()).url, clip ? (clip.clip_num ? `Clip #${clip.clip_num} · ${clip.title}` : clip.title) : 'Recording');
  });
  bindCommon();
  const curveSections = row.sections.filter(section => section.payload?.kind === 'command_error_curves');
  $('detail').querySelectorAll('[data-command-error-curves]').forEach((host, index) => {
    const details = host.closest('details'); let started = false;
    const load = async () => {
      if (started || !details.open || generation !== viewGeneration) return;
      started = true;
      try {
        const {mountCommandErrorCurves} = await import('/command-error-curves.js?v=native-curves-20261030');
        if (generation !== viewGeneration) return;
        const view = mountCommandErrorCurves(host, curveSections[index].payload, row.assets, () => generation === viewGeneration);
        curveViews.push(view); await view.load();
      } catch (error) { if (generation === viewGeneration) host.textContent = `Curves unavailable: ${error.message}`; }
    };
    details.addEventListener('toggle', () => { if (details.open) load(); });
    host.querySelector('[data-load-command-error-curves]').onclick = load;
  });
  if (catalogs.length) {
    const host = document.createElement('div');
    $('detail').querySelector('.cards').before(host);
    const {mountCatalog} = await import('./catalog.js?v=bom-ranges-20261123');
    if (generation !== viewGeneration) return;
    const mounted = await mountCatalog(host, id, catalogs[0].payload);
    if (generation !== viewGeneration) { mounted.dispose(); return; }
    catalogView = mounted;
    if (requestedHash) document.getElementById(requestedHash.slice(1))?.scrollIntoView();
  }
  if (requestedClip && !isPolicySkill) {
    const button = [...$('detail').querySelectorAll('[data-video]')].find(item => item.dataset.video === requestedClip || item.dataset.video === `${id}:${requestedClip}`);
    if (button) button.closest('.clip').scrollIntoView({block:'center'});
  }
}

async function loadArchive() {
  if (!sessions.length) sessions = (await (await fetch(`/api/sessions?dashboard=${dashboard}`)).json()).sessions;
}

async function showSession(id) {
  disposeCurveViews();
  ++viewGeneration; catalogView?.dispose(); catalogView = null;
  $('detail').classList.remove('catalog-page');
  selected = id; modelViewer?.dispose(); modelViewer = null;
  const row = sessions.find(item => item.id === id);
  if (!row) return;
  if (row.result_id) return showResult(row.result_id, {keepPage: true});
  history.replaceState(null, '', `${pagePath(page)}?session=${encodeURIComponent(id)}`);
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

async function setPage(nextPage, preserveQuery = false) {
  disposeCurveViews();
  ++viewGeneration; catalogView?.dispose(); catalogView = null;
  $('detail').classList.remove('catalog-page');
  page = nextPage;
  mode = nextPage === 'archive' ? 'archive' : 'published';
  selected = null; workspace = 'All'; modelViewer?.dispose(); modelViewer = null;
  $('detail').innerHTML = '<p>Loading evidence…</p>';
  if (mode === 'archive') await loadArchive();
  if (!preserveQuery) history.replaceState(null, '', pagePath(page));
  renderList();
  const params = new URLSearchParams(location.search);
  if (mode === 'published' && params.get('skill')) {
    await showResult({skill: params.get('skill'), version: params.get('version')});
    return;
  }
  const requested = mode === 'archive' ? params.get('session') : (params.get('version_id') || params.get('result'));
  if (requested && mode === 'published') { await showResult(requested); return; }
  if (requested && mode === 'archive') { await showSession(requested); return; }
  const list = records();
  const initial = (page === 'policies' ? list[0] : page === 'perception' ? (list.find(r => r.id === 'spectacular-room-world002-20261027') || list[0]) : list[0]);
  if (initial) mode === 'archive' ? showSession(initial.id) : showResult(initial.version_id || initial.id);
  else renderEmptyState();
}

async function setMode(next) {
  if (!isDesign) return setPage(next === 'archive' ? 'archive' : 'policies', true);
  disposeCurveViews();
  ++viewGeneration; catalogView?.dispose(); catalogView = null;
  $('detail').classList.remove('catalog-page');
  mode = next; selected = null; workspace = 'All'; modelViewer?.dispose(); modelViewer = null;
  $('published-mode')?.classList.toggle('active', next === 'published');
  $('archive-mode')?.classList.toggle('active', next === 'archive');
  $('detail').innerHTML = '<p>Loading evidence…</p>';
  if (next === 'archive') await loadArchive();
  const requestedSection = new URLSearchParams(location.search).get('section');
  if (isDesign && requestedSection) workspace = requestedSection;
  renderList();
  const params = new URLSearchParams(location.search);
  if (isDesign && next === 'published' && params.get('view') === 'bom') {await showRobotBom(); return;}
  const hashResult = location.hash && !location.hash.startsWith('#catalog-group-') && location.hash !== '#design-catalog'
    ? decodeURIComponent(location.hash.slice(1))
    : null;
  const requested = next === 'archive' ? params.get('session') : (params.get('result') || hashResult);
  if (requested && next === 'published') {await showResult(requested); return;}
  if (isDesign && next==='published' && !requested && requestedSection) {
    const first=records().find(row=>belongs(row,workspace));
    if(first) return showResult(first.id);
    $('detail').innerHTML='<h2>'+escapeHtml(workspace)+'</h2><p>No current studies in this section. Earlier work is retained in Archive.</p>';
    return;
  }
  if (isDesign && next==='published' && !requested) {
    $('results').innerHTML='<p class="muted">Choose a body section to see its design studies.</p>';
    $('detail').innerHTML='<div class="design-welcome"><span class="eyebrow">DESIGN EXPLORER</span><h2>Start with a part of the robot.</h2><p>Select a joint or body part on the drawing, or choose its name in the section list.</p><div class="cards"><div class="card"><strong>Current body</strong>Design studies show the selected complete printed body and its electronics. Studies marked Selected are the owner’s choice for that part.</div><div class="card"><strong>Versions from Git</strong>Each study opens its current version. Earlier versions are named by date and commit.</div><div class="card"><strong>History</strong>Earlier bodies, finished earlier designs and alternatives are in Archive, each with its reason.</div></div><img src="/body-map.png" alt="APE0 aesthetic reference, showing front, side and articulated views"></div>';
    return;
  }
  const initial = records().find(row => row.id === requested) || (isDesign && workspace !== 'All' ? records().find(row => belongs(row, workspace)) : null) || (next === 'published' ? results.find(row => row.id === 'cove50-cad-competition-20261016') : null) || records()[0];
  if (initial) next === 'archive' ? showSession(initial.id) : showResult(initial.id);
  else $('detail').textContent = 'No evidence records yet.';
}

async function start() {
  if (isDesign) {
    selection = await (await fetch('/api/design-selection')).json();
    $('design-status').textContent = `${new Set(Object.values(selection)).size} selected studies cover ${Object.keys(selection).length} sections`;
    results = await (await fetch(`/api/results?dashboard=${dashboard}`)).json();
    await setMode(new URLSearchParams(location.search).has('session') ? 'archive' : 'published');
    return;
  }
  results = await (await fetch(`/api/results?dashboard=${dashboard}`)).json();
  await setPage(initialLearningPage(), true);
}

async function showRobotBom() {
  disposeCurveViews();
  const generation = ++viewGeneration;
  catalogView?.dispose(); catalogView = null;
  modelViewer?.dispose(); modelViewer = null;
  mode = 'published'; selected = null;
  history.replaceState(null, '', '/design/?view=bom');
  $('published-mode')?.classList.add('active'); $('archive-mode')?.classList.remove('active');
  renderList();
  $('detail').classList.remove('single-study');
  $('detail').classList.add('catalog-page');
  const {mountRobotBom} = await import('./design-bom.js?v=bom-ranges-20261123');
  if (generation !== viewGeneration) return;
  const view = await mountRobotBom($('detail'), results, () => generation === viewGeneration);
  if (generation !== viewGeneration) {view.dispose(); return;}
  catalogView = view;
}

if (isDesign && $('robot-bom')) $('robot-bom').onclick = showRobotBom;
$('search').addEventListener('input', renderList);
if ($('published-mode')) $('published-mode').onclick = () => setMode('published');
if ($('archive-mode')) $('archive-mode').onclick = () => setMode('archive');
document.querySelectorAll('#top-nav [data-page]').forEach(link => {
  link.onclick = event => {
    event.preventDefault();
    setPage(link.dataset.page, false);
  };
});
$('refresh').onclick = async () => {
  results = await (await fetch(`/api/results?dashboard=${dashboard}`)).json();
  sessions = [];
  if (isDesign && new URLSearchParams(location.search).get('view') === 'bom') { await showRobotBom(); return; }
  if (mode === 'archive' || page === 'archive') await loadArchive();
  renderList();
  if (selected) (mode === 'archive' || page === 'archive') ? showSession(selected) : showResult(selected);
};
window.addEventListener('hashchange', () => {
  const hashId = location.hash && !location.hash.startsWith('#catalog-group-') && location.hash !== '#design-catalog'
    ? decodeURIComponent(location.hash.slice(1))
    : null;
  if (hashId && mode === 'published' && results.some(r => r.id === hashId || r.study_id === hashId)) {
    const match = results.find(r => r.id === hashId) || results.find(r => r.study_id === hashId);
    if (match) showResult(match.id);
  }
});
start().catch(error => { $('detail').textContent = `Could not load dashboard: ${error.message}`; });
