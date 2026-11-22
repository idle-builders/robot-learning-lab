import {inspectionMarkup, bindInspectionTools, navigationMarkup} from './cad-catalog.js?v=body-view-20261122';
import {designEvidence} from './design-evidence.js?v=fem-pictures-20261024';
import {esc, briefText, evidenceText, shortText} from './readable.js?v=studies-final-20261020';
import {mountCatalog} from './catalog.js?v=body-view-20261122';
import {createCadViewer} from './cad-viewer.js?v=body-view-20261122';
import {mountStudyBom} from './design-bom.js?v=body-view-20261122';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';
// Where this version's sources are: a folder at a commit, or a history tag to check out.
const sourceLine = row => {
  const commit = String(row.source_commit || '').replace('+dirty', '').slice(0, 9);
  if (row.history_tag) return `<span title="Sources of this version, as built">git checkout ${esc(row.history_tag)}${row.source_path ? ` · ${esc(row.source_path)}` : ''}</span>`;
  if (row.source_path) return `<span title="Sources of this version">${esc(row.source_path)}${commit ? ` @ ${esc(commit)}` : ''}</span>`;
  return row.archived_at ? '<span>Outputs only: no build code was kept for this version</span>' : '';
};
import {assetUrl as asset} from './asset-url.js';

// Only use models actually published for this option's exact source directory.
// This adds an inspection view; it never creates an articulation or alters evidence.
function withCad(option, assets) {
  // A related source model is not an assembly of a proposed schematic.
  if (option.visual_kind === 'engineering_schematic' || option.visual_kind === 'concept_schematic') return {...option, cad: null};
  if (option.cad?.views?.length) return {...option, cad: {...option.cad, motion_contract_path: option.cad.motion_contract_path || option.source_refs?.find(ref => ref.path?.endsWith('/kinematic-mass-contract.json'))?.path}};
  const refs = option.source_refs || [];
  const directories = refs.map(ref => ref.path?.split('/').slice(0, -1).join('/')).filter(Boolean);
  const explicit = new Set(refs.map(ref => ref.path).filter(path => path?.endsWith('.glb')));
  const manifest = refs.find(ref => /\/design\.json$/.test(ref.path || ''));
  const native = manifest?.path.match(/\/([A-C])\/([^/]+)\/design\.json$/);
  const neutralName = native ? `${native[1]}-${native[2]}-neutral-native.glb` : null;
  const models = assets.filter(item => item.kind === 'model' && item.path.endsWith('.glb') &&
    (item.path.split('/').at(-1) === neutralName || explicit.has(item.path) || directories.includes(item.path.split('/').slice(0, -1).join('/'))));
  if (!models.length) return option;
  const step = refs.find(ref => /\.step$/i.test(ref.path || ''));
  return {...option, cad: {views: models.map(model => ({label: model.title, model_path: model.path})),
    step_path: step?.path, parts_path: manifest?.path, scope: 'Published native CAD exports. Saved configurations are inspection evidence; contact, clearance and physical behavior require their own evidence.'}};
}

export async function mountDesignDetail(host, row, section, widgets, current) {
  let active = true, mounted = null, viewer = null, groupGeneration = 0;
  const events = new AbortController();
  const groups = row.sections.filter(s => s.payload?.kind === 'design_catalog').flatMap(s => s.payload.groups.map(group => ({...group, catalogTitle: s.payload.title || s.title})))
    .filter(group => section === 'All' || group.design_sections?.includes(section));
  const params = new URLSearchParams(location.search);
  const anchorGroup = location.hash.startsWith('#catalog-group-') ? location.hash.slice('#catalog-group-'.length) : null;
  let group = groups.find(g => g.id === (params.get('group') || anchorGroup)) || groups[0];
  const currentPaths = group?.options.flatMap(option => option.cad?.views?.map(view => view.model_path) || []) || [];
  const modelOrder = new Map(currentPaths.map((path, index) => [path, index]));
  // Per-part exports belong to the BOM, keeping the assembly library readable.
  // Start with this version's declared neutral view, ahead of retained source/attempt models.
  const schematic = group?.options.every(option => ['engineering_schematic', 'concept_schematic'].includes(option.visual_kind));
  const models = (schematic ? [] : row.assets).filter(item => item.kind === 'model' && item.path.endsWith('.glb') && !item.path.includes('/parts/'))
    .sort((a, b) => (modelOrder.get(a.path) ?? currentPaths.length) - (modelOrder.get(b.path) ?? currentPaths.length));
  const files = row.assets.filter(item => item.kind === 'data');
  const images = row.assets.filter(item => item.kind === 'image');
  // A generated body view (robot/body/bom/publish_body.py) feeds the part view, the CAD BOM and the CAD part card.
  const bodyView = row.sections.find(s => s.payload?.kind === 'body_view')?.payload || null;
  const otherSections = row.sections.filter(s => !['design_catalog', 'body_view'].includes(s.payload?.kind));
  host.classList.add('catalog-page', 'single-study');
  host.innerHTML = `<div class="meta"><span>${esc(section === 'All' ? row.design_sections.join(' · ') : section)}</span><span>${row.archived_at ? 'Archived design study' : 'Design study'}</span></div><h2>${esc(designStudyName(row))}</h2>${row.archived_at ? `<div class="card"><strong>Archived · retained for reference</strong><p>${esc(row.archive_reason)}</p></div>` : group?.options[0]?.use_context ? `<div class="card"><strong>Use of this study</strong><p>${esc(group.options[0].use_context)}</p></div>` : ''}<div class="study-strip"><span>One design variant</span><span>${row.versions.length === 1 ? 'One version' : `${row.versions.length} versions`}${row.id === row.study_id ? '' : ' · viewing an earlier version'}</span><span>${models.length} CAD scenes</span><details class="text-more"><summary>Study overview</summary>${evidenceText(row.observation || row.summary,3)}</details>${sourceLine(row)}</div>
    <div class="exploration-bar"><label for="design-version">Version<select id="design-version">${[...row.versions].sort((a,b)=>String(b.version_at||'').localeCompare(String(a.version_at||''))).map(v=>`<option value="${esc(v.id)}" ${v.id === row.id ? 'selected' : ''} title="${esc(v.legacy_version ? `Published as ${v.legacy_version}` : '')}">${esc(String(v.version_at||'').slice(0,10))} · commit ${esc(String(v.source_commit||v.study_version||'').slice(0,7))}${v.id === row.study_id ? ' · current' : ''}</option>`).join('')}</select></label><div class="preference-actions"><button id="prefer-study" type="button">Select for ${esc(section === 'All' ? row.design_sections[0] : section)}</button><span id="preference-status" role="status"></span></div></div>${groups.length ? '<div id="focused-designs"></div>' : ''}
    ${models.length ? `<details class="cad-library" ${groups.length ? '' : 'open'}><summary>CAD model library · ${models.length} published scenes</summary><div class="inside"><label for="study-model">Model / saved configuration</label><select id="study-model">${models.map((m, i) => `<option value="${i}">${esc(m.title)}</option>`).join('')}</select><div class="study-cad-canvas"></div>${navigationMarkup()}${inspectionMarkup()}<p class="study-cad-label" role="status">Open the model library to inspect a scene.</p><button type="button" id="fit-study">Fit view</button><button type="button" id="snapshot-study">Snapshot & annotate</button><a id="download-study" download>Download source model</a></div></details>` : ''}
    ${widgets.videoHost()}${row.clips.length ? `<details id="design-recordings"><summary>Simulation & inspection recordings · ${row.clips.length}</summary><div class="inside clip-list">${row.clips.map(clip=>`<div class="clip"><strong>${esc(clip.title)}</strong>${briefText(clip.observation, 'Recording context')}<button type="button" data-design-video="${esc(clip.id)}">Watch recording</button></div>`).join('')}</div></details>` : ''}<div class="cards study-evidence"><div class="card"><strong>Measured evidence</strong>${evidenceText(row.measurements, 2)}</div><div class="card"><strong>Open engineering questions</strong>${evidenceText(row.limitations, 2)}</div></div>
    ${!groups.length && !models.length ? '<p class="empty-cad">No interactive CAD was published for this study. Inspect its source images and reports below.</p>' : ''}
    <section id="study-bom" class="study-bom"></section>
    ${widgets.renderGrades(row.grades)}
    ${images.length ? `<details><summary>Source renders & drawings · ${images.length}</summary><div class="inside gallery">${images.map(m => widgets.imageWidget(m.title, m.path)).join('')}</div></details>` : ''}
    ${files.length ? `<details><summary>Reports & source files · ${files.length}</summary><div class="inside clip-list">${files.map(f => widgets.fileWidget(f.title, f.path)).join('')}</div></details>` : ''}
    ${otherSections.map(s => `<details><summary>${esc(s.title)}</summary><div class="inside">${s.note ? evidenceText(s.note) : ''}${s.payload?.kind === 'design_evidence' ? designEvidence(s.payload.items) : widgets.renderMatrix(s.payload)}</div></details>`).join('')}
    <details><summary>Provenance & study context</summary><div class="inside">${evidenceText(row.summary)}<ul class="source-paths">${row.provenance.map(p => `<li>${esc(p)}</li>`).join('')}</ul></div></details>`;
  widgets.bindCommon();
  host.addEventListener('click', async event => {
    const button = event.target.closest('[data-design-video]');
    if (!button || !active || !current()) return;
    const id = button.dataset.designVideo, clip = row.clips.find(c => c.id === id);
    if (!clip) return;
    try {
      const response = await fetch(`/api/video?id=${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error('Recording unavailable');
      const data = await response.json(); if (active && current()) widgets.playVideo(data.url, clip.title);
    } catch (error) {button.textContent = error.message;}
  }, {signal:events.signal});
  async function showGroup() {
    const generation = ++groupGeneration;
    mounted?.dispose(); mounted = null;
    const options = group.options.map(option => withCad({...option, title: designStudyName({title: option.title, study_version: row.study_version})}, row.assets))
      .map(option => bodyView && option.cad ? {...option, cad: {...option.cad, body_view: bodyView}} : option);
    const result = await mountCatalog(host.querySelector('#focused-designs'), row.id,
      {title: group.title, description: '', groups: [{...group, options}]}, {compact: true});
    if (!active || !current() || generation !== groupGeneration) { result.dispose(); return; }
    mounted = result;
    const url = new URL(location.href);
    if (section === 'All') url.searchParams.delete('section'); else url.searchParams.set('section', section);
    url.searchParams.set('group', group.id);
    history.replaceState(null, '', url.pathname + url.search + url.hash);
  }
  host.querySelector('#design-version').onchange=event=>widgets.showResult(event.target.value);
  const prefSection=section==='All'?row.design_sections[0]:section;
  const prefButton=host.querySelector('#prefer-study');
  if (row.archived_at) {prefButton.disabled=true;prefButton.textContent='Archived · restore before selecting';}
  if(widgets.selection[prefSection]===row.study_id){prefButton.textContent=`Selected for ${prefSection}`;prefButton.disabled=true;}
  prefButton.onclick=async()=>{try{await widgets.select(prefSection,row.study_id);if(current()){prefButton.textContent=`Selected for ${prefSection}`;prefButton.disabled=true;host.querySelector('#preference-status').textContent='Saved to robot/design.json';}}catch(error){host.querySelector('#preference-status').textContent=error.message;}};
  // A part opened in the part view is selected in the study's CAD view too (same names as the GLB nodes).
  const selectInCad = name => host.querySelector('#focused-designs .design-option')?.dispatchEvent(new CustomEvent('cad-select-part', {bubbles: true, detail: {name}}));
  const bomView = await mountStudyBom(host.querySelector('#study-bom'), row.id, () => active && current(), bodyView, selectInCad);
  if (groups.length && active && current()) await showGroup();
  if (!active || !current()) return {dispose() {active = false; events.abort(); mounted?.dispose(); bomView.dispose();}};
  if (models.length) {
    const library = host.querySelector('.cad-library'), choice = host.querySelector('#study-model');
    const applyInspection=bindInspectionTools(library,()=>viewer);
    let navigationMode='orbit';
    library.querySelectorAll('[data-cad-navigation]').forEach(button=>button.onclick=()=>{
      navigationMode=button.dataset.cadNavigation;
      library.querySelectorAll('[data-cad-navigation]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
      viewer?.navigation(navigationMode);
    });
    async function loadModel() {
      if (!active || !current()) return;
      try {
        if (!viewer) viewer = createCadViewer(host.querySelector('.study-cad-canvas'), host.querySelector('.study-cad-label'));
        const model = models[Number(choice.value)];
        host.querySelector('#download-study').href = asset(model.path);
        await viewer.load(asset(model.path));
        viewer.navigation(navigationMode);
        applyInspection();
      } catch (error) { if (active && current()) host.querySelector('.study-cad-label').textContent = `Could not display CAD: ${error.message}. Download the source model to inspect it locally.`; }
    }
    library.ontoggle = () => { if (library.open) loadModel(); else {viewer?.dispose(); viewer = null;} };
    choice.onchange = loadModel;
    host.querySelector('#fit-study').onclick = () => viewer?.fit();
    host.querySelector('#snapshot-study').onclick = async () => {
      const shot = viewer?.snapshot(), model = models[Number(choice.value)];
      if (!shot) return;
      const {openFeedback} = await import('./cad-feedback.js?v=legend-only-20261120');
      openFeedback({snapshot: shot, mmPerUnit: 1000, context: {dashboard_url: location.href,
        study: {result_id: row.id, title: designStudyName(row), section: section === 'All' ? null : section},
        cad: {view_label: model.title, model_path: model.path, motion: 'model library (no joint sliders)'}, joints_deg: {}}});
    };
    if (library.open) await loadModel();
  }
  return {dispose() {active = false; events.abort(); ++groupGeneration; mounted?.dispose(); viewer?.dispose(); bomView.dispose();}};
}
