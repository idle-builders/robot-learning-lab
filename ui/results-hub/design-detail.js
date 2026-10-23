import {inspectionMarkup, bindInspectionTools, navigationMarkup} from './cad-catalog.js?v=pan-20261021';
import {designEvidence} from './design-evidence.js?v=evidence-links-20261021';
import {esc, briefText, evidenceText, shortText} from './readable.js?v=studies-final-20261020';
import {mountCatalog} from './catalog.js?v=hip-body-first-20261020';
import {createCadViewer} from './cad-viewer.js?v=pan-20261021';
import {mountStudyBom} from './design-bom.js?v=mechanism-names-20261021';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';
const asset = path => `/asset?path=${encodeURIComponent(path)}`;

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
  const otherSections = row.sections.filter(s => s.payload?.kind !== 'design_catalog');
  host.classList.add('catalog-page', 'single-study');
  host.innerHTML = `<div class="meta"><span>${esc(section === 'All' ? row.design_sections.join(' · ') : section)}</span><span>${row.archived_at ? 'Archived design study' : 'Design study'}</span></div><h2>${esc(designStudyName(row))}</h2>${row.archived_at ? `<div class="card"><strong>Archived · retained for reference</strong><p>${esc(row.archive_reason)}</p></div>` : group?.options[0]?.use_context ? `<div class="card"><strong>Use of this study</strong><p>${esc(group.options[0].use_context)}</p></div>` : ''}<div class="study-strip"><span>One design variant</span><span>${row.versions.length} retained versions</span><span>${models.length} CAD scenes</span><details class="text-more"><summary>Study overview</summary>${evidenceText(row.observation || row.summary,3)}</details></div>
    <div class="exploration-bar"><label for="design-version">Study version<select id="design-version">${[...row.versions].sort((a,b)=>String(b.study_version).localeCompare(String(a.study_version),undefined,{numeric:true})).map((v,i)=>`<option value="${esc(v.id)}" ${v.id === row.id ? 'selected' : ''}>${esc(v.study_version)}${i === 0 ? ' · latest' : ''}</option>`).join('')}</select></label><div class="preference-actions"><button id="prefer-study" type="button">Make preferred for ${esc(section === 'All' ? row.design_sections[0] : section)}</button><span id="preference-status" role="status"></span></div></div>${groups.length ? '<div id="focused-designs"></div>' : ''}
    ${models.length ? `<details class="cad-library" ${groups.length ? '' : 'open'}><summary>CAD model library · ${models.length} published scenes</summary><div class="inside"><label for="study-model">Model / saved configuration</label><select id="study-model">${models.map((m, i) => `<option value="${i}">${esc(m.title)}</option>`).join('')}</select><div class="study-cad-canvas"></div>${navigationMarkup()}${inspectionMarkup()}<p class="study-cad-label" role="status">Open the model library to inspect a scene.</p><button type="button" id="fit-study">Fit view</button><a id="download-study" download>Download source model</a></div></details>` : ''}
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
    const options = group.options.map(option => withCad({...option, title: designStudyName({title: option.title, study_version: row.study_version})}, row.assets));
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
  if (row.archived_at) {prefButton.disabled=true;prefButton.textContent='Archived · restore before preferring';}
  if(widgets.preferences.some(p=>p.section===prefSection&&p.study_id===row.study_id)){prefButton.textContent=`Preferred for ${prefSection}`;prefButton.disabled=true;}
  prefButton.onclick=async()=>{try{await widgets.prefer(prefSection,row.study_id);if(current()){prefButton.textContent=`Preferred for ${prefSection}`;prefButton.disabled=true;host.querySelector('#preference-status').textContent='Saved for this body section';}}catch(error){host.querySelector('#preference-status').textContent=error.message;}};
  const bomView = await mountStudyBom(host.querySelector('#study-bom'), row.id, () => active && current());
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
    if (library.open) await loadModel();
  }
  return {dispose() {active = false; events.abort(); ++groupGeneration; mounted?.dispose(); viewer?.dispose(); bomView.dispose();}};
}
