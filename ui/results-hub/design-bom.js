import {esc} from './readable.js?v=studies-final-20261020';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';

const amount = value => value == null ? 'Unknown' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 4});
const endpoint = id => `/api/design-bom?result=${encodeURIComponent(id)}`;
async function request(url, value) {
  const response = await fetch(url, value ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'BOM unavailable');
  return data;
}
function links(refs = []) {
  return refs.map(ref => {
    const url = ref.path ? `/asset?path=${encodeURIComponent(ref.path)}` : /^https?:\/\//.test(ref.url || '') ? ref.url : null;
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(ref.label || ref.path || ref.url)}</a>` : '';
  }).join(' · ');
}
function summary(totals) {
  const costs = Object.entries(totals.known_cost_by_currency).map(([currency, value]) => `${esc(currency)} ${amount(value)}`).join(' · ');
  return `<div class="bom-totals"><div><strong>${costs || 'No priced lines'}</strong><span>Known cost subtotal · currencies kept separate</span></div><div><strong>${amount(totals.known_mass_g)} g</strong><span>Known mass subtotal</span></div><div><strong>${totals.line_count} part lines</strong><span>${totals.missing_cost_lines} with unknown cost · ${totals.missing_mass_lines} with unknown mass · ${totals.missing_quantity_lines} with unknown quantity · ${totals.unreferenced_lines} without sources</span></div></div>`;
}
function lineTable(lines, robot = false) {
  if (!lines.length) return '<p class="muted">No part lines recorded. Subtotals do not represent a complete robot.</p>';
  return `<div class="matrix bom-table"><table><caption>${robot ? 'Included parts across selected assemblies' : 'Parts per one assembly of this study version'}</caption><thead><tr>${robot ? '<th>Study version</th>' : ''}<th>Part</th><th>Quantity</th><th>Unit cost</th><th>Extended cost</th><th>Mass</th><th>Purchase & source details</th></tr></thead><tbody>${lines.map(item => `<tr>${robot ? `<td><a href="/design/?result=${encodeURIComponent(item.result_id)}">${esc(designStudyName({title:item.assembly,study_version:item.study_version}))}</a><small>${esc(item.study_version)} · ×${item.assembly_count} assemblies · BOM r${item.bom_revision}</small></td>` : ''}<td><strong>${esc(item.name)}</strong><small>${esc([item.part_number, item.category, item.material].filter(Boolean).join(' · '))}</small></td><td>${amount(item.quantity)} ${esc(item.unit)}</td><td>${item.unit_cost == null ? 'Unknown' : `${esc(item.currency)} ${amount(item.unit_cost)}`}<small>${esc(item.cost_basis)}</small></td><td>${item.extended_cost == null ? 'Unknown' : `${esc(item.currency)} ${amount(item.extended_cost)}`}</td><td>${amount(item.extended_mass_g)} g<small>${amount(item.unit_mass_g)} g / ${esc(item.unit)}</small></td><td><details><summary>${esc(item.supplier || 'Supplier / evidence')}</summary><p>${esc(item.part_number || 'No part number')}</p><p>Price date: ${esc(item.price_date || 'Unknown')} · Lead time: ${amount(item.lead_time_days)} days</p><p>${links(item.source_refs) || 'No source reference recorded.'}</p><p>${esc(item.notes)}</p></details></td></tr>`).join('')}</tbody></table></div>`;
}
const exclusions = values => values?.length ? `<p><strong>Excluded / not counted:</strong> ${values.map(esc).join(' · ')}</p>` : '';
const exportsMarkup = url => `<div class="bom-actions"><a href="${url}&format=csv" download>Export parts CSV</a><a href="${url}" download="design-bom.json">Export BOM & context JSON</a></div>`;
function input(name, label, value = '', type = 'text') {
  return `<label>${esc(label)}<input name="${name}" type="${type}" value="${esc(value ?? '')}" ${type === 'number' ? 'min="0" step="any"' : ''}></label>`;
}
function itemEditor(item = {}, index = 0) {
  return `<fieldset class="bom-item-editor" data-index="${index}"><legend>${esc(item.name || 'New part')}</legend><div class="bom-fields">
    ${input('name', 'Part name', item.name)}${input('id', 'Stable part ID', item.id || `part-${crypto.randomUUID()}`)}
    ${input('quantity', 'Quantity per assembly', item.quantity, 'number')}${input('unit', 'Quantity unit', item.unit || 'ea')}
    ${input('unit_cost', 'Cost per quantity unit', item.unit_cost, 'number')}${input('currency', 'Currency (CHF, EUR, USD…)', item.currency)}
    <label>Cost basis<select name="cost_basis">${['unknown', 'estimate', 'quote', 'paid'].map(v => `<option ${v === (item.cost_basis || 'unknown') ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
    ${input('unit_mass_g', 'Mass per quantity unit (g)', item.unit_mass_g, 'number')}${input('supplier', 'Supplier', item.supplier)}${input('part_number', 'Manufacturer part number', item.part_number)}
    ${input('category', 'Category', item.category)}${input('material', 'Material / process', item.material)}${input('price_date', 'Price date', item.price_date, 'date')}${input('lead_time_days', 'Lead time (days)', item.lead_time_days, 'number')}
    <label class="bom-wide">Sources (one URL or published file path per line)<textarea name="sources" rows="2">${esc((item.source_refs || []).map(r => r.path || r.url).join('\n'))}</textarea></label>
    <label class="bom-wide">Part notes / price assumptions<textarea name="notes" rows="2">${esc(item.notes)}</textarea></label></div><button type="button" data-remove-part>Remove part</button></fieldset>`;
}
function editor(bom) {
  return `<details class="bom-edit"><summary>${bom ? 'Edit this version’s BOM' : 'Add a BOM for this version'}</summary><form class="inside bom-form"><p>Quantities describe one assembly. Blank quantities, prices and mass stay unknown. Include purchased components only once; record costs for printed parts and fabrication explicitly.</p><div class="bom-fields">${input('scope', 'Assembly scope', bom?.scope)}<label>Scope coverage (author’s declaration)<select name="completeness"><option value="partial">Partial inventory</option><option value="complete" ${bom?.completeness === 'complete' ? 'selected' : ''}>All parts in stated scope recorded</option></select></label><label class="bom-wide">Cost assumptions / notes<textarea name="notes" rows="2">${esc(bom?.notes)}</textarea></label><label class="bom-wide">Exclusions (one per line)<textarea name="exclusions" rows="2">${esc((bom?.exclusions || []).join('\n'))}</textarea></label></div><div class="bom-edit-items">${(bom?.items || []).map(itemEditor).join('')}</div><div class="bom-actions"><button type="button" data-add-part>Add part</button><button type="submit">Save BOM for this version</button><button type="button" data-reload-bom>Reload saved BOM</button><span role="status" class="bom-save-status"></span></div></form></details>`;
}

export async function mountStudyBom(host, resultId, current = () => true) {
  let active = true, report;
  const events = new AbortController();
  const valid = () => active && current();
  host.innerHTML = '<p role="status">Loading bill of materials…</p>';
  async function load() {
    try {
      report = await request(endpoint(resultId));
      if (valid()) render();
    } catch (error) { if (valid()) host.textContent = `BOM could not load: ${error.message}`; }
  }
  function render() {
    const bom = report.bom;
    host.innerHTML = `<h3>Bill of materials & costs</h3><p>${bom ? esc(bom.scope || 'Assembly scope not stated.') : 'No structured BOM registered for this version. Existing BOM documents remain in its source files.'}</p>${bom ? `<p class="muted">BOM revision ${report.revision} · ${esc(report.updated_at)} UTC · ${bom.completeness === 'complete' ? 'Author declares all parts in stated scope recorded' : 'Partial inventory'}</p>${summary(report.totals)}${bom.notes ? `<p>${esc(bom.notes)}</p>` : ''}${exclusions(bom.exclusions)}${lineTable(report.lines)}${exportsMarkup(endpoint(resultId))}` : ''}${editor(bom)}`;
  }
  host.addEventListener('click', event => {
    if (event.target.closest('[data-add-part]')) host.querySelector('.bom-edit-items').insertAdjacentHTML('beforeend', itemEditor());
    if (event.target.closest('[data-remove-part]')) event.target.closest('fieldset').remove();
    if (event.target.closest('[data-reload-bom]')) load();
  }, {signal: events.signal});
  host.addEventListener('input', () => { const status = host.querySelector('.bom-save-status'); if (status) status.textContent = 'Unsaved changes'; }, {signal: events.signal});
  host.addEventListener('submit', async event => {
    const form = event.target.closest('.bom-form');
    if (!form) return;
    event.preventDefault();
    const status = form.querySelector('.bom-save-status');
    const controls = [...form.querySelectorAll('input,select,textarea,button')];
    try {
      const fields = form.querySelector('.bom-fields');
      const val = name => fields.querySelector(`[name="${name}"]`).value;
      const bom = {scope: val('scope'), completeness: val('completeness'), notes: val('notes'), exclusions: val('exclusions').split('\n').map(s => s.trim()).filter(Boolean), items: []};
      for (const fieldset of form.querySelectorAll('.bom-item-editor')) {
        const item = Object.fromEntries([...fieldset.querySelectorAll('input,select,textarea')].map(field => [field.name, field.value]));
        for (const name of ['quantity', 'unit_cost', 'unit_mass_g', 'lead_time_days']) item[name] = item[name] === '' ? null : Number(item[name]);
        const oldRefs = report.bom?.items.find(i => i.id === item.id)?.source_refs || [];
        item.source_refs = item.sources.split('\n').map(s => s.trim()).filter(Boolean).map(value => oldRefs.find(ref => (ref.path || ref.url) === value) || (/^https?:\/\//.test(value) ? {label: value, url: value} : {label: value, path: value}));
        delete item.sources;
        bom.items.push(item);
      }
      status.textContent = 'Saving…';
      controls.forEach(control => {control.disabled = true;});
      const saved = await request('/api/design-bom', {result_id: resultId, expected_revision: report.revision, bom});
      if (!valid()) return;
      report = saved; render();
      host.querySelector('.bom-save-status').textContent = 'Saved';
    } catch (error) { if (valid()) status.textContent = `Not saved: ${error.message} Your edits remain here.`; }
    finally {controls.forEach(control => {control.disabled = false;});}
  }, {signal: events.signal});
  await load();
  return {dispose() {active = false; events.abort();}};
}

export async function mountRobotBom(host, results, current = () => true) {
  let active = true, report;
  const events = new AbortController();
  const valid = () => active && current();
  const choices = results.filter(r => r.record_kind === 'study').sort((a, b) => designStudyName(a).localeCompare(designStudyName(b)) || String(b.study_version).localeCompare(String(a.study_version), undefined, {numeric: true}));
  host.innerHTML = '<p role="status">Loading robot BOM…</p>';
  function options(query = '') {
    return choices.filter(r => `${r.title} ${r.study_version} ${r.design_sections?.join(' ')} ${r.id}`.toLowerCase().includes(query.toLowerCase()))
      .map(r => `<option value="${esc(r.id)}">${esc(r.design_sections?.join(' / '))} · ${esc(designStudyName(r))} · ${esc(r.study_version)}</option>`).join('');
  }
  function assemblyRow(assembly) {
    const row = choices.find(r => r.id === assembly.result_id);
    const bom = report.assemblies.find(a => a.result_id === assembly.result_id);
    return `<tr data-result="${esc(assembly.result_id)}"><td><a href="/design/?result=${encodeURIComponent(assembly.result_id)}">${esc(row ? designStudyName(row) : assembly.result_id)}</a><small>${esc(row?.study_version)} · ${esc(row?.design_sections?.join(' · '))}</small></td><td><input aria-label="Assembly count for ${esc(row ? designStudyName(row) : assembly.result_id)}" type="number" min="1" step="1" value="${assembly.count}"></td><td>${bom ? bom.bom ? `${bom.bom.items.length} lines · BOM r${bom.revision}${bom.bom.completeness === 'partial' ? ' · partial' : ''}` : 'No BOM registered' : 'Save to calculate'}</td><td><button type="button" data-remove-assembly>Remove</button></td></tr>`;
  }
  function render() {
    host.innerHTML = `<div class="meta"><span>Whole robot</span><span>Planning composition</span></div><h2>Robot BOM & costs</h2><p>Choose exact study versions and how many assemblies the robot needs (for example, two knees). Preferences and mechanical selection stay separate. Review shared hardware ownership before including overlapping studies.</p><form class="robot-bom-form"><div class="bom-fields">${input('title', 'BOM name', report.plan.title)}<label>Planning notes<textarea name="notes" rows="2">${esc(report.plan.notes)}</textarea></label><label class="bom-wide">Exclusions / unresolved scope (one per line)<textarea name="exclusions" rows="2">${esc(report.plan.exclusions.join('\n'))}</textarea></label></div><div class="bom-add-assembly"><label>Find a study version<input type="search" data-bom-search placeholder="Body section, mechanism, version…"></label><label>Study version<select data-bom-choice>${options()}</select></label><button type="button" data-add-assembly>Add assembly</button></div><div class="matrix"><table><caption>Explicit robot composition · no nested assemblies added automatically</caption><thead><tr><th>Study version</th><th>Assembly count</th><th>BOM coverage</th><th>Action</th></tr></thead><tbody class="bom-assemblies">${report.plan.assemblies.map(assemblyRow).join('')}</tbody></table></div><div class="bom-actions"><button type="submit">Save robot BOM & calculate</button><button type="button" data-reload-plan>Reload saved composition</button><span role="status" class="bom-save-status">${report.revision ? `Saved composition r${report.revision} · ${esc(report.updated_at)} UTC` : 'No composition saved yet'}</span></div></form><section class="bom-saved"><h3>Saved composition totals</h3><p>Subtotals include only recorded parts and known amounts. Changes above appear here after saving. Prices use the latest saved BOM estimate for each pinned design version; exports record those BOM revisions.</p>${summary(report.totals)}${exclusions(report.plan.exclusions)}${report.missing_bom_versions.length ? `<p class="bom-notice">${report.missing_bom_versions.length} included study versions have no BOM. Their parts, costs and mass are uncounted.</p>` : ''}${report.empty_bom_versions.length ? `<p class="bom-notice">${report.empty_bom_versions.length} included BOMs contain no part lines.</p>` : ''}${report.assemblies.filter(a => a.bom?.completeness === 'partial').length ? '<p class="bom-notice">Some included BOMs are partial inventories. Review their scope and exclusions below.</p>' : ''}${report.uncovered_sections.length ? `<p><strong>Sections without an explicitly included study:</strong> ${report.uncovered_sections.map(esc).join(' · ')}. Inclusion in a cross-section study still needs an ownership check.</p>` : ''}${Object.keys(report.overlapping_sections).length ? `<p class="bom-notice"><strong>Review potential overlap:</strong> ${Object.entries(report.overlapping_sections).map(([section, names]) => `${esc(section)} (${names.map(esc).join(' / ')})`).join(' · ')}. Shared parts are counted in every included BOM; remove duplicates at their source.</p>` : ''}${report.assemblies.length ? `<details><summary>Assembly scope, exclusions & assumptions</summary><div class="inside">${report.assemblies.map(a => `<h4>${esc(designStudyName(a))} · ${esc(a.study_version)} ×${a.count}</h4><p>${esc(a.bom?.scope || 'No assembly scope recorded.')}</p><p>${esc(a.bom?.notes || '')}</p>${exclusions(a.bom?.exclusions)}`).join('')}</div></details>` : '<p>No assemblies included yet. Add study versions to build a robot estimate.</p>'}${exportsMarkup('/api/robot-bom?export=1')}${lineTable(report.lines, true)}</section>`;
  }
  async function load() {
    try {report = await request('/api/robot-bom'); if (valid()) render();}
    catch (error) {if (valid()) host.textContent = `Robot BOM could not load: ${error.message}`;}
  }
  host.addEventListener('click', event => {
    if (event.target.closest('[data-reload-plan]')) load();
    if (event.target.closest('[data-add-assembly]')) {
      const id = host.querySelector('[data-bom-choice]').value;
      if (!id) return;
      const picked = choices.find(r => r.id === id);
      const duplicate = [...host.querySelectorAll('[data-result]')].some(row => choices.find(r => r.id === row.dataset.result)?.study_id === picked.study_id);
      if (duplicate) {host.querySelector('.bom-save-status').textContent = 'This study is already included. Adjust its assembly count, or remove it before choosing another version.'; return;}
      host.querySelector('.bom-assemblies').insertAdjacentHTML('beforeend', assemblyRow({result_id: id, count: 1}));
      host.querySelector('.bom-save-status').textContent = 'Unsaved composition changes';
    }
    if (event.target.closest('[data-remove-assembly]')) {event.target.closest('tr').remove(); host.querySelector('.bom-save-status').textContent = 'Unsaved composition changes';}
  }, {signal: events.signal});
  host.addEventListener('input', event => {
    if (event.target.matches('[data-bom-search]')) {host.querySelector('[data-bom-choice]').innerHTML = options(event.target.value); return;}
    if (event.target.closest('.robot-bom-form')) host.querySelector('.bom-save-status').textContent = 'Unsaved composition changes';
  }, {signal: events.signal});
  host.addEventListener('submit', async event => {
    event.preventDefault();
    const form = event.target, status = form.querySelector('.bom-save-status');
    const plan = {title: form.elements.title.value, notes: form.elements.notes.value,
      exclusions: form.elements.exclusions.value.split('\n').map(s => s.trim()).filter(Boolean),
      assemblies: [...form.querySelectorAll('[data-result]')].map(row => ({result_id: row.dataset.result, count: Number(row.querySelector('input').value)}))};
    const controls = [...form.querySelectorAll('input,select,textarea,button')];
    controls.forEach(control => {control.disabled = true;}); status.textContent = 'Saving…';
    try {const saved = await request('/api/robot-bom', {plan, expected_revision: report.revision}); if (valid()) {report = saved; render();}}
    catch (error) {if (valid()) status.textContent = `Not saved: ${error.message} Your edits remain here.`;}
    finally {controls.forEach(control => {control.disabled = false;});}
  }, {signal: events.signal});
  await load();
  return {dispose() {active = false; events.abort();}};
}
