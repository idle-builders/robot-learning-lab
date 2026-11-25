import {esc} from './readable.js?v=studies-final-20261020';
import {designStudyName} from './design-names.js?v=mechanism-names-20261021';
import {assetUrl} from './asset-url.js';
import {bodyViewMarkup, bindBodyView} from './cad-bom.js?v=bom-only-20261125';

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
    const url = ref.path ? assetUrl(ref.path) : /^https?:\/\//.test(ref.url || '') ? ref.url : null;
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(ref.label || ref.path || ref.url)}</a>` : '';
  }).join(' · ');
}
function summary(totals) {
  const costs = Object.entries(totals.known_cost_by_currency).map(([currency, value]) => `${esc(currency)} ${amount(value)}`).join(' · ');
  return `<div class="bom-totals"><div><strong>${costs || 'No priced lines'}</strong><span>Known cost subtotal · currencies kept separate</span></div><div><strong>${amount(totals.known_mass_g)} g</strong><span>Known mass subtotal</span></div><div><strong>${totals.line_count} part lines</strong><span>${totals.missing_cost_lines} with unknown cost · ${totals.missing_mass_lines} with unknown mass · ${totals.missing_quantity_lines} with unknown quantity · ${totals.unreferenced_lines} without sources</span></div></div>`;
}
function lineTable(lines, robot = false) {
  if (!lines.length) return '<p class="muted">No part lines recorded. Subtotals do not represent a complete robot.</p>';
  return `<div class="matrix bom-table"><table><caption>${robot ? 'Parts of the body' : 'Parts of this record'}</caption><thead><tr>${robot ? '<th>Record</th>' : ''}<th>Part</th><th>Quantity</th><th>Unit cost</th><th>Extended cost</th><th>Mass</th><th>Purchase & source details</th></tr></thead><tbody>${lines.map(item => `<tr>${robot ? `<td><a href="/design/?result=${encodeURIComponent(item.result_id)}">${esc(designStudyName({title:item.assembly,study_version:item.study_version}))}</a><small>${esc(item.study_version)}</small></td>` : ''}<td><strong>${esc(item.name)}</strong><small>${esc([item.part_number, item.category, item.material].filter(Boolean).join(' · '))}</small></td><td>${amount(item.quantity)} ${esc(item.unit)}</td><td>${item.unit_cost == null ? 'Unknown' : `${esc(item.currency)} ${amount(item.unit_cost)}`}<small>${esc(item.cost_basis)}</small></td><td>${item.extended_cost == null ? 'Unknown' : `${esc(item.currency)} ${amount(item.extended_cost)}`}</td><td>${amount(item.extended_mass_g)} g<small>${amount(item.unit_mass_g)} g / ${esc(item.unit)}</small></td><td><details><summary>${esc(item.supplier || 'Supplier / evidence')}</summary><p>${esc(item.part_number || 'No part number')}</p><p>Price date: ${esc(item.price_date || 'Unknown')} · Lead time: ${amount(item.lead_time_days)} days</p><p>${links(item.source_refs) || 'No source reference recorded.'}</p><p>${esc(item.notes)}</p></details></td></tr>`).join('')}</tbody></table></div>`;
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

// view: the record's generated body view (robot/body/bom/publish_body.py), shown above the cost BOM of the same build;
// onPart(name) is called when a part card opens (the study page selects that part in its CAD view).
export async function mountStudyBom(host, resultId, current = () => true, view = null, onPart = () => {}) {
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
    // A section view lists the BOM lines of its parts; the totals stay those of the whole record.
    const lines = view?.section ? report.lines.filter(line => view.lines[line.id]) : report.lines;
    const origin = report.generated ? 'Generated by robot/body/bom/build.py from the body definitions (edit the definitions, not this BOM)' : `BOM revision ${report.revision} · ${esc(report.updated_at)} UTC`;
    host.innerHTML = `${view ? bodyViewMarkup(view) : ''}<h3>Bill of materials & costs</h3><p>${bom ? esc(bom.scope || 'Assembly scope not stated.') : 'No structured BOM registered for this version.'}</p>${bom ? `<p class="muted">${origin} · ${bom.completeness === 'complete' ? 'Author declares all parts in stated scope recorded' : 'Partial inventory'}</p>${summary(report.totals)}${view?.section ? `<p class="muted">Totals: the whole record. Lines below: the ${lines.length} lines with a part in ${esc(view.section)}.</p>` : ''}${bom.notes ? `<p>${esc(bom.notes)}</p>` : ''}${exclusions(bom.exclusions)}${lineTable(lines)}${exportsMarkup(endpoint(resultId))}` : ''}${report.generated ? '' : editor(bom)}`;
    if (view) bindBodyView(host.querySelector('.body-view'), view, onPart);
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

// The robot BOM: the BOM of the body record(s) publish_body.py generated (the robot is the sum of its parts).
export async function mountRobotBom(host, results, current = () => true) {
  let active = true;
  const valid = () => active && current();
  host.innerHTML = '<p role="status">Loading robot BOM…</p>';
  try {
    const report = await request('/api/robot-bom');
    if (!valid()) return {dispose() {active = false;}};
    const bodies = report.assemblies.map(a => `<li><a href="/design/?result=${encodeURIComponent(a.id)}">${esc(designStudyName(a))}</a> · version ${esc(a.study_version)} · ${a.bom ? `${a.bom.items.length} lines · ${esc(a.bom.scope || '')}` : 'no BOM in the record'}</li>`).join('');
    host.innerHTML = `<div class="meta"><span>Body BOM</span><span>Generated from the body definitions</span></div><h2>Robot BOM & costs</h2><p>The robot is the sum of its parts: this is the BOM of the body record, built by <code>robot/body/bom/build.py</code> from the body definitions (<code>robot/**/&lt;builder&gt;.bom.yaml</code>) and the body manifest. Change a part, quantity or supplier in its definition and republish the body (<code>robot/body/bom/publish_body.py --publish</code>).</p>${bodies ? `<ul>${bodies}</ul>` : '<p class="bom-notice">No body record is published yet.</p>'}${summary(report.totals)}${report.missing_bom_versions.length ? `<p class="bom-notice">${report.missing_bom_versions.length} body records carry no BOM.</p>` : ''}${exportsMarkup('/api/robot-bom?export=1')}${lineTable(report.lines, true)}`;
  } catch (error) { if (valid()) host.textContent = `Robot BOM could not load: ${error.message}`; }
  return {dispose() {active = false;}};
}
