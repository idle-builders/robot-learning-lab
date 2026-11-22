import {esc} from './readable.js?v=studies-final-20261020';
import {assetUrl as asset} from './asset-url.js';
const names = row => row.part_names || row.mesh_part_names || [];
const categories = {purchased:'Purchased', produced:'Produced', modified_purchased:'Purchased + modified', context:'Reference / context', unresolved:'To specify'};
const category = row => categories[row.category] ? row.category : 'unresolved';
const massNumber = new Intl.NumberFormat(undefined, {maximumSignificantDigits: 4});
const plain = value => value == null ? '' : Array.isArray(value) ? value.map(plain).join('; ') : typeof value === 'object' ? JSON.stringify(value) : String(value);
const safeUrl = value => {try {const url = new URL(value); return /^https?:$/.test(url.protocol) ? url.href : null;} catch {return null;}};
const num = value => Number.isFinite(value) ? massNumber.format(value) : value ?? '';
const vec = value => Array.isArray(value) ? `[${value.map(v => Number.isFinite(v) ? Math.round(v * 1000) / 1000 : v).join(', ')}]` : value || '-';

// ---------------------------------------------------------------- generated body view (robot/body/bom/publish_body.py)
// One payload (`body_view` section of the record) feeds the study's part view, the CAD BOM table and the part card shown
// when a CAD part is clicked: what the definitions declare plus what the files and checks measured.
const makeBuy = {make: 'produced', buy: 'purchased', included: 'context', context: 'context'};
export function viewRows(view) {
  return Object.entries(view.lines).map(([id, line]) => {
    const parts = Object.entries(view.parts).filter(([, p]) => p.line === id).map(([n]) => n);
    const first = view.parts[parts[0]];
    return {id, name: line.name, category: makeBuy[line.make_buy] || 'unresolved', quantity: line.quantity, part_names: parts.length ? parts : line.in_parts || [],
      manufacturer: line.supplier, part_number: line.part_number, material: line.material, unit_mass_g: line.unit_mass_g,
      process: first?.print ? `${first.print.printer || '-'} · ${first.print.material || '-'} · nozzle ${first.print.nozzle_mm ?? '-'} mm · layer ${first.print.layer_height_mm ?? '-'} mm · up ${vec(first.print.up)}` : null,
      notes: [line.notes, `definition: ${line.definitions.join(', ')}`].filter(Boolean)};
  });
}
function stepIndex(view) {
  if (view._steps) return view._steps;
  const map = new Map();
  for (const sub of view.subassemblies) for (const step of sub.steps) map.set(step.id, {...step, subassembly: sub.name, sub_number: sub.number});
  Object.defineProperty(view, '_steps', {value: map, enumerable: false});
  return map;
}
const generatorOf = (view, id) => view.generators.find(g => g.id === id);
const fileLine = (label, f) => f?.path ? `<li><strong>${label}</strong> <code>${esc(f.path)}</code> · ${esc(f.status)}${f.sha256 ? ` · sha256 ${esc(f.sha256)}…` : ''}${f.cold_record ? ` · bucket record ${esc(plain(f.cold_record))}` : ''}</li>` : `<li><strong>${label}</strong> none</li>`;

export function partCard(view, name) {
  const p = view?.parts?.[name];
  if (!p) return `<div class="body-part"><strong>${esc(name)}</strong><p class="muted">Not in this study's part view (context geometry or outside its scope).</p></div>`;
  const line = view.lines[p.line] || {}, steps = stepIndex(view), g = generatorOf(view, p.generator);
  const asm = (p.assembly || []).map(a => {
    const s = steps.get(a.step);
    return `<li>${esc(a.subassembly)} · ${a.step === 'base' ? 'present at the start (base)' : `step ${esc(a.step)} (#${a.number}, as ${esc(a.as)})`}${s ? ` · direction ${esc(vec(s.direction))} · ${esc(s.status)} · check: ${esc(s.check)}${s.measured ? ` · ${esc(s.measured)}` : ''}${s.tools?.length ? ` · tool ${esc(s.tools.join(', '))}` : ''} · <code>${esc(s.definition)}</code>` : ''}</li>`;
  }).join('') || '<li>No assembly step names this part.</li>';
  const pr = p.print;
  const print = pr ? `<ul>${['printer', 'material', 'nozzle_mm', 'layer_height_mm', 'line_width_mm', 'up'].map(k => `<li>${esc(k.replace(/_/g, ' '))}: ${esc(vec(pr[k]))}${pr.source?.[k] ? ` <small>(${esc(pr.source[k])})</small>` : ''}</li>`).join('')}</ul>` : '';
  const checks = (p.checks || []).map(c => `<tr><td>${esc(c.check)}</td><td>${c.worst.length ? c.worst.map(w => `${esc(num(w.value))} ${esc(w.unit)} · ${esc(w.what || '')}${w.mating_part ? ` · with ${esc(w.mating_part)}` : ''}${w.pose ? ` · ${esc(w.pose)}` : ''}`).join('<br>') : 'measured, no worst case listed'}${c.worst_cases > c.worst.length ? `<br><small>${c.worst_cases - c.worst.length} more in the report</small>` : ''}</td><td><small>${esc(c.label)} · ${esc(c.at)}</small></td></tr>`).join('');
  const findings = (p.findings || []).map(f => `<li><strong>${esc(f.kind)}</strong> ${esc(f.what)} <small>fix in <code>${esc(f.file || '(no definition claims it)')}</code>${f.entry ? ` ${esc(f.entry)}` : ''}</small></li>`).join('');
  return `<div class="body-part"><strong>${esc(name)}</strong> <small>${esc([p.role, p.owning_link, p.section].filter(Boolean).join(' · '))}</small>
    <p><strong>Definition to fix:</strong> <code>${esc(p.definition)}</code>${p.entry ? ` · ${esc(p.entry)}` : ''}</p>
    <h4>BOM line</h4><p>${esc(line.name || p.line)} <small>(${esc(p.line)})</small> · ${esc(line.make_buy)} · qty ${esc(line.quantity)}${line.quantity_declared && line.quantity_declared !== line.quantity ? ` (declared ${esc(line.quantity_declared)})` : ''}${line.spec ? ` · spec ${esc(plain(line.spec))}` : ''} · supplier ${esc(line.supplier || '-')}${line.part_number ? ` · ${esc(line.part_number)}` : ''}${line.tool ? ` · ${esc(line.tool)}` : ''}${p.mass_g != null ? ` · ${esc(num(p.mass_g))} g` : ''}${p.volume_mm3 != null ? ` · ${esc(num(p.volume_mm3))} mm3` : ''}</p>
    <h4>Files</h4><ul>${fileLine('STEP', p.step)}${fileLine('Mesh', p.mesh)}</ul>${p.flags?.length ? `<p><small>File flags: ${esc(p.flags.join(', '))}</small></p>` : ''}
    <h4>Generator</h4>${g ? `<p>${esc(g.id)} · <code>${esc(g.file)}</code>${g.rebuild_position ? ` · rebuild step ${g.rebuild_position}` : ' · section builder (linked read-only by the rebuild)'}</p><p><small>Regenerate:</small> <code>${esc(g.regenerate)}</code></p>` : '<p>No generator declared for this part.</p>'}
    <h4>Assembly</h4><ul>${asm}</ul>
    ${pr ? `<h4>Print setup</h4>${print}` : ''}
    <h4>Checks (newest report per check on this manifest)</h4>${checks ? `<div class="matrix"><table><thead><tr><th>Check</th><th>Worst cases</th><th>Report</th></tr></thead><tbody>${checks}</tbody></table></div>` : '<p class="muted">No complete check report on this manifest measured this part.</p>'}
    <h4>BOM findings</h4>${findings ? `<ul>${findings}</ul>` : '<p class="muted">None for this part.</p>'}</div>`;
}

// The subassemblies in assembly order with their steps; each part name opens its card.
export function bodyViewMarkup(view) {
  const c = view.counts, a = c.assembly;
  const partLink = n => `<button type="button" class="link-button" data-body-part="${esc(n)}">${esc(n)}</button>`;
  const subs = view.subassemblies.map(s => `<details class="body-sub"><summary>${s.number}. ${esc(s.name)} · ${s.steps.length} steps · <small>${esc(s.definition)}</small></summary><div class="inside">
    ${s.doc ? `<p>${esc(s.doc)}</p>` : ''}<p><small>After: ${esc(s.after.join(', ') || 'nothing (first)')} · present at the start: ${s.base.length ? s.base.map(partLink).join(', ') : 'what the earlier subassemblies built'}${s.tools?.length ? ` · tools: ${esc(s.tools.map(t => `${t.tool} (${t.screws})`).join('; '))}` : ''}</small></p>
    <div class="matrix bom-table"><table><thead><tr><th>Step</th><th>Parts / screws</th><th>With</th><th>Direction</th><th>Status</th><th>Assembly check</th><th>Definition</th></tr></thead><tbody>${s.steps.map(st => `<tr><td>#${st.number} ${esc(st.id)}<small>${esc(st.kind)}</small></td><td>${st.parts.map(partLink).join('<br>')}</td><td>${st.with_parts.map(partLink).join('<br>')}</td><td>${esc(vec(st.direction))}</td><td>${esc(st.status)}</td><td>${esc(st.check)}${st.measured ? `<small>${esc(st.measured)}</small>` : ''}</td><td><small>${esc(st.definition)}</small></td></tr>`).join('')}</tbody></table></div></div></details>`).join('');
  const general = view.findings_general.length ? `<details><summary>Findings not tied to one part · ${view.findings_general.length}</summary><ul>${view.findings_general.map(f => `<li><strong>${esc(f.kind)}</strong> ${esc(f.what)} <small><code>${esc(f.file || '')}</code> ${esc(f.entry || '')}</small></li>`).join('')}</ul></details>` : '';
  const gens = `<details><summary>Generators · ${view.generators.length} (rebuild order)</summary><ol>${view.generators.map(g => `<li>${esc(g.id)} · <code>${esc(g.file)}</code>${g.manifest_out ? ` → <code>${esc(g.manifest_out)}</code>` : ''}<br><small><code>${esc(g.regenerate)}</code></small></li>`).join('')}</ol></details>`;
  const checks = `<details><summary>Check reports used · ${view.checks.length} check runs</summary><ul>${view.checks.map(x => `<li>${esc(x.check)} · ${esc(x.label)} · ${esc(x.at)} · ${x.parts} parts</li>`).join('')}</ul></details>`;
  const defects = view.defects.length ? `<details><summary>Declared defects · ${view.defects.length}</summary><ul>${view.defects.map(d => `<li><strong>${esc(d.id)}</strong> (${d.parts.length} parts) ${esc(d.why || '')} · fix: ${esc(d.fix || '-')} ${d.task ? `· ${esc(d.task)}` : ''} <small><code>${esc(d.file)}</code></small></li>`).join('')}</ul></details>` : '';
  return `<section class="body-view" id="body-view"><h3>Selected body: subassemblies and parts</h3>
    <p>Generated by <code>robot/body/bom/publish_body.py</code> from the body definitions, <code>${esc(view.manifest)}</code> (sha256 ${esc(view.manifest_sha256.slice(0, 12))}) and the check reports on it. Do not edit here: each row names the definition file to fix.</p>
    <p>${Object.keys(view.parts).length} parts listed of ${view.scope.parts_in_manifest} in the manifest · ${c.lines} BOM lines · ${view.subassemblies.length} subassemblies · ${a.steps} steps (${a.check_supported} clear in the assembly check, ${a.not_supported} with contact or obstruction, ${a.not_measured} not measured; ${a.designer} designer, ${a.seeded} seeded) · ${c.findings} BOM findings · display GLB ${esc(view.glb.state)}</p>
    <label>Find part<input type="search" class="body-part-search" placeholder="Part name…" list="body-part-names"></label><datalist id="body-part-names">${Object.keys(view.parts).map(n => `<option value="${esc(n)}">`).join('')}</datalist>
    <div class="body-part-host"></div>${subs}${view.parts_in_no_step.length ? `<p><strong>Parts in no step:</strong> ${view.parts_in_no_step.map(partLink).join(', ')}</p>` : ''}${general}${defects}${gens}${checks}</section>`;
}
export function bindBodyView(host, view, onPart = () => {}) {
  const target = host.querySelector('.body-part-host'), search = host.querySelector('.body-part-search');
  const show = name => {target.innerHTML = partCard(view, name); target.scrollIntoView({block: 'nearest'}); onPart(name);};
  host.addEventListener('click', event => {const b = event.target.closest('[data-body-part]'); if (b) show(b.dataset.bodyPart);});
  search?.addEventListener('change', () => {if (view.parts[search.value]) show(search.value);});
  return show;
}

// ---------------------------------------------------------------- CAD panel BOM table
export function bomMarkup(cad) {
  if (!cad?.bom_path && !cad?.body_view) return '';
  return `<details class="cad-bom"><summary>Bill of materials · purchased & produced</summary><div class="inside"><p class="bom-status" role="status">Open to load this version's parts list.</p><div class="bom-actions"><label>Parts category<select class="bom-filter" aria-label="BOM category"><option value="all">All parts</option>${Object.entries(categories).map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label><label>Find part<input type="search" class="bom-search" aria-label="Find BOM part" placeholder="Part, supplier, material…"></label><button type="button" class="bom-show-all">Show assembly</button></div>${cad.body_view ? '<div class="bom-part" aria-live="polite"><p class="muted">Click a CAD part, or a part name in the table, for its BOM line, files, generator, assembly step and checks.</p></div>' : ''}<div class="bom-table matrix"></div><div class="cad-downloads">${cad.bom_path ? `<a href="${asset(cad.bom_path)}" download>Download BOM JSON</a>` : ''}${cad.bom_csv_path ? `<a href="${asset(cad.bom_csv_path)}" download>Download BOM CSV</a>` : ''}</div></div></details>`;
}

// The manifest maps procurement items to exact CAD identities: a motor's body,
// output and reference solids can represent one purchased unit.
export function bindBom(card, cad) {
  const panel = card.querySelector('.cad-bom');
  if (!panel) return () => {};
  let active = true, rows = null, loading = false, selectedName = null;
  const view = cad.body_view;
  const status = panel.querySelector('.bom-status'), filter = panel.querySelector('.bom-filter'), search = panel.querySelector('.bom-search');
  const isolate = partNames => card.dispatchEvent(new CustomEvent('cad-isolate-parts', {bubbles:true, detail:{names:partNames}}));
  const selected = () => panel.querySelectorAll('[data-bom-row]').forEach(tr => tr.classList.toggle('bom-selected', names(rows[Number(tr.dataset.bomRow)]).includes(selectedName)));
  const card_ = name => {const host = panel.querySelector('.bom-part'); if (host && name) host.innerHTML = partCard(view, name);};
  const partName = n => view ? `<button type="button" class="link-button" data-bom-part="${esc(n)}">${esc(n)}</button>` : esc(n);
  const show = () => {
    const query = search.value.trim().toLowerCase();
    const visible = rows.map((row,index)=>({row,index})).filter(({row}) => (filter.value === 'all' || category(row) === filter.value) && JSON.stringify(row).toLowerCase().includes(query));
    status.textContent = `${visible.length} of ${rows.length} BOM lines. ${panel.dataset.scope || 'Quantities describe one modeled assembly.'} CAD solids and procurement quantities are different counts.`;
    panel.querySelector('.bom-table').innerHTML = `<table><thead><tr><th>Part / CAD identity</th><th>Qty</th><th>Make / buy</th><th>Specification / process</th><th>Mass</th><th>Source / open issue</th><th>CAD</th></tr></thead><tbody>${visible.map(({row,index}) => {
      const url = safeUrl(row.source_url), refs = (row.source_refs || []).map(ref => ref.path ? `<a href="${asset(ref.path)}" download>${esc(ref.label || 'Source file')}</a>` : safeUrl(ref.url) ? `<a href="${esc(safeUrl(ref.url))}" target="_blank" rel="noopener">${esc(ref.label || 'Source')}</a>` : '').join(' · ');
      return `<tr data-bom-row="${index}"><td>${esc(row.name || row.id)}${names(row).length ? `<details><summary>CAD parts · ${names(row).length}</summary>${names(row).map(partName).join('<br>')}</details>` : ''}</td><td>${esc(row.quantity ?? 'TBD')}</td><td>${esc(categories[category(row)])}</td><td>${esc([row.manufacturer || row.supplier, row.part_number, plain(row.material)].filter(Boolean).join(' · ') || 'Not specified')}${row.process ? `<details><summary>Process / sourcing</summary>${esc(plain(row.process))}</details>` : ''}</td><td>${Number.isFinite(row.unit_mass_g) ? `${esc(massNumber.format(row.unit_mass_g))} g each` : 'Unknown'}${row.mass_basis ? `<br>${esc(row.mass_basis)}` : ''}</td><td><details><summary>Sources & open details</summary>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">Supplier / source</a>` : ''}${refs}${row.notes?.length ? `<p>${esc(plain(row.notes))}</p>` : ''}</details></td><td>${names(row).length ? `<button type="button" data-bom-isolate="${index}">Isolate</button>` : 'Unmodeled'}</td></tr>`;
    }).join('')}</tbody></table>`;
    panel.querySelectorAll('[data-bom-isolate]').forEach(button => button.onclick = () => isolate(names(rows[Number(button.dataset.bomIsolate)])));
    panel.querySelectorAll('[data-bom-part]').forEach(button => button.onclick = () => {
      selectedName = button.dataset.bomPart; card_(selectedName); selected();
      card.dispatchEvent(new CustomEvent('cad-select-part', {bubbles:true, detail:{name:selectedName}}));
    });
    selected();
  };
  const load = async () => {
    if (!panel.open || rows || loading) return;
    if (view) {panel.dataset.scope = `Generated from ${view.manifest} and the body definitions.`; rows = viewRows(view); show(); return;}
    loading = true;
    try {
      const response = await fetch(asset(cad.bom_path));
      if (!response.ok) throw new Error('BOM source unavailable');
      const data = await response.json();
      if (!Array.isArray(data.items)) throw new Error('Expected BOM items array');
      if (!active) return;
      panel.dataset.scope = plain(data.scope || data.quantity_scope || data.summary);
      rows = data.items; show();
    } catch (error) {if (active) status.textContent = `Could not display BOM: ${error.message}. Download the source below.`;}
    finally {loading = false;}
  };
  const pick = event => {selectedName = event.detail.name; if (view && selectedName) {if (!panel.open) {panel.open = true;} card_(selectedName);} if (rows) selected();};
  panel.ontoggle = load;
  filter.onchange = search.oninput = () => {if (rows) show();};
  panel.querySelector('.bom-show-all').onclick = () => isolate(null);
  card.addEventListener('cad-part-selected', pick);
  return () => {active = false; card.removeEventListener('cad-part-selected', pick);};
}
