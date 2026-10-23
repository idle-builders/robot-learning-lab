import {esc} from './readable.js?v=studies-final-20261020';
const asset = path => `/asset?path=${encodeURIComponent(path)}`;
const names = row => row.part_names || row.mesh_part_names || [];
const categories = {purchased:'Purchased', produced:'Produced', modified_purchased:'Purchased + modified', context:'Reference / context', unresolved:'To specify'};
const category = row => categories[row.category] ? row.category : 'unresolved';
const massNumber = new Intl.NumberFormat(undefined, {maximumSignificantDigits: 4});
const plain = value => value == null ? '' : Array.isArray(value) ? value.map(plain).join('; ') : typeof value === 'object' ? JSON.stringify(value) : String(value);
const safeUrl = value => {try {const url = new URL(value); return /^https?:$/.test(url.protocol) ? url.href : null;} catch {return null;}};

export function bomMarkup(cad) {
  if (!cad?.bom_path) return '';
  return `<details class="cad-bom"><summary>Bill of materials · purchased & produced</summary><div class="inside"><p class="bom-status" role="status">Open to load this version's parts list.</p><div class="bom-actions"><label>Parts category<select class="bom-filter" aria-label="BOM category"><option value="all">All parts</option>${Object.entries(categories).map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label><label>Find part<input type="search" class="bom-search" aria-label="Find BOM part" placeholder="Part, supplier, material…"></label><button type="button" class="bom-show-all">Show assembly</button></div><div class="bom-table matrix"></div><div class="cad-downloads"><a href="${asset(cad.bom_path)}" download>Download BOM JSON</a>${cad.bom_csv_path ? `<a href="${asset(cad.bom_csv_path)}" download>Download BOM CSV</a>` : ''}</div></div></details>`;
}

// The manifest maps procurement items to exact CAD identities: a motor's body,
// output and reference solids can represent one purchased unit.
export function bindBom(card, cad) {
  const panel = card.querySelector('.cad-bom');
  if (!panel) return () => {};
  let active = true, rows = null, loading = false, selectedName = null;
  const status = panel.querySelector('.bom-status'), filter = panel.querySelector('.bom-filter'), search = panel.querySelector('.bom-search');
  const isolate = partNames => card.dispatchEvent(new CustomEvent('cad-isolate-parts', {bubbles:true, detail:{names:partNames}}));
  const selected = () => panel.querySelectorAll('[data-bom-row]').forEach(tr => tr.classList.toggle('bom-selected', names(rows[Number(tr.dataset.bomRow)]).includes(selectedName)));
  const show = () => {
    const query = search.value.trim().toLowerCase();
    const visible = rows.map((row,index)=>({row,index})).filter(({row}) => (filter.value === 'all' || category(row) === filter.value) && JSON.stringify(row).toLowerCase().includes(query));
    status.textContent = `${visible.length} of ${rows.length} BOM lines. ${panel.dataset.scope || 'Quantities describe one modeled assembly.'} CAD solids and procurement quantities are different counts.`;
    panel.querySelector('.bom-table').innerHTML = `<table><thead><tr><th>Part / CAD identity</th><th>Qty</th><th>Make / buy</th><th>Specification / process</th><th>Mass</th><th>Source / open issue</th><th>CAD</th></tr></thead><tbody>${visible.map(({row,index}) => {
      const url = safeUrl(row.source_url), refs = (row.source_refs || []).map(ref => ref.path ? `<a href="${asset(ref.path)}" download>${esc(ref.label || 'Source file')}</a>` : safeUrl(ref.url) ? `<a href="${esc(safeUrl(ref.url))}" target="_blank" rel="noopener">${esc(ref.label || 'Source')}</a>` : '').join(' · ');
      return `<tr data-bom-row="${index}"><td>${esc(row.name || row.id)}${names(row).length ? `<details><summary>CAD parts · ${names(row).length}</summary>${names(row).map(esc).join('<br>')}</details>` : ''}</td><td>${esc(row.quantity ?? 'TBD')}</td><td>${esc(categories[category(row)])}</td><td>${esc([row.manufacturer || row.supplier, row.part_number, plain(row.material)].filter(Boolean).join(' · ') || 'Not specified')}${row.process ? `<details><summary>Process / sourcing</summary>${esc(plain(row.process))}</details>` : ''}</td><td>${Number.isFinite(row.unit_mass_g) ? `${esc(massNumber.format(row.unit_mass_g))} g each` : 'Unknown'}${row.mass_basis ? `<br>${esc(row.mass_basis)}` : ''}</td><td><details><summary>Sources & open details</summary>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">Supplier / source</a>` : ''}${refs}${row.notes?.length ? `<p>${esc(plain(row.notes))}</p>` : ''}</details></td><td>${names(row).length ? `<button type="button" data-bom-isolate="${index}">Isolate</button>` : 'Unmodeled'}</td></tr>`;
    }).join('')}</tbody></table>`;
    panel.querySelectorAll('[data-bom-isolate]').forEach(button => button.onclick = () => isolate(names(rows[Number(button.dataset.bomIsolate)])));
    selected();
  };
  const load = async () => {
    if (!panel.open || rows || loading) return;
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
  const pick = event => {selectedName = event.detail.name; if (rows) selected();};
  panel.ontoggle = load;
  filter.onchange = search.oninput = () => {if (rows) show();};
  panel.querySelector('.bom-show-all').onclick = () => isolate(null);
  card.addEventListener('cad-part-selected', pick);
  return () => {active = false; card.removeEventListener('cad-part-selected', pick);};
}
