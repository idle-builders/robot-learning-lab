// Explicit native groups; one assessment GET only after a person opens the section.
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const own = (object, key) => object && Object.hasOwn(object, key) ? object[key] : undefined;
const count = value => Number.isSafeInteger(value) && value >= 0;
const fraction = value => value === null || Number.isFinite(value) && value >= 0 && value <= 1;
const display = value => value === null || value === undefined ? 'Unavailable' : String(value);
const short = value => Number.isInteger(value) ? String(value) : Number(value.toPrecision(4)).toString();
import {assetUrl} from './asset-url.js';
const colors = ['#285e75','#976039','#68658a','#287c83','#4d5560','#9b6a80','#756128'];

function sectionContract(payload, assets) {
  const path = payload.assessment_path;
  if (typeof path !== 'string' || !path.endsWith('.json') || path.startsWith('/') || path.includes('\\') || path.split('/').some(p => !p || p === '.' || p === '..') || /^[a-z]+:/i.test(path)) throw Error('A confined assessment JSON asset path is required.');
  if (!assets.some(asset => asset.path === path && asset.kind === 'data')) throw Error('Assessment is not a registered data asset for this record.');
  if (!Array.isArray(payload.series) || !payload.series.length || payload.series.some(s => !s || ['label','group_key','coverage_key'].some(k => typeof s[k] !== 'string' || !s[k].trim()))) throw Error('Each series needs an explicit label, error group key and coverage group key.');
  return path;
}

function axisOrder(data, series) {
  if (Array.isArray(data.curve_axes)) return data.curve_axes;
  const axes = [];
  for (const entry of series) {
    const order = own(data.requested_horizon_coverage?.groups, entry.coverage_key)?.joint_order22;
    if (Array.isArray(order) && order.length === 22 && order.every(x => typeof x === 'string') && new Set(order).size === 22) for (const axis of order) if (!axes.includes(axis)) axes.push(axis);
    const accuracy = own(data.condition_evidence?.groups, entry.group_key)?.accuracy;
    for (const key of Object.keys(accuracy || {})) if (/\/(applied|reference)$/.test(key)) {
      const axis = key.slice(0, key.lastIndexOf('/'));
      if (axis && !axes.includes(axis)) axes.push(axis);
    }
  }
  return axes;
}

function accuracyCurve(raw, notes) {
  if (!raw) { notes.push('The selected axis/target error curve was not recorded.'); return null; }
  const bounds = raw.boundaries || raw.boundaries_rad;
  if (!count(raw.n) || !count(raw.finite) || raw.finite > raw.n || !Array.isArray(bounds) || !bounds.length || bounds.some((x,i) => !Number.isFinite(x) || x < 0 || i > 0 && x <= bounds[i-1])) {
    notes.push('Error counts or increasing native-unit boundaries are missing or malformed.'); return null;
  }
  const curves = {};
  for (const key of ['all_row_fraction','finite_fraction']) {
    const values = raw[key];
    if (!Array.isArray(values) || values.length !== bounds.length || !values.every(fraction)) notes.push(`${key} is missing or malformed; its line is unavailable.`);
    else curves[key] = values;
  }
  const within = Array.isArray(raw.within) && raw.within.length === bounds.length && raw.within.every(x => count(x) && x <= raw.finite) ? raw.within : null;
  if (!within) notes.push('Exact within-boundary numerator counts are unavailable.');
  return {...raw, bounds, curves, within};
}

function coverageCurves(raw, axis, notes) {
  if (!raw) { notes.push('The explicitly linked coverage group was not recorded.'); return null; }
  const order = raw.axis_order || raw.joint_order22;
  const axisIndex = Array.isArray(order) && new Set(order).size === order.length ? order.indexOf(axis) : -1;
  const horizon = raw.horizon_controls ?? 1700;
  if (!count(horizon) || horizon < 1) { notes.push('Requested horizon is missing or malformed.'); return null; }
  const fields = [
    ['potential_requested',`Potential ${horizon}-control episode request`],
    ['experiment_opportunity','Remaining experiment-budget opportunity'],
    ['observed','Observed rows · includes floor tails'],
    ['finite_tracking',raw.tracking_scope ? 'Finite command error · selected axis' : 'Finite applied-target tracking · selected axis'],
    ['before_source_ending',raw.tracking_scope ? 'Before first physical predicate · not upright' : 'Rows before source ending · not upright'],
    ['finite_before_source_ending',raw.tracking_scope ? 'Finite command error before physical predicate' : 'Finite applied tracking before ending'],
    ['at_or_after_source_ending',raw.tracking_scope ? 'At/after physical predicate · retained tails' : 'Rows at/after source ending · retained tails']
  ];
  const lines = fields.map(([key,label], index) => {
    const values = raw[key]; const matrix = key.startsWith('finite_');
    let valid = Array.isArray(values) && values.length === horizon;
    if (matrix) valid = valid && axisIndex >= 0 && values.every(row => Array.isArray(row) && row.length === order.length && row.every(count));
    else valid = valid && values.every(count);
    if (!valid) notes.push(`${key} is missing or malformed; those counts are unavailable.`);
    return {key, label, values: valid ? (matrix ? values.map(row => row[axisIndex]) : values) : null, color: colors[index], dash: index % 3 === 1 ? '7 4' : index % 3 === 2 ? '2 3' : ''};
  });
  return {raw, lines, horizon};
}

function plot(lines, {title, xLabel, yLabel, xMax, yMax, marker = null, step = false, xUnit = 'rad'}) {
  const width = 720, height = 260, left = 58, right = 18, top = 14, bottom = 44;
  const px = x => left + x / xMax * (width-left-right);
  const py = y => height-bottom - y / yMax * (height-top-bottom);
  const available = lines.filter(line => line.values?.some(value => value !== null));
  if (!available.length) return '<p class="cec-empty">No finite curve values were recorded for this selection.</p>';
  const grid = [0,.25,.5,.75,1].map(t => `<path class="cec-grid" d="M${left} ${py(t*yMax)}H${width-right}"/><text x="${left-8}" y="${py(t*yMax)+4}" text-anchor="end">${esc(short(t*yMax))}</text><text x="${px(t*xMax)}" y="${height-bottom+20}" text-anchor="middle">${esc(short(t*xMax))}</text>`).join('');
  const paths = available.map(line => {
    let d = '', active = false;
    line.values.forEach((value,i) => {
      if (value === null) { active = false; return; }
      const x = line.x ? line.x[i] : i/50;
      d += `${active ? (step ? 'V' : 'L') : 'M'}${active && step ? py(value) : `${px(x)},${py(value)}`}`;
      if (step) d += `H${px((i+1)/50)}`;
      active = true;
    });
    const dots = step ? '' : line.values.map((value,i) => value === null ? '' : `<circle cx="${px(line.x[i])}" cy="${py(value)}" r="2.7" fill="${line.color}"><title>${esc(line.x[i])} ${esc(xUnit)}; fraction ${esc(value)}</title></circle>`).join('');
    return `<path d="${d}" fill="none" stroke="${line.color}" stroke-width="2" stroke-dasharray="${line.dash || ''}"/>${dots}`;
  }).join('');
  const legend = available.map(line => `<li><i style="border-color:${line.color};border-top-style:${line.dash ? 'dashed' : 'solid'}"></i>${esc(line.label)}</li>`).join('');
  return `<div class="cec-plot"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${grid}${paths}${marker === null ? '' : `<path class="cec-marker" d="M${px(marker)} ${top}V${height-bottom}"/>`}<text x="${(width+left-right)/2}" y="${height-6}" text-anchor="middle">${esc(xLabel)}</text><text transform="translate(14 ${(height-bottom+top)/2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text></svg><ul class="cec-legend">${legend}</ul></div>`;
}

function accuracyView(curve) {
  if (!curve) return '<p class="cec-empty">Error accuracy is unavailable.</p>';
  const unit = curve.unit || 'rad';
  const lines = [
    {values: curve.curves.all_row_fraction, x: curve.bounds, label: `All selected rows · denominator n=${curve.n}`, color: colors[0]},
    {values: curve.curves.finite_fraction, x: curve.bounds, label: `Finite-error rows · denominator finite=${curve.finite}`, color: colors[1], dash:'7 4'}
  ];
  const rows = curve.bounds.map((bound,i) => `<tr><td>${esc(bound)}</td><td>${display(curve.within?.[i])}</td><td>${display(curve.curves.all_row_fraction?.[i])}</td><td>${display(curve.curves.finite_fraction?.[i])}</td></tr>`).join('');
  return `<h4>Error accuracy</h4><p>n=${curve.n} selected rows; finite=${curve.finite}; excluded nonfinite=${esc(display(curve.excluded_nonfinite))}. Both curves use the same within-boundary numerator. Fractions range from 0 to 1.</p>${plot(lines,{title:curve.title || 'Empirical absolute joint-error fractions',xLabel:`Saved absolute-error boundary (${unit})`,xUnit:unit,yLabel:'Fraction of rows',xMax:curve.bounds.at(-1) || 1,yMax:1})}<p class="cec-note">Only saved boundaries are evaluated; connecting lines guide the eye. No tolerance is selected or scored.</p><details><summary>Exact error numerators and fractions</summary><div class="matrix"><table><thead><tr><th>Boundary (${esc(unit)})</th><th>Within boundary (rows)</th><th>Within / n=${curve.n}</th><th>Within / finite=${curve.finite}</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

function coverageView(coverage, k, coverageKey) {
  if (!coverage) return '<p class="cec-empty">Requested-horizon coverage is unavailable.</p>';
  const known = coverage.lines.filter(line => line.values);
  const ymax = Math.max(1, ...known.flatMap(line => line.values));
  const potential = coverage.lines[0].values?.[k], opportunity = coverage.lines[1].values?.[k];
  const share = (value, denominator) => !count(value) || !count(denominator) || denominator === 0 ? 'Unavailable' : `${value} / ${denominator} = ${value/denominator}`;
  const rows = coverage.lines.map(line => `<tr><td>${esc(line.label)}</td><td>${display(line.values?.[k])}</td><td>${share(line.values?.[k],potential)}</td><td>${share(line.values?.[k],opportunity)}</td></tr>`).join('');
  return `<h4>Requested-horizon coverage</h4><p class="cec-scope">Coverage scope: whole retained episodes in <code>${esc(coverageKey)}</code>. Error-row phase and ending filters are not transferred to this group.</p><p>${esc(display(coverage.raw.episodes))} retained episodes; slots: ${esc(Array.isArray(coverage.raw.slots) ? coverage.raw.slots.join(', ') : 'Unavailable')}; invalid local-clock rows: ${esc(display(coverage.raw.invalid_local_clock_rows))}. Lines count episode-row opportunities at each reset-local interval. Potential horizons overlap; they are not extra compute.</p>${plot(coverage.lines,{title:'Native requested-horizon and retained-row coverage',xLabel:'Reset-local pre-control time k/50 (s)',yLabel:'Episode-row count',xMax:coverage.horizon/50,yMax:ymax,marker:k/50,step:true})}<label class="cec-interval">Inspect requested interval: <strong>k=${k} · ${(k/50).toFixed(2)}–${((k+1)/50).toFixed(2)} s</strong><input type="range" min="0" max="${coverage.horizon-1}" step="1" value="${k}" data-cec-interval></label><div class="matrix"><table><thead><tr><th>Native quantity</th><th>Rows at k=${k}</th><th>Fraction of potential request</th><th>Fraction of budget opportunity</th></tr></thead><tbody>${rows}</tbody></table></div><p class="cec-note">Zero or missing denominators remain unavailable. ${esc(coverage.raw.tracking_scope || 'Finite tracking here always means applied-target error, including when the error plot shows the saved reference. Before source ending is not an upright guarantee; observed and finite rows may include floor tails.')}</p>`;
}

// Counts are rearranged for the shared viewer; no new errors or tolerances are
// evaluated here. Every numerator comes from the retained native reduction.
function velocityAssessment(source) {
  const axes = ['forward', 'lateral', 'yaw'], groups = {}, coverage = {};
  const sum = values => {
    if (!Array.isArray(values) || !values.every(count)) throw Error('Malformed native row counts.');
    return values.reduce((a,b) => a+b, 0);
  };
  for (const [key, raw] of Object.entries(source.groups || {})) {
    const n = raw.horizon_controls;
    if (!count(n) || n < 1) throw Error('Native horizon unavailable.');
    for (const name of ['observed','before_first_physical','requested_episode_horizon','budget_opportunity']) {
      if (!Array.isArray(raw[name]) || raw[name].length !== n || !raw[name].every(count)) throw Error(`Malformed native ${name} coverage.`);
    }
    const accuracy = {};
    for (const axis of axes) {
      const e = raw.errors?.[axis];
      if (!e || !Array.isArray(e.boundaries) || !['m/s','rad/s'].includes(e.unit)) throw Error('Native command error unavailable.');
      for (const [target, numerator, denominator, finiteKey] of [
        ['command','within','observed','finite'],
        ['command_before_physical','before_within','before_first_physical','before_finite']
      ]) {
        const rows = sum(raw[denominator]), finite = sum(e[finiteKey]);
        if (e[finiteKey].length !== n || !Array.isArray(e[numerator]) || e[numerator].length !== n || e[numerator].some(row => !Array.isArray(row) || row.length !== e.boundaries.length || !row.every(count))) throw Error('Malformed native command numerators.');
        const within = e.boundaries.map((_,i) => sum(e[numerator].map(row => row[i])));
        accuracy[`${axis}/${target}`] = {n:rows,finite,excluded_nonfinite:rows-finite,boundaries:e.boundaries,unit:e.unit,title:'Empirical absolute command-error fractions',within,
          all_row_fraction:within.map(x => rows ? x/rows : null),finite_fraction:within.map(x => finite ? x/finite : null)};
      }
    }
    groups[key] = {controls:sum(raw.observed),episodes_with_rows:raw.episodes,accuracy};
    coverage[key] = {horizon_controls:n,axis_order:axes,episodes:raw.episodes,
      potential_requested:raw.requested_episode_horizon,experiment_opportunity:raw.budget_opportunity,observed:raw.observed,
      before_source_ending:raw.before_first_physical,at_or_after_source_ending:raw.observed.map((x,i) => x-raw.before_first_physical[i]),
      finite_tracking:Array.from({length:n},(_,i) => axes.map(axis => raw.errors[axis].finite[i])),
      finite_before_source_ending:Array.from({length:n},(_,i) => axes.map(axis => raw.errors[axis].before_finite[i])),
      tracking_scope:'Finite tracking here means issued velocity-command error. Counts are indexed by pre-action time; measurements are reached states20ms later. Before the first physical predicate is not an upright guarantee.'};
  }
  return {schema:source.schema,curve_axes:axes,curve_targets:[['command','Issued command · all observed rows'],['command_before_physical','Issued command · before physical predicate']],
    condition_evidence:{groups,meaning:'Errors compare reached body-frame forward/lateral velocity and body-z angular velocity with the issued command. '+source.meaning},requested_horizon_coverage:{groups:coverage,meaning:source.meaning}};
}

export function mountCommandErrorCurves(host, payload, assets, isCurrent = () => true) {
  let disposed = false, started = false, data = null, selectedSeries = 0, selectedAxis = '', target = 'applied', interval = 0;
  const controller = new AbortController();
  const alive = () => !disposed && host.isConnected && isCurrent();
  if (!document.querySelector('link[data-command-error-curves]')) {
    const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = '/command-error-curves.css?v=native-curves-20261029'; link.dataset.commandErrorCurves = ''; document.head.append(link);
  }
  host.classList.add('command-error-curves');
  function redraw() {
    if (!alive()) return;
    const series = payload.series[selectedSeries], axes = axisOrder(data, payload.series), notes = [];
    if (!selectedAxis || !axes.includes(selectedAxis)) selectedAxis = axes[0] || '';
    const targets = data.curve_targets || [['applied','Applied target'],['reference','Saved reference']];
    if (!targets.some(([value]) => value === target)) target = targets[0][0];
    const group = own(data.condition_evidence?.groups, series.group_key);
    const curve = accuracyCurve(own(group?.accuracy, `${selectedAxis}/${target}`), notes);
    const coverage = coverageCurves(own(data.requested_horizon_coverage?.groups, series.coverage_key), selectedAxis, notes);
    if (coverage) interval = Math.min(interval, coverage.horizon-1);
    if (!data.curve_axes && axes.length !== 22) notes.push(`Source exposes ${axes.length} distinct named axes; a complete 22-axis order is unavailable.`);
    const options = (values,selected) => values.map(([value,label]) => `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`).join('');
    host.innerHTML = `${data.synthetic_fixture ? '<p class="cec-empty">Synthetic source fixture · no robot evidence.</p>' : ''}<p>Groups overlap and must not be added. Error rows and linked coverage can describe different phases or ending strata; their exact source keys are shown below.</p><div class="cec-controls"><label>Source series<select data-cec-series>${options(payload.series.map((s,i)=>[String(i),s.label]),String(selectedSeries))}</select></label><label>${data.curve_axes ? 'Command axis' : 'Physical axis'} (${axes.length})<select data-cec-axis${axes.length ? '' : ' disabled'}>${options(axes.map(axis=>[axis,axis]),selectedAxis)}</select></label><label>Saved target<select data-cec-target>${options(targets,target)}</select></label></div>${notes.length ? `<ul class="cec-empty" role="status">${notes.map(note=>`<li>${esc(note)}</li>`).join('')}</ul>` : ''}${accuracyView(curve)}${coverageView(coverage,interval,series.coverage_key)}<details><summary>Source groups and denominator scope</summary><dl><dt>Error group</dt><dd>${esc(series.group_key)}</dd><dt>Coverage group</dt><dd>${esc(series.coverage_key)}</dd><dt>Source group rows / episodes with rows</dt><dd>${esc(display(group?.controls))} / ${esc(display(group?.episodes_with_rows))}</dd></dl><p>The coverage key is explicitly supplied by this record. Its whole-episode coverage may be broader than a prefix/PPO, phase or before-ending error subset. Slot counts are not independent starts. Resets start new episodes and do not establish recovery.</p><p>${esc(data.condition_evidence?.meaning || 'Error-group interpretation not recorded.')}</p><p>${esc(data.requested_horizon_coverage?.meaning || 'Coverage interpretation not recorded.')}</p><p><a href="${assetUrl(payload.assessment_path)}" download>Save exact assessment JSON</a></p></details>`;
    host.querySelector('[data-cec-series]').onchange = event => { selectedSeries = Number(event.target.value); redraw(); host.querySelector('[data-cec-series]').focus(); };
    host.querySelector('[data-cec-axis]').onchange = event => { selectedAxis = event.target.value; redraw(); host.querySelector('[data-cec-axis]').focus(); };
    host.querySelector('[data-cec-target]').onchange = event => { target = event.target.value; redraw(); host.querySelector('[data-cec-target]').focus(); };
    const slider = host.querySelector('[data-cec-interval]');
    if (slider) slider.onchange = event => { interval = Number(event.target.value); redraw(); host.querySelector('[data-cec-interval]')?.focus(); };
  }
  return {
    async load() {
      if (started || !alive()) return;
      started = true;
      try {
        const path = sectionContract(payload, assets);
        host.innerHTML = '<p role="status">Loading native curve source…</p>';
        const response = await fetch(assetUrl(path), {signal:controller.signal});
        if (!alive()) { await response.body?.cancel(); return; }
        if (!response.ok) throw Error(`Assessment source unavailable (HTTP ${response.status}).`);
        data = await response.json();
        if (data?.schema === 'cove50.forward-stop-curves.v1') data = velocityAssessment(data);
        else if (!data || data.schema !== 'cove50.body-std-pilot.saved-assessment.v1') throw Error('Unsupported or missing native assessment schema.');
        redraw();
      } catch (error) {
        if (alive()) host.textContent = `Curves unavailable: ${error.message}`;
      }
    },
    dispose() { disposed = true; controller.abort(); data = null; }
  };
}
