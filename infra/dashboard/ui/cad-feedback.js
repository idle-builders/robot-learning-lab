// Annotated CAD snapshot for the owner's design feedback to Claude / Codex / ChatGPT.
// A mark (pen, arrow, box) is numbered on the image and resolved to the parts under it
// with the frozen snapshot camera, so the feedback names parts and 3D points, not just pixels.
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const COLORS = {red: '#e0312b', yellow: '#f2b705', blue: '#1f6fe0'};
const round = (v, d = 1) => Number.isFinite(v) ? Number(v.toFixed(d)) : v;
const mm = (vector, scale) => vector.map(v => round(v*scale));

// Orientation in the manifest frame: +X is the R_ joint side, -Y the face side, +Z up.
export function viewPhrase(direction) {
  const [x, y, z] = direction, sides = [];
  if (y < -.38) sides.push('front (−Y, face side)'); else if (y > .38) sides.push('back (+Y)');
  if (x > .38) sides.push('+X side (R_ joints)'); else if (x < -.38) sides.push('−X side (L_ joints)');
  const elevation = Math.asin(Math.max(-1, Math.min(1, z)))*180/Math.PI;
  const height = elevation > 60 ? 'from above (top view)' : elevation < -60 ? 'from below' :
    elevation > 12 ? `${Math.round(elevation)}° above` : elevation < -12 ? `${Math.round(-elevation)}° below` : 'at part height';
  return sides.length ? `from ${sides.join(' and ')}, ${height}` : elevation >= 0 ? 'from straight above (top view)' : 'from straight below';
}

function partsAt(samples, pick, scale) {
  const counts = new Map(), points = [];
  for (const [x, y] of samples) {
    const hit = pick(x, y); if (!hit) continue;
    const entry = counts.get(hit.name) || {...hit, samples: 0, points: []};
    entry.samples++; entry.points.push(hit.point); counts.set(hit.name, entry); points.push(hit.point);
  }
  const parts = [...counts.values()].sort((a, b) => b.samples - a.samples).slice(0, 6).map(({points: p, point, ...part}) =>
    ({...part, point_mm: mm([0, 1, 2].map(i => p.reduce((s, q) => s + q[i], 0)/p.length), scale)}));
  return {parts, samples: samples.length, hits: points.length};
}

function samplesFor(mark) {
  const pts = mark.points;
  if (mark.tool === 'box') {
    const [[x0, y0], [x1, y1]] = [pts[0], pts.at(-1)], out = [];
    for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) out.push([x0 + (x1-x0)*i/6, y0 + (y1-y0)*j/6]);
    return out;
  }
  if (mark.tool === 'arrow') {
    const [tx, ty] = pts.at(-1), r = .008;
    return [[tx, ty], [tx+r, ty], [tx-r, ty], [tx, ty+r], [tx, ty-r]];
  }
  const step = Math.max(1, Math.floor(pts.length/60)), line = pts.filter((_, i) => i % step === 0);
  // A loop drawn around something means its inside: sample the enclosed area, not only the line.
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const [a, b] = [pts[0], pts.at(-1)];
  if (pts.length < 8 || Math.hypot(b[0]-a[0], b[1]-a[1]) > .3*Math.hypot(x1-x0, y1-y0)) return line;
  const inside = ([x, y]) => pts.reduce((odd, [xi, yi], i) => { const [xj, yj] = pts[(i || pts.length) - 1];
    return (yi > y) !== (yj > y) && x < (xj-xi)*(y-yi)/(yj-yi) + xi ? !odd : odd; }, false);
  const area = [];
  for (let i = 1; i < 8; i++) for (let j = 1; j < 8; j++) area.push([x0 + (x1-x0)*i/8, y0 + (y1-y0)*j/8]);
  return area.filter(inside);
}

export function feedbackMarkdown(c) {
  const lines = [`# Design feedback · ${c.study?.title || c.study?.result_id || 'CAD view'}`, ''];
  if (c.note) lines.push('## Owner note', '', c.note, '');
  if (c.marks?.length) {
    lines.push('## Marks on the snapshot', '');
    for (const m of c.marks) {
      const parts = m.parts.length ? m.parts.map(p => `${p.name}${p.role ? ` (${p.role}${p.owning_link ? ', link ' + p.owning_link : ''})` : ''} at [${p.point_mm.join(', ')}] mm`).join('; ') : 'no part under the mark (background)';
      lines.push(`${m.n}. ${m.color} ${m.tool}${m.label ? ` “${m.label}”` : ''}: ${parts}`);
    }
    lines.push('');
  }
  const v = c.view, joints = Object.entries(c.joints_deg || {}).filter(([, a]) => Math.abs(a) > .05);
  lines.push('## What was on screen', '',
    `- Study: ${c.study?.result_id || '?'}${c.study?.section ? ` · section ${c.study.section}` : ''}${c.dashboard_url ? ` · ${c.dashboard_url}` : ''}`,
    `- CAD configuration: ${c.cad?.view_label || '?'} (${c.cad?.model_path || '?'})`,
    `- Camera: ${v.phrase}; looking at [${v.target_mm.join(', ')}] mm from ${round(v.distance_mm, 0)} mm, fov ${round(v.fov_deg, 0)}°`,
    `- Joint angles: ${joints.length ? joints.map(([k, a]) => `${k} ${round(a)}°`).join(', ') + ' (others 0°)' : 'all 0° (neutral)'}`,
    `- Display: body panels ${c.display.shells ? 'shown' : 'hidden'}, explode ${c.display.explode_percent}%, cut ${c.display.cut.enabled ? `${c.display.cut.axis} at ${c.display.cut.position}%${c.display.cut.reverse ? ' reversed' : ''}` : 'off'}${c.display.isolated_parts ? `, isolated ${c.display.isolated_parts.join(', ')}` : ''}`,
    `- Selected part: ${c.selected_part?.name || 'none'} · part at the view centre: ${c.center_part?.name || 'none'}`);
  if (c.display.explode_percent) lines.push('- Points are in the exploded display, not the assembled positions.');
  lines.push('', 'Frame: manifest world mm, +X = R_ joint side, −Y = face side, +Z up.');
  return lines.join('\n');
}

export function openFeedback({snapshot, context, mmPerUnit = 1000}) {
  // The remote view is read-only: the clipboard carries the feedback there, not a save on the lab computer.
  const remote = document.documentElement.classList.contains('web-snapshot');
  const marks = []; let tool = 'pen', color = 'red', drawing = null;
  const dialog = document.createElement('div');
  dialog.className = 'cad-feedback'; dialog.setAttribute('role', 'dialog'); dialog.setAttribute('aria-label', 'Annotate CAD snapshot');
  dialog.innerHTML = `<div class="cad-feedback-stage"><canvas></canvas></div><aside class="cad-feedback-side">
    <h3>Snapshot feedback</h3><p class="muted">Mark each area, then write its comment below. The image gets a numbered legend with each comment, the parts under the mark and the view.</p>
    <div class="cad-feedback-tools" role="group" aria-label="Mark tool">${['pen', 'arrow', 'box'].map(t => `<button type="button" data-tool="${t}" aria-pressed="${t === tool}">${t[0].toUpperCase() + t.slice(1)}</button>`).join('')}</div>
    <div class="cad-feedback-tools" role="group" aria-label="Mark colour">${Object.entries(COLORS).map(([name, hex]) => `<button type="button" data-color="${name}" aria-pressed="${name === color}" aria-label="${name}" style="--swatch:${hex}"><i></i></button>`).join('')}<button type="button" class="feedback-undo">Undo</button></div>
    <ol class="cad-feedback-marks"></ol>
    <p class="cad-feedback-status" role="status"></p>
    <div class="cad-feedback-actions">${remote ? '' : '<button type="button" class="feedback-save primary">Save for Claude / Codex</button>'}<button type="button" class="feedback-copy-image${remote ? ' primary' : ''}">Copy image</button><button type="button" class="feedback-download">Download PNG</button><button type="button" class="feedback-close">Close</button></div>
    <p class="muted">${remote ? 'Copy image and paste it into the chat; write the question there.' : 'Save puts it where Claude Code and Codex read it; Copy image is one picture for a chat.'}</p></aside>`;
  document.body.append(dialog);
  const canvas = dialog.querySelector('canvas'), ctx = canvas.getContext('2d'), list = dialog.querySelector('.cad-feedback-marks');
  const status = dialog.querySelector('.cad-feedback-status');
  canvas.width = snapshot.width; canvas.height = snapshot.height;
  const image = new Image(); image.onload = render; image.src = snapshot.image;
  const lw = Math.max(3, snapshot.width/320), fs = Math.max(13, Math.round(snapshot.width/70));
  const partNames = m => m.parts.length ? m.parts.slice(0, 3).map(p => p.name).join(', ') : 'no part under the mark';
  function wrap(g, text, width) {
    const lines = [];
    for (const paragraph of String(text).split('\n')) {
      let line = '';
      for (const word of paragraph.split(/\s+/)) {
        const next = line ? `${line} ${word}` : word;
        if (line && g.measureText(next).width > width) { lines.push(line); line = word; } else line = next;
      }
      lines.push(line);
    }
    return lines;
  }
  // The copied / saved picture: the annotated view plus a legend strip (each mark's comment and the
  // parts under it, then study, view and joint angles), so one image carries everything.
  function composite() {
    const W = canvas.width, pad = fs, indent = fs*2.2, lh = fs*1.4, rows = [];
    const g = document.createElement('canvas').getContext('2d');
    for (const m of marks) {
      g.font = `600 ${fs}px system-ui`;
      const head = wrap(g, m.label.trim() || '(no comment)', W - pad*2 - indent);
      g.font = `400 ${Math.round(fs*.85)}px system-ui`;
      rows.push({m, head, parts: wrap(g, `Parts: ${partNames(m)}`, W - pad*2 - indent)});
    }
    const value = record(), v = value.view, joints = Object.entries(value.joints_deg || {}).filter(([, a]) => Math.abs(a) > .05);
    g.font = `400 ${Math.round(fs*.8)}px system-ui`;
    const footer = wrap(g, `${value.study?.title || ''} (${value.study?.result_id || '?'}) · ${value.cad?.view_label || ''} · camera ${v.phrase} · joints ${joints.length ? joints.map(([k, a]) => `${k} ${round(a)}°`).join(', ') : 'all 0°'} · frame: +X = R_ joint side, −Y = face side, +Z up`, W - pad*2);
    const height = rows.reduce((h, r) => h + (r.head.length + r.parts.length)*lh + fs*.5, 0) + footer.length*lh*.9 + pad*2;
    const out = document.createElement('canvas'); out.width = W; out.height = canvas.height + Math.ceil(height);
    const c = out.getContext('2d'); c.drawImage(canvas, 0, 0);
    c.fillStyle = '#f6f8f3'; c.fillRect(0, canvas.height, W, out.height - canvas.height);
    c.fillStyle = '#c9d4cc'; c.fillRect(0, canvas.height, W, 2);
    let y = canvas.height + pad; c.textBaseline = 'top'; c.textAlign = 'left';
    for (const {m, head, parts} of rows) {
      const r = fs*.75; c.fillStyle = COLORS[m.color]; c.beginPath(); c.arc(pad + r, y + r*.9, r, 0, Math.PI*2); c.fill();
      c.fillStyle = m.color === 'yellow' ? '#1d1d1d' : '#fff'; c.font = `700 ${Math.round(fs*.85)}px system-ui`; c.textAlign = 'center';
      c.fillText(String(m.n), pad + r, y + r*.25); c.textAlign = 'left';
      c.fillStyle = '#17231f'; c.font = `600 ${fs}px system-ui`; head.forEach(l => { c.fillText(l, pad + indent, y); y += lh; });
      c.fillStyle = '#4f645a'; c.font = `400 ${Math.round(fs*.85)}px system-ui`; parts.forEach(l => { c.fillText(l, pad + indent, y); y += lh; });
      y += fs*.5;
    }
    c.fillStyle = '#6a7d73'; c.font = `400 ${Math.round(fs*.8)}px system-ui`; footer.forEach(l => { c.fillText(l, pad, y); y += lh*.9; });
    return out;
  }
  function drawMark(m) {
    const P = m.points.map(([x, y]) => [x*canvas.width, y*canvas.height]);
    ctx.strokeStyle = ctx.fillStyle = COLORS[m.color]; ctx.lineWidth = lw; ctx.lineJoin = ctx.lineCap = 'round';
    ctx.beginPath();
    if (m.tool === 'box') { const [a, b] = [P[0], P.at(-1)]; ctx.strokeRect(a[0], a[1], b[0]-a[0], b[1]-a[1]); }
    else { P.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); }
    if (m.tool === 'arrow' && P.length > 1) {
      const [a, b] = [P[0], P.at(-1)], angle = Math.atan2(b[1]-a[1], b[0]-a[0]), head = lw*5;
      ctx.beginPath(); ctx.moveTo(...b);
      ctx.lineTo(b[0]-head*Math.cos(angle-.45), b[1]-head*Math.sin(angle-.45));
      ctx.lineTo(b[0]-head*Math.cos(angle+.45), b[1]-head*Math.sin(angle+.45)); ctx.closePath(); ctx.fill();
    }
    if (!m.n) return;
    const [bx, by] = P[0], r = lw*3.2;
    ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = m.color === 'yellow' ? '#1d1d1d' : '#fff'; ctx.font = `700 ${Math.round(r*1.2)}px system-ui`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(m.n), bx, by + r*.05);
  }
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    marks.forEach(drawMark); if (drawing) drawMark(drawing);
  }
  function renderList() {
    list.innerHTML = marks.map(m => `<li style="--swatch:${COLORS[m.color]}"><span>${m.n}</span><div><textarea rows="2" maxlength="400" data-mark="${m.n}" placeholder="Comment for mark ${m.n}">${esc(m.label)}</textarea><small>${esc(partNames(m))}</small></div></li>`).join('');
    list.querySelectorAll('textarea').forEach(input => input.oninput = () => { marks[Number(input.dataset.mark)-1].label = input.value; });
  }
  const at = event => { const r = canvas.getBoundingClientRect(); return [Math.max(0, Math.min(1, (event.clientX-r.left)/r.width)), Math.max(0, Math.min(1, (event.clientY-r.top)/r.height))]; };
  canvas.onpointerdown = event => { canvas.setPointerCapture(event.pointerId); drawing = {tool, color, points: [at(event)], label: ''}; render(); };
  canvas.onpointermove = event => {
    if (!drawing) return; const p = at(event);
    if (drawing.tool === 'pen') drawing.points.push(p); else drawing.points = [drawing.points[0], p];
    render();
  };
  canvas.onpointerup = () => {
    if (!drawing) return; const m = drawing; drawing = null;
    const [a, b] = [m.points[0], m.points.at(-1)];
    if (m.points.length < 2 || Math.hypot(b[0]-a[0], b[1]-a[1]) < .004 && m.tool !== 'pen') { render(); return; }
    m.n = marks.length + 1; Object.assign(m, partsAt(samplesFor(m), snapshot.pick, mmPerUnit));
    marks.push(m); render(); renderList();
    list.querySelector(`[data-mark="${m.n}"]`)?.focus({preventScroll: true});
  };
  dialog.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => { tool = b.dataset.tool; dialog.querySelectorAll('[data-tool]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); });
  dialog.querySelectorAll('[data-color]').forEach(b => b.onclick = () => { color = b.dataset.color; dialog.querySelectorAll('[data-color]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); });
  dialog.querySelector('.feedback-undo').onclick = () => { marks.pop(); render(); renderList(); };
  function record() {
    const cam = snapshot.camera, center = snapshot.pick(.5, .5);
    return {...context, created_at: new Date().toISOString(),
      view: {phrase: viewPhrase(cam.from_direction), position_mm: mm(cam.position, mmPerUnit), target_mm: mm(cam.target, mmPerUnit),
        distance_mm: round(cam.distance*mmPerUnit), from_direction: cam.from_direction.map(v => round(v, 3)), up: cam.up, fov_deg: cam.fov_deg},
      display: snapshot.display, selected_part: snapshot.selected_part,
      center_part: center && {name: center.name, role: center.role, owning_link: center.owning_link, point_mm: mm(center.point, mmPerUnit)},
      image: {width: canvas.width, height: canvas.height},
      marks: marks.map(({n, tool: t, color: c, label, parts, samples, hits, points}) => ({n, tool: t, color: c, label: label.trim(), parts, samples, hits,
        start: points[0].map(v => round(v, 3)), end: points.at(-1).map(v => round(v, 3))}))};
  }
  const png = () => new Promise(resolve => composite().toBlob(resolve, 'image/png'));
  const fileName = () => `design-feedback-${(context.study?.result_id || 'cad').replace(/[^\w.-]+/g, '-')}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')}.png`;
  const say = text => { status.textContent = text; status.scrollIntoView({block: 'nearest'}); };
  dialog.querySelector('.feedback-copy-image').onclick = async () => {
    try { await navigator.clipboard.write([new ClipboardItem({'image/png': png()})]); say('Image copied with the comments and parts. Paste it into the chat.'); }
    catch (error) { say(`Copy failed: ${error.message}. Use Download PNG instead.`); }
  };
  dialog.querySelector('.feedback-download').onclick = () => {
    const a = document.createElement('a'); a.href = composite().toDataURL('image/png'); a.download = fileName(); a.click();
  };
  const save = dialog.querySelector('.feedback-save');
  if (save) save.onclick = async () => {
    save.disabled = true; say('Saving…');
    try {
      const value = record();
      const response = await fetch('/api/design-feedback', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image_png: composite().toDataURL('image/png'), context: value, markdown: feedbackMarkdown(value)})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Save failed');
      status.innerHTML = `Saved to <code>${esc(result.folder)}</code>. <button type="button" class="feedback-copy-prompt">Copy prompt for the agent</button>`; status.scrollIntoView({block: 'nearest'});
      status.querySelector('button').onclick = async () => {
        try { await navigator.clipboard.writeText(result.prompt); status.querySelector('button').textContent = 'Prompt copied'; }
        catch (error) { say(result.prompt); }
      };
    } catch (error) { say(`Not saved: ${error.message}. Use Copy image instead.`); save.disabled = false; }
  };
  const close = () => { document.removeEventListener('keydown', key); dialog.remove(); };
  const key = event => { if (event.key === 'Escape' && !marks.length) close(); };
  document.addEventListener('keydown', key);
  dialog.querySelector('.feedback-close').onclick = () => { if (!marks.length || confirm('Discard this snapshot and its marks?')) close(); };
  return {close};
}
