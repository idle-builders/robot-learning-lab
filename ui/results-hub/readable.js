// Preserve source wording while giving long evidence a readable hierarchy.
export const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function chunks(value) {
  const source = typeof value === 'string' ? value : JSON.stringify(value ?? '');
  return source.split(/\n+|;\s+|(?<=[.!?])\s+(?=[A-Z])/).map(s => s.trim()).filter(Boolean).flatMap(s => {
    const words = s.split(/\s+/), lines = []; let line = '';
    for (const word of words) { if (line.length + word.length > 220) { lines.push(line); line = ''; } line += (line ? ' ' : '') + word; }
    if (line) lines.push(line); return lines;
  });
}
export function evidenceText(value, limit = 3) {
  const items = chunks(value);
  if (!items.length) return '<p class="muted">Not recorded.</p>';
  const list = data => `<ul class="evidence-points">${data.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
  return list(items.slice(0, limit)) + (items.length > limit ? `<details class="text-more"><summary>Read ${items.length - limit} more evidence notes</summary>${list(items.slice(limit))}</details>` : '');
}
export function briefText(value, label = 'Full observation') {
  const source = String(value ?? '').trim(); if (!source) return '';
  const first = chunks(source)[0] || '';
  return `<p class="lede">${esc(first)}${source.length > first.length ? ' …' : ''}</p>${source.length > first.length ? `<details class="text-more"><summary>${esc(label)}</summary>${evidenceText(source, 6)}</details>` : ''}`;
}
export function shortText(value, max = 150) {
  const text = String(value ?? ''); if (text.length <= max) return text;
  const part = text.slice(0, max); return part.slice(0, part.lastIndexOf(' ')) + '…';
}
