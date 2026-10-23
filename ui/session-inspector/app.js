const $ = id => document.getElementById(id);
const panels = ['overview', 'video', '3d', 'data', 'interactive'];
const labels = {overview:'Overview',video:'Video', '3d':'3D model',data:'Evidence',interactive:'Interactive'};
let sessions = [], selected = null, activeTab = 'overview', kind = 'Featured', viewer = null;

function route() {
  const params = new URLSearchParams(location.search);
  return {session: params.get('session'), tab: params.get('tab')};
}
function setRoute(push = true) {
  const url = new URL(location.href);
  url.searchParams.set('view', 'archive');
  url.searchParams.set('session', selected.id);
  url.searchParams.set('tab', activeTab);
  url.searchParams.delete('skill'); url.searchParams.delete('clip');
  (push ? history.pushState : history.replaceState).call(history, null, '', url);
}
function categoryIcon(value) { return value === 'Learning' ? '↗' : value === 'Engineering' ? '⬡' : value === 'Reference' ? '◎' : '•'; }
function stamp(epoch) { return epoch ? new Date(epoch * 1000).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'}) : 'Saved run'; }
function countText(n, noun) { return `${n} ${noun}${n === 1 ? '' : 's'}`; }
function selectOption(select, items) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement('option');
    option.value = item.url; option.textContent = item.name;
    select.append(option);
  }
}

function renderList() {
  const query = $('search').value.trim().toLowerCase();
  const filtered = sessions.filter(s => (!query && kind === 'Featured' ? s.featured : (query || kind === 'All' || s.kind === kind)) && (!query || `${s.title} ${s.id} ${s.summary}`.toLowerCase().includes(query)));
  const shown = query ? filtered.slice(0, 250) : filtered.slice(0, 150);
  $('session-count').textContent = shown.length < filtered.length ? `${shown.length}/${filtered.length}` : String(filtered.length);
  $('session-list').replaceChildren();
  for (const session of shown) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'session-item'; button.setAttribute('role','option');
    button.setAttribute('aria-selected', session.id === selected?.id ? 'true' : 'false');
    const icon = document.createElement('span'); icon.className = 'session-icon'; icon.textContent = session.current ? '★' : categoryIcon(session.kind);
    const copy = document.createElement('span');
    const name = document.createElement('strong'); name.textContent = session.title;
    const detail = document.createElement('small'); detail.textContent = `${session.current ? 'Current design · ' : ''}${session.kind} · ${stamp(session.updated)}`;
    copy.append(name, detail); button.append(icon, copy);
    button.addEventListener('click', () => selectSession(session.id));
    $('session-list').append(button);
  }
  if (!shown.length) $('session-list').textContent = 'No matching sessions.';
}

function availableTabs(session) {
  return panels.filter(tab => tab === 'overview' || (tab === 'video' && session.videos.length) || (tab === '3d' && session.models.length) || (tab === 'data' && session.data.length) || (tab === 'interactive' && session.landing));
}
function renderTabs() {
  $('tabs').replaceChildren();
  for (const tab of availableTabs(selected)) {
    const button = document.createElement('button'); button.type = 'button'; button.textContent = labels[tab];
    button.setAttribute('aria-selected', tab === activeTab ? 'true' : 'false');
    button.addEventListener('click', () => selectTab(tab));
    $('tabs').append(button);
  }
  for (const tab of panels) $('panel-' + tab).hidden = tab !== activeTab;
}
function renderOverview() {
  $('hero-image').hidden = !selected.thumbnail;
  $('hero-placeholder').hidden = !!selected.thumbnail;
  const images = selected.images || [];
  $('image-toolbar').hidden = !images.length;
  selectOption($('image-select'), images);
  if (images.length) {
    $('image-select').value = selected.thumbnail || images[0].url;
    updateImage();
  } else {
    $('hero-image').removeAttribute('src');
    $('image-link').removeAttribute('href');
  }
  $('overview-copy').textContent = selected.summary || `This retained session contains ${selected.videos.length ? countText(selected.videos.length,'recording') : 'saved evidence'}${selected.models.length ? `, ${countText(selected.models.length,'3D model')}` : ''}${selected.data.length ? `, and ${countText(selected.data.length,'evidence file')}` : ''}. Open a view to inspect it without leaving this page.`;
  $('quick-actions').replaceChildren();
  for (const tab of availableTabs(selected).filter(t => t !== 'overview')) {
    const button = document.createElement('button'); button.type = 'button';
    button.textContent = `Inspect ${labels[tab].toLowerCase()} →`;
    button.addEventListener('click', () => selectTab(tab));
    $('quick-actions').append(button);
  }
}
function updateImage() {
  const option = $('image-select').selectedOptions[0];
  if (!option) return;
  $('hero-image').src = option.value;
  $('hero-image').alt = option.textContent;
  $('image-link').href = option.value;
}
function showVideo() {
  selectOption($('video-select'), selected.videos);
  if (!selected.videos.length) return;
  $('video-select').value = selected.videos[0].url;
  updateVideo();
}
function updateVideo() {
  $('video').src = $('video-select').value;
  $('video-name').textContent = $('video-select').selectedOptions[0]?.textContent || '';
}
async function showModel() {
  selectOption($('model-select'), selected.models);
  if (!selected.models.length) return;
  $('model-select').value = selected.models[0].url;
  await updateModel();
}
async function updateModel() {
  $('model-message').textContent = 'Loading 3D viewer…';
  $('model-message').hidden = false;
  try {
    if (!viewer) {
      const module = await import('./viewer3d.js');
      viewer = await module.createViewer($('model-stage'), $('mesh-label'));
    }
    await viewer.load($('model-select').value);
    $('model-message').hidden = true;
  } catch (error) {
    $('model-message').textContent = `Unable to load this model: ${error.message}`;
    $('model-message').hidden = false;
  }
}
async function showData() {
  const preferred = selected.data.find(x => x.name === 'report.json') || selected.data.find(x => x.name === 'result.json') || selected.data[0];
  selectOption($('data-select'), selected.data);
  if (!preferred) return;
  $('data-select').value = preferred.url;
  await updateData();
}
async function updateData() {
  const file = $('data-select').selectedOptions[0]?.textContent;
  if (!file || !selected) return;
  $('data-link').href = $('data-select').value;
  $('data-text').textContent = 'Loading evidence…';
  try {
    const response = await fetch(`/api/preview?session=${encodeURIComponent(selected.id)}&file=${encodeURIComponent(file)}`, {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    let value = payload.text;
    if (!payload.truncated && file.endsWith('.json')) {
      try { value = JSON.stringify(JSON.parse(value), null, 2); } catch { /* display source */ }
    }
    $('data-text').textContent = value + (payload.truncated ? '\n\n…preview limited to 120 KB; download the full file above.' : '');
  } catch (error) { $('data-text').textContent = `Could not read evidence: ${error.message}`; }
}
function showInteractive() {
  if (selected.landing && $('legacy-frame').getAttribute('src') !== selected.landing) $('legacy-frame').src = selected.landing;
}
function selectTab(tab, push = true) {
  if (!selected || !availableTabs(selected).includes(tab)) return;
  if (activeTab === 'video' && tab !== 'video') $('video').pause();
  activeTab = tab;
  renderTabs();
  if (tab === 'video') showVideo();
  if (tab === '3d') showModel();
  if (tab === 'data') showData();
  if (tab === 'interactive') showInteractive();
  setRoute(push);
}
function selectSession(id, tab = 'overview', push = true) {
  const next = sessions.find(s => s.id === id);
  if (!next) return;
  $('video').pause(); $('video').removeAttribute('src'); $('video').load();
  $('legacy-frame').src = 'about:blank';
  selected = next;
  activeTab = availableTabs(next).includes(tab) ? tab : 'overview';
  $('content').hidden = false; $('empty').hidden = true;
  $('current-marker').hidden = !selected.current;
  $('kind').textContent = selected.kind;
  $('updated').textContent = stamp(selected.updated);
  $('run-id').textContent = selected.id;
  $('title').textContent = selected.title;
  $('summary').textContent = selected.summary || 'Saved experiment session';
  const counts = [['video',selected.videos.length],['3D',selected.models.length],['file',selected.data.length]];
  $('asset-counts').replaceChildren();
  for (const [name, count] of counts.filter(([,n]) => n)) {
    const badge = document.createElement('span'); badge.textContent = countText(count,name); $('asset-counts').append(badge);
  }
  if (selected.landing) { const badge = document.createElement('span'); badge.textContent = 'Interactive view'; $('asset-counts').append(badge); }
  renderOverview(); renderList(); renderTabs();
  if (activeTab !== 'overview') selectTab(activeTab, false);
  else setRoute(push);
}
async function refresh(preserve = true) {
  $('sync').textContent = 'Refreshing…';
  try {
    const response = await fetch('/api/sessions', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    sessions = (await response.json()).sessions;
    sessions.sort((a,b) => Number(b.current)-Number(a.current) || Number(b.featured)-Number(a.featured) || b.updated-a.updated || a.title.localeCompare(b.title));
    const wanted = preserve && selected ? selected.id : route().session;
    const found = sessions.find(s => s.id === wanted) || sessions.find(s => s.id === 'ape-cove50-transfer-20261008') || sessions.find(s => s.id === 'ape-frame50-direct-skeleton') || sessions[0];
    if (found) selectSession(found.id, preserve && selected ? activeTab : route().tab || 'overview', false);
    else { $('content').hidden = true; $('empty').hidden = false; $('empty').textContent = 'No inspectable artifact sessions found.'; renderList(); }
    $('sync').textContent = `${sessions.length} retained sessions`;
  } catch (error) {
    $('sync').textContent = 'Catalog unavailable';
    $('empty').textContent = `Could not load the session catalog: ${error.message}. Start with python tools/session_inspector.py serve.`;
  }
}

$('search').addEventListener('input', renderList);
$('filters').addEventListener('click', event => {
  const button = event.target.closest('button[data-kind]'); if (!button) return;
  kind = button.dataset.kind;
  for (const candidate of $('filters').querySelectorAll('button')) candidate.setAttribute('aria-pressed',candidate === button ? 'true':'false');
  renderList();
});
$('refresh').addEventListener('click', () => refresh());
$('copy-link').addEventListener('click', async () => { await navigator.clipboard.writeText(location.href); $('copy-link').textContent = 'Copied'; setTimeout(() => $('copy-link').textContent = 'Copy view link', 1500); });
$('video-select').addEventListener('change', updateVideo);
$('model-select').addEventListener('change', updateModel);
$('fit-model').addEventListener('click', () => viewer?.fit());
$('data-select').addEventListener('change', updateData);
$('image-select').addEventListener('change', updateImage);
$('reload-frame').addEventListener('click', () => { if (selected?.landing) $('legacy-frame').src = selected.landing; });
window.addEventListener('popstate', () => {
  const next = route();
  if (next.session && sessions.length) selectSession(next.session, next.tab || 'overview', false);
});
refresh(false);
