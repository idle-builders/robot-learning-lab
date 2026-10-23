import {designEvidence} from './design-evidence.js?v=evidence-links-20261021';
import {briefText, evidenceText} from './readable.js?v=studies-final-20261020';
import {cadMarkup, mountCadCards} from './cad-catalog.js?v=pan-20261021';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const asset = path => `/asset?path=${encodeURIComponent(path)}`;
const judgments = {unreviewed:'Not reviewed',shortlist:'Shortlist',revise:'Needs revision',reject:'Reject'};
const text = value => typeof value === 'string' ? value : JSON.stringify(value);
function sourceLink(ref) {
  const label = esc(ref.label || ref.path || ref.url);
  if (ref.url && /^https?:\/\//.test(ref.url)) return `<a href="${esc(ref.url)}" target="_blank" rel="noopener">${label}</a>`;
  if (/^(studies|artifacts)\//.test(ref.path || '')) return `<a href="${asset(ref.path)}" target="_blank" rel="noopener">${label}</a>`;
  return `<span title="${esc(ref.path)}">${label}</span>`;
}
export async function mountCatalog(host, resultId, catalog, {compact = false} = {}) {
  let disposed = false;
  const cards = new Map();
  const options = catalog.groups.flatMap(group => group.options);
  host.innerHTML = `<section id="design-catalog" class="design-catalog"><div class="catalog-intro"><span class="eyebrow">FULL DESIGN CATALOG</span><h2>${esc(catalog.title)}</h2><p>${esc(catalog.description)}</p><p><strong>${options.length} designs · ${catalog.groups.length} groups.</strong> Every alternative is shown below. Your saved comments and judgments belong to each individual design.</p><p class="catalog-progress" role="status">Loading saved reviews…</p><p class="muted">Shortlisting records your preference; it does not select or qualify a mechanical design. Illustrations marked schematic describe the proposal, not fabrication-ready CAD.</p></div><nav class="catalog-jump" aria-label="Jump to a design section">${catalog.groups.map((group,i) => `<a href="#catalog-group-${esc(group.id)}">${i+1}. ${esc(group.title)}</a>`).join('')}</nav>${catalog.groups.map((group,i) => `<section class="catalog-group" id="catalog-group-${esc(group.id)}"><div class="catalog-group-heading"><span>${esc(group.category)} · ${i+1} / ${catalog.groups.length}</span><h3>${esc(group.title)}</h3><a href="#design-catalog">Back to section index ↑</a></div><div class="design-options">${group.options.map(option => `<article class="design-option" data-design-id="${esc(option.id)}"><div class="option-heading"><span class="option-id">${esc(option.id)}</span><h4>${esc(option.title)}</h4></div>${cadMarkup(option.cad)}${option.cad ? '' : !option.image_path ? '<div class="empty-cad">Native source files only · no preview registered for this version.</div>' : `<a class="option-image" href="${asset(option.image_path)}" target="_blank" rel="noopener" aria-label="Enlarge ${esc(option.title)} illustration"><img src="${asset(option.image_path)}" loading="lazy" alt="${esc(option.title)} — ${esc(option.visual_kind)}"></a>`}<div class="option-body"><p class="option-maturity">${esc(option.visual_kind.replaceAll('_',' '))}</p>${briefText(option.summary, "Full design summary")}${option.evidence?.length ? `<details class="option-detail"><summary>Measurements & simulations · ${option.evidence.length}</summary>${designEvidence(option.evidence)}</details>` : ''}<details class="option-detail"><summary>Status & maturity</summary>${evidenceText(option.maturity)}</details>${option.description && option.description !== option.summary ? `<details class="option-detail"><summary>Mechanism & engineering detail</summary>${evidenceText(option.description)}</details>` : ''}${option.notes?.length ? `<details class="option-detail"><summary>Evidence notes · ${option.notes.length}</summary>${option.notes.map(note=>evidenceText(text(note),2)).join('')}</details>` : ''}<details class="option-detail"><summary>Sources · ${(option.source_refs || []).length}</summary><ul class="source-links">${(option.source_refs || []).map(ref=>`<li>${sourceLink(ref)}</li>`).join('')}</ul></details></div><details class="review-panel"><summary>My review & comments</summary><form class="design-review"><label>My judgment<select name="judgment" disabled>${Object.entries(judgments).map(([value,label])=>`<option value="${value}">${label}</option>`).join('')}</select></label><label>My comments<textarea name="comment" maxlength="10000" rows="4" placeholder="What works? What should change?" disabled></textarea></label><div class="review-actions"><button type="submit" disabled>Save review</button><span class="review-status" role="status">Loading…</span></div><div class="review-conflict" hidden></div></form></details></article>`).join('')}</div></section>`).join('')}</section>`;
  if (compact) {
    const intro = host.querySelector('.catalog-intro');
    intro.innerHTML = `<div class="compare-heading"><h3>${esc(catalog.title)}</h3><span>${options.length} alternatives</span></div><p class="catalog-progress" role="status">Loading saved reviews…</p>`;
    host.querySelector('.catalog-jump').remove();
    host.querySelectorAll('.catalog-group-heading').forEach(el => el.remove());
    host.querySelector('.design-catalog').classList.add('focused-catalog');
  }
  const cadCards = mountCadCards(host,options);
  const progress = host.querySelector('.catalog-progress');
  const updateProgress = () => { const saved = [...cards.values()].filter(state=>state.review.judgment !== 'unreviewed').length; progress.textContent = `${saved} of ${options.length} designs judged · ${[...cards.values()].filter(state=>state.review.comment).length} with saved comments`; };
  const draftKey = id => `ape-design-review:${resultId}:${id}`;
  const draftRead = id => { try { return JSON.parse(localStorage.getItem(draftKey(id))); } catch { return null; } };
  const draftWrite = (id,value) => { try { if(value) localStorage.setItem(draftKey(id),JSON.stringify(value)); else localStorage.removeItem(draftKey(id)); return true; } catch { return false; } };
  try {
    const response = await fetch(`/api/design-reviews?result=${encodeURIComponent(resultId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Reviews unavailable');
    if (disposed || !host.isConnected) return {dispose(){disposed=true;cadCards.dispose();}};
    const reviews = new Map(data.reviews.map(review=>[review.design_id,review]));
    host.querySelectorAll('.design-option').forEach(card=>{
      const id = card.dataset.designId, form = card.querySelector('form');
      const status = form.querySelector('.review-status'), conflict = form.querySelector('.review-conflict');
      const select = form.elements.judgment, comment = form.elements.comment, button = form.querySelector('button');
      const state = {review:reviews.get(id) || {design_id:id,judgment:'unreviewed',comment:'',version:0},saving:false};
      state.expectedVersion = state.review.version;
      cards.set(id,state);
      const values = ()=>({judgment:select.value,comment:comment.value});
      const apply = review=>{select.value=review.judgment;comment.value=review.comment;card.dataset.judgment=review.judgment;};
      const savedLabel = ()=>state.review.version ? `Saved · ${new Date(state.review.updated_at).toLocaleString()}` : 'No saved review yet';
      apply(state.review);
      const draft = draftRead(id);
      if (draft && judgments[draft.judgment] && typeof draft.comment === 'string' && (draft.comment !== state.review.comment || draft.judgment !== state.review.judgment)) { apply(draft); state.expectedVersion=Number.isInteger(draft.base_version) ? draft.base_version : 0; status.textContent=state.expectedVersion !== state.review.version ? 'Draft restored; saved review changed. Save will ask you to resolve the conflict.' : 'Unsaved draft restored'; }
      else { draftWrite(id,null); status.textContent=savedLabel(); }
      select.disabled=comment.disabled=button.disabled=false;
      const keepDraft=()=>draftWrite(id,{...values(),base_version:state.expectedVersion});
      const changed=()=>{const retained=keepDraft();status.textContent=retained ? 'Unsaved changes' : 'Unsaved changes — browser draft storage unavailable';card.dataset.judgment=select.value;};
      select.onchange=changed;comment.oninput=changed;
      form.onsubmit=async event=>{
        event.preventDefault(); if(state.saving)return;
        state.saving=true;button.disabled=true;status.textContent='Saving…';conflict.hidden=true;
        const submitted=values();
        try {
          const response = await fetch('/api/design-reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({result_id:resultId,design_id:id,...submitted,expected_version:state.expectedVersion})});
          const result=await response.json();
          if(disposed || !host.isConnected) return;
          if(response.status===409){
            state.review=result.review;
            state.expectedVersion=result.review.version;
            conflict.hidden=false;
            conflict.innerHTML=`<p>Another review was saved while you were editing. Your draft is still here.</p><p><strong>Saved judgment:</strong> ${esc(judgments[result.review.judgment])}</p><p class="saved-comment">${esc(result.review.comment || '(No comment)')}</p><button type="button" class="use-saved">Use saved review</button><p>Or click Save review again to replace it with your draft.</p>`;
            conflict.querySelector('button').onclick=()=>{apply(state.review);draftWrite(id,null);conflict.hidden=true;status.textContent=savedLabel();updateProgress();};
            status.textContent='Conflict — choose which review to keep';updateProgress();return;
          }
          if(!response.ok)throw new Error(result.error || 'Save failed');
          state.review=result.review;
          state.expectedVersion=result.review.version;
          if(JSON.stringify(values())===JSON.stringify(submitted)){draftWrite(id,null);status.textContent=savedLabel();}
          else {keepDraft();status.textContent='Earlier changes saved; current edits are unsaved';}
          updateProgress();
        } catch(error){if(disposed || !host.isConnected)return;const retained=keepDraft();status.textContent=`Not saved: ${error.message}. ${retained ? 'Your draft is kept in this browser.' : 'Draft storage unavailable; keep this page open or copy your comment.'}`;}
        finally {state.saving=false;button.disabled=false;}
      };
    });
    updateProgress();
  } catch(error){progress.textContent=`Saved reviews could not load: ${error.message}. Reload to try again. Designs remain available below.`;host.querySelectorAll('.review-status').forEach(status=>status.textContent='Reviews unavailable');}
  return {dispose(){disposed=true;cadCards.dispose();}};
}
