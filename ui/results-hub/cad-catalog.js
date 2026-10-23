import {evidenceText} from './readable.js?v=studies-final-20261020';
import {bomMarkup, bindBom} from './cad-bom.js?v=hip-inspection-20261020';
const asset=path=>`/asset?path=${encodeURIComponent(path)}`;
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function inspectionMarkup() { return `<details class="cad-inspect-tools"><summary>Expose, cut & explode</summary><div class="inspect-tools"><label>Explode parts<input type="range" class="cad-explode" aria-label="Explode parts" min="0" max="100" value="0"><output>0%</output></label><label class="cut-enable"><input type="checkbox" class="cad-cut-enabled"> Cut through assembly</label><label>Section axis<select class="cad-cut-axis"><option value="x">X · side</option><option value="y">Y · front</option><option value="z" selected>Z · height</option></select></label><label>Section position<input type="range" class="cad-cut-position" aria-label="Section position" min="0" max="100" value="50"><output>50%</output></label><label class="cut-enable"><input type="checkbox" class="cad-cut-reverse"> Reverse cut direction</label><p>Display separation and open section surfaces; source geometry stays intact.</p></div></details>`; }
export function bindInspectionTools(host, getViewer) {
  const tools=host.querySelector('.inspect-tools');
  const apply=()=>{
    const explode=tools.querySelector('.cad-explode'),position=tools.querySelector('.cad-cut-position'),viewer=getViewer();
    explode.nextElementSibling.value=`${explode.value}%`;position.nextElementSibling.value=`${position.value}%`;
    viewer?.explode(explode.value);
    viewer?.section({enabled:tools.querySelector('.cad-cut-enabled').checked,axis:tools.querySelector('.cad-cut-axis').value,position:position.value,reverse:tools.querySelector('.cad-cut-reverse').checked});
  };
  tools.querySelectorAll('input,select').forEach(input=>input.oninput=apply);
  return apply;
}
export function navigationMarkup() { return `<div class="cad-navigation" role="group" aria-label="CAD navigation"><span>Drag mode</span><button type="button" data-cad-navigation="orbit" aria-pressed="true">Orbit</button><button type="button" data-cad-navigation="pan" aria-pressed="false">Pan</button></div>`; }
export function cadMarkup(cad){
  if(!cad?.views?.length)return '';
  const anchors=cad.inspection_anchor_links || [];
  const anchorMarkup=anchors.length ? `<label class="inspection-anchor-wrap" hidden>Hold still<select class="inspection-anchor" aria-label="Inspection anchor">${anchors.map(a=>`<option value="${esc(a.link)}">${esc(a.label || a.link)}</option>`).join('')}</select><small>Reference frame only; support and loads are not simulated.</small></label>` : '';
  return `<div class="cad-inspection"><div class="cad-canvas" aria-label="Interactive CAD assembly"></div><p class="cad-part-label" role="status">CAD preview loads when visible.</p><label>CAD configuration<select class="cad-view-choice">${cad.views.map((v,i)=>`<option value="${i}">${esc(v.label)}</option>`).join('')}</select></label><div class="cad-controls"><label><input type="checkbox" class="cad-show-shells" ${cad.show_shells === false ? '' : 'checked'}> ${esc(cad.body_panel_label || 'Ape body panels')}</label>${cad.neighbor_reference_label ? `<label><input type="checkbox" class="cad-show-reference" ${cad.show_neighbor_reference ? 'checked' : ''}> ${esc(cad.neighbor_reference_label)}</label>` : ''}<button type="button" class="cad-fit">Fit mechanism</button>${cad.has_body_reference ? `<button type="button" class="cad-fit-all">${esc(cad.fit_reference_label || 'Fit ape body')}</button>` : ''}</div>${navigationMarkup()}${inspectionMarkup()}<div class="cad-motion" aria-label="Motion controls"><p class="motion-status" role="status">Checking native joint data…</p>${anchorMarkup}<div class="joint-controls"></div><div class="pose-controls" ${cad.views.length > 1 ? '' : 'hidden'}><label>Saved configuration<input type="range" class="pose-scrub" min="0" max="${cad.views.length-1}" step="1" value="0"><output class="pose-output">${esc(cad.views[0].label)}</output></label></div><div class="motion-actions"><button type="button" class="motion-play" disabled>Play motion</button><button type="button" class="motion-reset" disabled>Reset</button><label>Speed<select class="motion-speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label></div></div><details class="cad-scope"><summary>Scope of this CAD view</summary>${evidenceText(cad.scope || 'Saved native CAD configurations; inspection does not establish physical behavior.',2)}</details><div class="cad-downloads">${cad.step_path?`<a href="${asset(cad.step_path)}" download>Download assembly STEP</a>`:''}${cad.report_path?`<a href="${asset(cad.report_path)}" target="_blank" rel="noopener">Evaluation and failures</a>`:''}</div>${bomMarkup(cad)}</div>`;
}
export function mountCadCards(host,options){
  let active=true;const states=[];
  const byId=new Map(options.filter(o=>o.cad?.views?.length).map(o=>[o.id,o]));
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{
    const state=states.find(s=>s.card===entry.target);if(!state)return;
    state.visible=entry.isIntersecting;
    if(!entry.isIntersecting){stop(state);++state.generation;state.viewer?.dispose();state.viewer=null;state.rig=null;return;}
    initialize(state);
  }),{rootMargin:'100px'});
  function stop(state){state.playing=false;cancelAnimationFrame(state.frame);state.play.textContent='Play motion';}
  function pose(state){
    const values=Object.fromEntries([...state.joints.querySelectorAll('input')].map(input=>[input.dataset.joint,Number(input.value)]));
    state.jointValues=values;
    state.joints.querySelectorAll('input').forEach(input=>{input.nextElementSibling.value=`${Number(input.value).toFixed(1)}°`;});
    const anchorLink=state.card.querySelector('.inspection-anchor')?.value || null;
    const anchor=state.option.cad.inspection_anchor_links?.find(a=>a.link===anchorLink);
    state.rig?.pose(values,{anchorLink,anchorLinksByRoot:anchor?.links_by_root || null});
  }
  function configureRig(state){
    state.rig=state.option.cad.motion_mode!=='saved_configurations'&&Number(state.choice.value)===0&&state.metadata ? state.viewer?.rig(state.metadata,{scale:state.option.cad.model_units==='mm'?1:.001,limits:state.option.cad.motion_limits || {}}) : null;
    state.joints.innerHTML=state.rig ? state.rig.controls.map(j=>`<label>${esc(j.name.replaceAll('_',' '))}<input type="range" data-joint="${esc(j.name)}" aria-label="${esc(j.name.replaceAll('_',' '))} angle" min="${j.range[0]}" max="${j.range[1]}" step="0.1" value="${Math.max(j.range[0],Math.min(j.range[1],state.jointValues?.[j.name] ?? 0))}"><output>0.0°</output></label>`).join('') : '';
    state.joints.querySelectorAll('input').forEach(input=>input.oninput=()=>{stop(state);pose(state);});
    const anchor=state.card.querySelector('.inspection-anchor-wrap');if(anchor)anchor.hidden=!state.rig;
    const references=state.rig?.mappedReferenceParts || 0;
    const meshCount=state.rig ? `${state.rig.mappedParts-references} mechanism meshes${references ? ` + ${references} reference meshes` : ''}` : '';
    state.status.textContent=state.rig ? `Joint motion · ${meshCount}${state.envelope ? "; ranges restricted to native and saved-pose bounds" : ""}. ${state.option.cad.motion_note || 'Kinematic preview; contact, loads, flexible cables and auxiliary mechanisms are not recomputed.'}` : state.option.cad.views.length>1 ? 'Saved configurations · scrub or play the recorded CAD states. Continuous articulation is unavailable for this export.' : 'Static export · this source has no joint model or alternate configurations.';
    state.play.disabled=state.reset.disabled=!state.rig&&state.option.cad.views.length<2;
    if(state.rig)pose(state);
  }
  function applyInspection(state){
    const tools=state.card.querySelector('.inspect-tools'), explode=tools.querySelector('.cad-explode'), position=tools.querySelector('.cad-cut-position');
    explode.nextElementSibling.value=`${explode.value}%`;position.nextElementSibling.value=`${position.value}%`;
    state.viewer?.explode(explode.value);
    state.viewer?.section({enabled:tools.querySelector('.cad-cut-enabled').checked,axis:tools.querySelector('.cad-cut-axis').value,position:position.value,reverse:tools.querySelector('.cad-cut-reverse').checked});
  }
  async function loadChoice(state,preserveView=true){
    if(!state.viewer)return;
    const request=++state.loadGeneration;
    const requested=Number(state.choice.value),view=state.option.cad.views[requested];
    state.scrub.value=String(requested);state.output.value=view.label;
    try {await state.viewer.load(asset(view.model_path),{preserveView});if(active&&state.visible&&request===state.loadGeneration){state.loadedChoice=requested;configureRig(state);if(!preserveView&&state.option.cad.fit_reference_on_load)state.viewer.fit({all:true});}}
    catch(error){if(active&&request===state.loadGeneration){
      stop(state);
      if(state.loadedChoice!==undefined){
        state.choice.value=state.scrub.value=String(state.loadedChoice);
        const retained=state.option.cad.views[state.loadedChoice].label;
        state.output.value=retained;
        state.label.textContent=`Could not load ${view.label}: ${error.message}. Still showing ${retained}.`;
      }else{state.rig=null;state.joints.innerHTML='';state.play.disabled=state.reset.disabled=true;state.label.textContent=`Could not load CAD configuration: ${error.message}`;}
    }}
  }
  async function initialize(state){
    if(state.viewer||state.loading||!active)return;state.loading=true;const request=++state.generation;
    try{
      const {createCadViewer}=await import('./cad-viewer.js?v=pan-20261021');
      if(!state.metadata&&state.option.cad.parts_path){const response=await fetch(asset(state.option.cad.parts_path));if(!response.ok)throw new Error('Native part manifest unavailable');state.metadata=await response.json();}
      if(state.metadata && state.option.cad.motion_contract_path && !state.option.cad.motion_limits) {
        const responses=await Promise.all([fetch(asset(state.option.cad.motion_contract_path)),fetch(asset(state.option.cad.report_path))]);
        if(responses.every(r=>r.ok)) {
          const [contract,report]=await Promise.all(responses.map(r=>r.json()));
          if(Array.isArray(contract.physical_order)&&report.pose_glbs?.every(p=>p.q_deg?.length===contract.physical_order.length)) {
            state.option.cad.motion_limits=Object.fromEntries(contract.physical_order.map((name,i)=>[name,[Math.min(0,...report.pose_glbs.map(p=>p.q_deg[i])),Math.max(0,...report.pose_glbs.map(p=>p.q_deg[i]))]]));
            state.envelope=true;
          }
        }
      }
      if(!active||!state.visible||request!==state.generation)return;
      state.viewer=createCadViewer(state.canvas,state.label,state.metadata?.parts||[],{bodyPanelOpacity:state.option.cad.body_panel_opacity,hiddenReferenceParts:state.option.cad.hidden_reference_parts});
      state.viewer.navigation(state.navigationMode || 'orbit');
      state.loadedChoice=undefined;
      state.viewer.shells(state.shells.checked);
      state.viewer.references(state.references?.checked || false);
      await loadChoice(state,false);
      applyInspection(state);
      if (state.isolated) state.viewer.isolate(state.isolated);
    }catch(error){if(active&&request===state.generation)state.label.textContent=`CAD preview unavailable: ${error.message}. Download the source to inspect locally.`;}
    finally{state.loading=false;if(active&&state.visible&&!state.viewer&&request!==state.generation)initialize(state);}
  }
  host.querySelectorAll('.design-option').forEach(card=>{
    const option=byId.get(card.dataset.designId);if(!option)return;
    if(host.closest('.single-study')) {
      const inspection=card.querySelector('.cad-inspection'),panel=document.createElement('div');panel.className='cad-tool-panel';
      [...inspection.children].filter(child=>!child.matches('.cad-canvas,.cad-part-label')).forEach(child=>panel.append(child));
      card.querySelector('.option-body').prepend(panel);
    }

    const state={card,option,viewer:null,metadata:null,rig:null,loading:false,visible:false,generation:0,loadGeneration:0,playing:false,frame:0,canvas:card.querySelector('.cad-canvas'),label:card.querySelector('.cad-part-label'),choice:card.querySelector('.cad-view-choice'),shells:card.querySelector('.cad-show-shells'),references:card.querySelector('.cad-show-reference'),status:card.querySelector('.motion-status'),joints:card.querySelector('.joint-controls'),play:card.querySelector('.motion-play'),reset:card.querySelector('.motion-reset'),speed:card.querySelector('.motion-speed'),scrub:card.querySelector('.pose-scrub'),output:card.querySelector('.pose-output')};
    state.navigationMode='orbit';
    card.querySelectorAll('[data-cad-navigation]').forEach(button=>button.onclick=()=>{
      state.navigationMode=button.dataset.cadNavigation;
      card.querySelectorAll('[data-cad-navigation]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
      state.viewer?.navigation(state.navigationMode);
    });
    state.disposeBom=bindBom(card,option.cad);
    state.isolateHandler=event=>{state.isolated=event.detail.names;state.viewer?.isolate(state.isolated);};
    card.addEventListener('cad-isolate-parts',state.isolateHandler);
    states.push(state);observer.observe(card);
    state.card.querySelectorAll('.inspect-tools input,.inspect-tools select').forEach(input=>input.oninput=()=>applyInspection(state));
    state.choice.onchange=()=>{stop(state);loadChoice(state);};
    state.scrub.oninput=()=>{stop(state);state.choice.value=state.scrub.value;loadChoice(state);};
    state.shells.onchange=()=>state.viewer?.shells(state.shells.checked);
    const anchor=card.querySelector('.inspection-anchor');if(anchor)anchor.onchange=()=>{stop(state);pose(state);};
    if(state.references)state.references.onchange=()=>state.viewer?.references(state.references.checked);
    const fitAll=card.querySelector('.cad-fit-all');if(fitAll)fitAll.onclick=()=>state.viewer?.fit({all:true});
    card.querySelector('.cad-fit').onclick=()=>state.viewer?.fit();
    state.reset.onclick=async()=>{stop(state);state.jointValues={};state.choice.value='0';await loadChoice(state);};
    state.play.onclick=()=>{
      if(state.playing){stop(state);return;}
      state.playing=true;state.play.textContent='Pause motion';
      const native=!!state.rig,start=performance.now();let lastPose=start;
      const tick=async time=>{
        if(!active||!state.visible||!state.playing)return;
        if(native&&state.rig){
          const fraction=(1-Math.cos((time-start)/3000*Number(state.speed.value)*Math.PI*2))/2;
          state.joints.querySelectorAll('input').forEach(input=>input.value=Number(input.min)+(Number(input.max)-Number(input.min))*fraction);pose(state);
        } else if(time-lastPose>1200/Number(state.speed.value)) {
          state.choice.value=String((Number(state.choice.value)+1)%option.cad.views.length);await loadChoice(state);lastPose=performance.now();
        }
        if(state.playing)state.frame=requestAnimationFrame(tick);
      };
      state.frame=requestAnimationFrame(tick);
    };
  });
  return {dispose(){active=false;observer.disconnect();states.forEach(state=>{stop(state);state.disposeBom?.();state.card.removeEventListener('cad-isolate-parts',state.isolateHandler);++state.generation;state.viewer?.dispose();});}};
}
