import {displayGroups, isReference, visibleDisplayGroups} from './cad-display.js?v=undeclared-overlays-20261113';
import {createJointRig} from './cad-motion.js?v=hip-controls-20261030';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {MeshoptDecoder} from 'three/addons/libs/meshopt_decoder.module.js';

export function createCadViewer(host, label, parts = [], {bodyPanelOpacity, hiddenReferenceParts = [], describe = null} = {}) {
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#e9ede6');
  const camera = new THREE.PerspectiveCamera(40, 1, .00001, 100); camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); host.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  controls.enablePan = true; controls.screenSpacePanning = true;
  let navigationMode = 'orbit';
  const navigationHint = () => navigationMode === 'pan' ? 'Drag to pan · scroll to zoom · click a part' : 'Drag to orbit · right-drag to pan · scroll to zoom · click a part';
  function navigation(mode) {
    navigationMode = mode === 'pan' ? 'pan' : 'orbit';
    controls.mouseButtons.LEFT = navigationMode === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    controls.touches.ONE = navigationMode === 'pan' ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
    renderer.domElement.style.cursor = navigationMode === 'pan' ? 'move' : 'grab';
    label.textContent = navigationHint();
  }
  navigation(navigationMode);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x9aa49b, 1.15); hemi.position.set(0, 0, 1); scene.add(hemi);
  for (const [x, y, z] of [
    [ 2.2, -2.6, 2.8],
    [-2.2, -2.6, 2.8],
    [-2.2,  2.6, 2.8],
    [ 2.2,  2.6, 2.8],
  ]) {
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.75);
    dirLight.position.set(x, y, z);
    scene.add(dirLight);
  }
  let active = true, model = null, generation = 0, showShells = true, showReferences = false, down = null;
  let exploded = 0, displayNodes = [], cutBounds = null, isolatedParts = null;
  const shownOverlays = {};
  let cut = {enabled:false, axis:'z', position:50, reverse:false};
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder), ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const metadata = new Map(parts.map(part => [THREE.PropertyBinding.sanitizeNodeName(part.name), part]));
  let highlightedMaterials = [], selectedPart = null;
  function highlight(name) {
    highlightedMaterials.forEach(({material, emissive, intensity}) => {material.emissive.copy(emissive); material.emissiveIntensity=intensity;});
    highlightedMaterials=[];
    model?.traverse(obj => {
      if (!obj.isMesh || !name || partInfo(obj)?.name !== name) return;
      (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(material => {
        if (!material.emissive) return;
        highlightedMaterials.push({material, emissive:material.emissive.clone(), intensity:material.emissiveIntensity});
        material.emissive.setHex(0x735400); material.emissiveIntensity=.7;
      });
    });
  }
  const hiddenReferences = new Set(hiddenReferenceParts.map(THREE.PropertyBinding.sanitizeNodeName));
  const release = m => m?.traverse(obj => {if (obj.isMesh) {obj.geometry?.dispose(); (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(mat => mat?.dispose());}});
  const resize = () => {const r = host.getBoundingClientRect(); if (r.width && r.height) {renderer.setSize(r.width, r.height); camera.aspect = r.width/r.height; camera.updateProjectionMatrix();}};
  const observer = new ResizeObserver(resize); observer.observe(host);
  function visibleBounds({all = false} = {}) {
    const bounds=new THREE.Box3();model.updateMatrixWorld(true);
    model.traverseVisible(obj=>{if(obj.isMesh&&obj.geometry && (all || !isReference(partInfo(obj)))){
      if(!obj.geometry.boundingBox)obj.geometry.computeBoundingBox();
      bounds.union(obj.geometry.boundingBox.clone().applyMatrix4(obj.matrixWorld));
    }});
    return bounds.isEmpty()?new THREE.Box3().setFromObject(model):bounds;
  }
  function fit(options = {}) {
    if (!model) return; resize();
    const box = visibleBounds(options), center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length()/2, .001);
    const half = Math.min(THREE.MathUtils.degToRad(camera.fov)/2, Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)*camera.aspect));
    const distance = radius*1.12/Math.sin(half); controls.target.copy(center);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(1, -1.2, .8).normalize(), distance);
    camera.near = Math.max(radius/1000, .000001); camera.far = Math.max(10, distance+radius*5); camera.updateProjectionMatrix(); controls.update();
  }
  function partInfo(obj) {
    while (obj && obj !== model) {if (metadata.has(obj.name)) return metadata.get(obj.name); obj = obj.parent;}
    return null;
  }
  function visibility() {
    model?.traverse(obj => {if (obj.isMesh) {
      const part=partInfo(obj);
      obj.visible=!(isReference(part) && hiddenReferences.has(THREE.PropertyBinding.sanitizeNodeName(part.name))) &&
        (isolatedParts ? isolatedParts.has(THREE.PropertyBinding.sanitizeNodeName(part?.name || obj.name)) :
        visibleDisplayGroups(part, {shells:showShells, references:showReferences, overlays:shownOverlays}));
    }});
  }
  function shells(visible) {
    showShells = visible;
    visibility();
  }
  function references(visible) {showReferences = visible; visibility();}
  function overlay(id, visible) {shownOverlays[id] = !!visible; visibility();}
  function isolate(names) {
    isolatedParts=names?.length ? new Set(names.map(THREE.PropertyBinding.sanitizeNodeName)) : null;
    visibility();
    label.textContent=isolatedParts ? `Isolated: ${names.join(', ')} · use Show assembly to restore parts` : navigationHint();
  }
  function undoExplosion() {
    displayNodes.forEach(item => {item.node.position.sub(item.offset); item.offset.set(0,0,0);});
    model?.updateMatrixWorld(true);
  }
  function applyExplosion() {
    if (!model || !exploded) return;
    const box = visibleBounds(), center = box.getCenter(new THREE.Vector3());
    const expansion = 2*exploded/100;
    // Diagnostic skin/frame meshes can represent one integrated printed part.
    // Keep those regions together during explosion, just as they are manufactured.
    const physicalParts = new Map();
    for (const item of displayNodes) {
      const key = metadata.get(item.node.name)?.physical_part_id || item;
      if (!physicalParts.has(key)) physicalParts.set(key, []);
      physicalParts.get(key).push(item);
    }
    for (const items of physicalParts.values()) {
      const physicalBounds = new THREE.Box3();
      for (const item of items) physicalBounds.union(new THREE.Box3().setFromObject(item.node));
      const partCenter = physicalBounds.getCenter(new THREE.Vector3());
      // Expand the spacing between physical part centres. A fixed normalized
      // offset moves coaxial parts by the same amount and leaves them touching.
      const displacement = partCenter.sub(center).multiplyScalar(expansion);
      for (const item of items) {
        const target = item.node.getWorldPosition(new THREE.Vector3()).add(displacement);
        const newPosition = item.node.parent.worldToLocal(target);
        item.offset.copy(newPosition).sub(item.node.position); item.node.position.copy(newPosition);
      }
    }
    model.updateMatrixWorld(true);
  }
  function explode(amount) {undoExplosion(); exploded = Number(amount); applyExplosion();}
  function section(value) {
    cut = {...cut, ...value};
    if (!cut.enabled || !cutBounds) {renderer.clippingPlanes = []; return;}
    const axis = new THREE.Vector3(cut.axis === 'x' ? 1 : 0, cut.axis === 'y' ? 1 : 0, cut.axis === 'z' ? 1 : 0);
    const position = THREE.MathUtils.lerp(cutBounds.min[cut.axis], cutBounds.max[cut.axis], Number(cut.position)/100);
    renderer.clippingPlanes = [new THREE.Plane(axis, -position)];
    if (cut.reverse) renderer.clippingPlanes[0].negate();
  }
  async function load(url, {preserveView = false} = {}) {
    const request = ++generation; label.textContent = 'Loading native CAD export…';
    const result = await loader.loadAsync(url);
    if (!active || request !== generation) {release(result.scene); return;}
    if (model) {highlight(null); selectedPart = null; scene.remove(model); release(model);}
    model = result.scene;
    host.dispatchEvent(new CustomEvent('cad-part-selected', {bubbles:true, detail:{name:null}}));
    model.traverse(obj => {if (obj.isMesh) {
      if (obj.geometry && !obj.geometry.attributes.normal) obj.geometry.computeVertexNormals();
      obj.material = Array.isArray(obj.material) ? obj.material.map(m => m.clone()) : obj.material.clone(); (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => {
      m.side = THREE.DoubleSide;
      const part=partInfo(obj);
      const opacity=displayGroups(part).includes('ape_body_shell') && Number.isFinite(bodyPanelOpacity) ? bodyPanelOpacity : part?.display_opacity;
      if (Number.isFinite(opacity)) {m.opacity=Math.max(.03,Math.min(1,opacity));m.transparent=m.opacity<1;m.depthWrite=!m.transparent;}
    });}});
    scene.add(model); model.updateMatrixWorld(true);
    cutBounds = new THREE.Box3().setFromObject(model); displayNodes = [];
    model.traverse(obj => {
      if (isReference(metadata.get(obj.name))) return;
      const named = metadata.has(obj.name), fallback = !metadata.size && (obj.isMesh || obj.children.some(child => child.isMesh));
      if (!(named || fallback)) return;
      let ancestor = obj.parent;
      while (ancestor && ancestor !== model) {if (metadata.has(ancestor.name) || (!metadata.size && ancestor.children.some(child => child.isMesh))) return; ancestor = ancestor.parent;}
      displayNodes.push({node:obj, offset:new THREE.Vector3()});
    });
    shells(showShells); applyExplosion(); section(cut); if (!preserveView) fit();
    label.textContent = navigationHint();
  }
  // Wheel, trackpad pinch and swipe over the viewer belong to the viewer only: never scroll,
  // zoom or swipe the page (OrbitControls skips preventDefault mid-drag; Safari pinch is a gesture event).
  const keep = event => event.preventDefault();
  renderer.domElement.style.overscrollBehavior = 'contain'; renderer.domElement.style.touchAction = 'none';
  renderer.domElement.addEventListener('wheel', keep, {passive:false});
  ['gesturestart','gesturechange','gestureend'].forEach(type => renderer.domElement.addEventListener(type, keep, {passive:false}));
  renderer.domElement.addEventListener('pointerdown', event => {down = [event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup', event => {
    if (!model || !down || Math.hypot(event.clientX-down[0], event.clientY-down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect(); pointer.set((event.clientX-r.left)/r.width*2-1, -(event.clientY-r.top)/r.height*2+1); ray.setFromCamera(pointer,camera);
    const hit = ray.intersectObject(model,true).find(hit => hit.object.visible && renderer.clippingPlanes.every(p => p.distanceToPoint(hit.point) >= 0));
    const p = hit ? partInfo(hit.object) : null;
    highlight(p?.name); selectedPart = p || null;
    host.dispatchEvent(new CustomEvent('cad-part-selected', {bubbles:true, detail:{name:p?.name || null}}));
    label.textContent = p ? partLabel(p) : hit ? `Part: ${hit.object.name || hit.object.parent?.name || 'unnamed mesh'}` : navigationHint();
  });
  // describe(part) (optional): the label of a selected part, e.g. its BOM line and assembly step from a body view.
  const partLabel = p => describe?.(p) || `${p.name} · ${p.role || 'part'}${p.material ? ' · '+p.material : ''}`;
  // Select a part by name (from a BOM row or a part view): highlight it and tell the panels, as a click would.
  function select(name) {
    const p = name ? metadata.get(THREE.PropertyBinding.sanitizeNodeName(name)) : null;
    highlight(p?.name || null); selectedPart = p || null;
    host.dispatchEvent(new CustomEvent('cad-part-selected', {bubbles:true, detail:{name:p?.name || null}}));
    label.textContent = p ? partLabel(p) : navigationHint();
  }
  // One rendered frame plus the frozen camera, for an annotated feedback snapshot.
  // pick(x, y) takes 0..1 canvas coordinates and returns the visible part there.
  function snapshot() {
    if (!model) return null;
    controls.update(); renderer.render(scene, camera);
    const image = renderer.domElement.toDataURL('image/png');
    const frozen = camera.clone(), planes = renderer.clippingPlanes.map(p => p.clone());
    const caster = new THREE.Raycaster(), at = new THREE.Vector2();
    function pick(x, y) {
      at.set(x*2-1, -y*2+1); caster.setFromCamera(at, frozen);
      const hit = caster.intersectObject(model, true).find(h => h.object.visible && planes.every(p => p.distanceToPoint(h.point) >= 0));
      if (!hit) return null;
      const part = partInfo(hit.object);
      return {name: part?.name || hit.object.name || hit.object.parent?.name || 'unnamed mesh', role: part?.role || null,
        material: part?.material || null, owning_link: part?.owning_link || null, point: hit.point.toArray()};
    }
    const direction = frozen.position.clone().sub(controls.target);
    return {image, width: renderer.domElement.width, height: renderer.domElement.height, pick,
      camera: {position: frozen.position.toArray(), target: controls.target.toArray(), up: frozen.up.toArray(),
        fov_deg: frozen.fov, distance: direction.length(), from_direction: direction.normalize().toArray()},
      display: {shells: showShells, references: showReferences, overlays: {...shownOverlays}, explode_percent: exploded,
        cut: {...cut}, isolated_parts: isolatedParts ? [...isolatedParts] : null},
      selected_part: selectedPart && {name: selectedPart.name, role: selectedPart.role || null, material: selectedPart.material || null, owning_link: selectedPart.owning_link || null}};
  }
  function tick() {if (!active) return; requestAnimationFrame(tick); if (host.offsetParent !== null) {controls.update(); renderer.render(scene,camera);}}
  resize(); tick();
  return {load, fit, shells, references, overlay, explode, section, isolate, select, navigation, snapshot,
    rig(data, options) {
      if (!model) return null; undoExplosion();
      const native = createJointRig(model, data, options); applyExplosion();
      return native && {...native, pose(values, frame) {undoExplosion(); native.pose(values, frame); applyExplosion();}};
    },
    dispose() {active = false; ++generation; observer.disconnect(); controls.dispose(); highlight(null); release(model); renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();}
  };
}
