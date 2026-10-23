import {displayGroups, isReference, visibleDisplayGroups} from './cad-display.js?v=body-panels-20261021';
import {createJointRig} from './cad-motion.js?v=studies-final-20261020';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

export function createCadViewer(host, label, parts = [], {bodyPanelOpacity, hiddenReferenceParts = []} = {}) {
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
  scene.add(new THREE.HemisphereLight(0xffffff, 0x768475, 2.8));
  const light = new THREE.DirectionalLight(0xffffff, 2.7); light.position.set(2, -3, 4); scene.add(light);
  let active = true, model = null, generation = 0, showShells = true, showReferences = false, down = null;
  let exploded = 0, displayNodes = [], cutBounds = null, isolatedParts = null;
  let cut = {enabled:false, axis:'z', position:50, reverse:false};
  const loader = new GLTFLoader(), ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const metadata = new Map(parts.map(part => [THREE.PropertyBinding.sanitizeNodeName(part.name), part]));
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
        visibleDisplayGroups(part, {shells:showShells, references:showReferences}));
    }});
  }
  function shells(visible) {
    showShells = visible;
    visibility();
  }
  function references(visible) {showReferences = visible; visibility();}
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
    if (model) {scene.remove(model); release(model);}
    model = result.scene;
    model.traverse(obj => {if (obj.isMesh) {obj.material = Array.isArray(obj.material) ? obj.material.map(m => m.clone()) : obj.material.clone(); (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => {
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
  renderer.domElement.addEventListener('pointerdown', event => {down = [event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup', event => {
    if (!model || !down || Math.hypot(event.clientX-down[0], event.clientY-down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect(); pointer.set((event.clientX-r.left)/r.width*2-1, -(event.clientY-r.top)/r.height*2+1); ray.setFromCamera(pointer,camera);
    const hit = ray.intersectObject(model,true).find(hit => hit.object.visible && renderer.clippingPlanes.every(p => p.distanceToPoint(hit.point) >= 0)); if (!hit) return;
    const p = partInfo(hit.object);
    if (p) host.dispatchEvent(new CustomEvent('cad-part-selected', {bubbles:true, detail:{name:p.name}}));
    label.textContent = p ? `${p.name} · ${p.role || 'part'}${p.material ? ' · '+p.material : ''}` : `Part: ${hit.object.name || hit.object.parent?.name || 'unnamed mesh'}`;
  });
  function tick() {if (!active) return; requestAnimationFrame(tick); if (host.offsetParent !== null) {controls.update(); renderer.render(scene,camera);}}
  resize(); tick();
  return {load, fit, shells, references, explode, section, isolate, navigation,
    rig(data, options) {
      if (!model) return null; undoExplosion();
      const native = createJointRig(model, data, options); applyExplosion();
      return native && {...native, pose(values, frame) {undoExplosion(); native.pose(values, frame); applyExplosion();}};
    },
    dispose() {active = false; ++generation; observer.disconnect(); controls.dispose(); release(model); renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();}
  };
}
