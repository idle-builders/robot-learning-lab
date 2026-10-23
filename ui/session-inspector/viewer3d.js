import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

export async function createViewer(host, label) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#1d2822');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10000);
  camera.up.set(0,0,1);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x5d7163, 2.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(2,-3,5); scene.add(key);
  let model = null, box = null, chosen = null;
  const ray = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const loader = new GLTFLoader();
  const resize = () => { const {width,height} = host.getBoundingClientRect(); if (width && height) { renderer.setSize(width,height); camera.aspect = width/height; camera.updateProjectionMatrix(); } };
  const observer = new ResizeObserver(resize); observer.observe(host);
  let down = null;
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX,e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!model || !down || Math.hypot(e.clientX-down[0],e.clientY-down[1]) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((e.clientX-rect.left)/rect.width*2-1, -(e.clientY-rect.top)/rect.height*2+1);
    ray.setFromCamera(pointer,camera);
    const hit = ray.intersectObject(model,true)[0];
    if (chosen) chosen.material?.emissive?.setHex(0x000000);
    chosen = hit?.object || null;
    if (chosen?.material?.emissive) chosen.material.emissive.setHex(0x194a25);
    label.textContent = chosen ? `Part: ${chosen.name || chosen.parent?.name || 'mesh'}` : 'Drag to orbit · scroll to zoom · click a part';
  });
  function fit() {
    if (!box) return;
    resize();
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() * .5, .001);
    const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
    const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * camera.aspect);
    // Fit the bounding sphere inside both frustum dimensions, including narrow views.
    const distance = radius * 1.05 / Math.sin(Math.min(verticalHalfFov, horizontalHalfFov));
    const direction = new THREE.Vector3(1, -1.15, .75).normalize();
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.near = Math.max(radius / 1000, .00001);
    camera.far = Math.max(distance + radius * 4, 10);
    camera.updateProjectionMatrix(); controls.update();
  }
  async function load(url) {
    if (model) { scene.remove(model); model.traverse(obj => { if (obj.isMesh) { obj.geometry?.dispose(); if (Array.isArray(obj.material)) obj.material.forEach(m=>m.dispose()); else obj.material?.dispose(); } }); model = null; }
    chosen = null;
    const result = await loader.loadAsync(url);
    model = result.scene;
    model.traverse(obj => { if (obj.isMesh) { obj.material = obj.material.clone(); obj.material.side = THREE.DoubleSide; } });
    scene.add(model); box = new THREE.Box3().setFromObject(model); fit();
    label.textContent = 'Drag to orbit · scroll to zoom · click a part';
  }
  let active = true;
  function tick() { if (!active) return; requestAnimationFrame(tick); if (host.offsetParent !== null) { controls.update(); renderer.render(scene,camera); } }
  resize(); tick();
  function dispose() {
    active = false; observer.disconnect(); controls.dispose();
    if (model) model.traverse(obj => { if (obj.isMesh) { obj.geometry?.dispose(); if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose()); else obj.material?.dispose(); } });
    renderer.dispose(); renderer.domElement.remove();
  }
  return {load,fit,dispose};
}
