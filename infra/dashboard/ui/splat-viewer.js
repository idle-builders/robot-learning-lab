import * as THREE from 'three';

// Pinned upstream Gaussian renderer; model bytes stay on the private local dashboard.
const library = 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js';

export async function createViewer(host, label) {
  const {Viewer, SceneFormat, SceneRevealMode} = await import(library);
  host.style.position = 'relative';
  let viewer = null, renderer = null, observer = null, disposed = false;

  async function clear() {
    observer?.disconnect(); observer = null;
    if (viewer) await viewer.dispose();
    viewer = null;
    renderer?.dispose();
    renderer?.domElement.remove();
    renderer = null;
  }

  async function load(url) {
    await clear();
    if (disposed) return;
    label.textContent = 'Loading Gaussian scene…';
    const configUrl = new URL(url, location.href);
    if (configUrl.searchParams.has('path')) {
      configUrl.searchParams.set('path', configUrl.searchParams.get('path').replace(/[^/]+$/, 'viewer-config.json'));
    } else {
      configUrl.pathname = configUrl.pathname.replace(/[^/]+$/, 'viewer-config.json');
    }
    const response = await fetch(configUrl);
    if (!response.ok) throw new Error('Saved camera configuration unavailable');
    const config = await response.json();
    if (disposed) return;
    // Keep a captured portrait view from expanding into unrecorded side rays.
    const aspect = config.image_size ? config.image_size[0] / config.image_size[1] : null;
    host.style.maxWidth = aspect ? `calc(55vh * ${aspect})` : '';
    host.style.aspectRatio = aspect ? String(aspect) : '';
    host.style.height = aspect ? 'auto' : '';
    host.style.marginInline = aspect ? 'auto' : '';
    const camera = new THREE.PerspectiveCamera(config.fov_deg || 45, host.clientWidth / host.clientHeight, .001, 10000);
    camera.position.fromArray(config.position);
    camera.up.fromArray(config.up).normalize();
    camera.lookAt(new THREE.Vector3(...config.look_at));
    renderer = new THREE.WebGLRenderer({antialias: false});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer.domElement);
    viewer = new Viewer({rootElement: host, renderer, camera, selfDrivenMode: true,
      useBuiltInControls: true, initialCameraLookAt: config.look_at,
      sharedMemoryForWorkers: false, gpuAcceleratedSort: false,
      sphericalHarmonicsDegree: 2, integerBasedSort: false,
      sceneRevealMode: SceneRevealMode.Instant, halfPrecisionCovariancesOnGPU: true});
    // The upstream position refresh distance is one world unit, which is too
    // coarse for these metre-scale room scans. Coalesce control changes while
    // a sort runs, then fully sort the latest camera before the next settled view.
    let cameraSortRequested = true, lastSortedCamera = [];
    const upstreamSort = viewer.runSplatSort.bind(viewer);
    viewer.runSplatSort = (force = false, all = false) => {
      const state = [...camera.position.toArray(), ...camera.quaternion.toArray(), camera.fov, camera.aspect];
      if (state.some((value, index) => value !== lastSortedCamera[index])) cameraSortRequested = true;
      if (cameraSortRequested && !viewer.sortRunning) {
        cameraSortRequested = false;
        lastSortedCamera = state;
        camera.updateMatrixWorld();
        return upstreamSort(true, true);
      }
      return upstreamSort(force, all);
    };
    observer = new ResizeObserver(() => {
      if (!renderer || !host.clientWidth || !host.clientHeight) return;
      renderer.setSize(host.clientWidth, host.clientHeight);
      camera.aspect = host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
    });
    observer.observe(host);
    const sourcePath = new URL(url, location.href).searchParams.get('path') || new URL(url, location.href).pathname;
    const format = sourcePath.endsWith('.ksplat') ? SceneFormat.KSplat : sourcePath.endsWith('.splat') ? SceneFormat.Splat : SceneFormat.Ply;
    await viewer.addSplatScene(url, {format, progressiveLoad: false, showLoadingUI: false});
    if (disposed) return;
    viewer.start();
    label.textContent = 'Drag to orbit · right-drag to move · wheel to zoom. Distances are uncalibrated. Browser preview uses SH degree 2; saved CUDA renders use degree 3.';
  }

  function dispose() {disposed = true; return clear();}
  return {load, dispose};
}
