// Programmatic RGB observation adapter. No presentation page or gallery is saved.
// NODE_PATH must resolve the locally installed playwright package.
const {chromium} = require('playwright');
const readline = require('readline');

(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox']});
  const page = await browser.newPage({viewport: {width: 640, height: 360}, deviceScaleFactor: 1});
  page.on('pageerror', error => process.stderr.write(error.message+'\n'));
  const lines = readline.createInterface({input: process.stdin});
  try {
    for await (const line of lines) {
      const command = JSON.parse(line);
      if (command.op === 'close') break;
      if (command.op === 'init') {
        command.width=command.width||640;command.height=command.height||360;
        await page.setViewportSize({width:command.width,height:command.height});
        await page.goto(command.dashboard);
        await page.evaluate(async command => {
          const THREE = await import('three');
          const GS = await import('https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js');
          document.body.replaceChildren();document.body.style.margin='0';
          const {width,height}=command;
          const host = document.createElement('div');host.style.width=width+'px';host.style.height=height+'px';
          document.body.append(host);
          const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
          renderer.setSize(width,height);host.append(renderer.domElement);
          const camera = new THREE.PerspectiveCamera(45,width/height,.001,10000);
          const viewer = new GS.Viewer({rootElement:host, renderer, camera, selfDrivenMode:false,
            useBuiltInControls:false, sharedMemoryForWorkers:false, gpuAcceleratedSort:false,
            sphericalHarmonicsDegree:2, integerBasedSort:false, sceneRevealMode:GS.SceneRevealMode.Instant});
          await viewer.addSplatScene(command.model_url,{format:GS.SceneFormat.Ply,showLoadingUI:false,progressiveLoad:false});
          window.observer = {THREE,renderer,camera,viewer,width,height};
        }, command);
        process.stdout.write(JSON.stringify({ready:true,renderer:'GaussianSplats3D 0.4.7 / SH2 / native WebGL',width:command.width,height:command.height})+'\n');
      } else if (command.op === 'frame') {
        const response = await page.evaluate(async command => {
          const {THREE,renderer,camera,viewer,width,height} = window.observer;
          const m=command.c2w, k=command.K;
          if(viewer.sortRunning && viewer.sortPromise) await viewer.sortPromise;
          viewer.updateForRendererSizeChanges();
          camera.fov=2*Math.atan(height/(2*k[1][1]))*180/Math.PI;
          camera.aspect=width/height;
          camera.position.set(m[0][3],m[1][3],m[2][3]);
          camera.up.set(-m[0][1],-m[1][1],-m[2][1]).normalize();
          camera.lookAt(new THREE.Vector3(m[0][3]+m[0][2],m[1][3]+m[1][2],m[2][3]+m[2][2]));
          // Full calibrated pinhole projection, including principal point.
          camera.projectionMatrix.set(2*k[0][0]/width,0,1-2*k[0][2]/width,0,
            0,2*k[1][1]/height,2*k[1][2]/height-1,0,
            0,0,-(10000+.001)/(10000-.001),-2*10000*.001/(10000-.001),0,0,-1,0);
          camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
          camera.updateMatrixWorld();
          // Interactive partial sorts can span several animation frames. A
          // requested sensor observation must finish a full sort for this pose.
          await viewer.runSplatSort(true,true);
          if(viewer.sortPromise) await viewer.sortPromise;
          viewer.updateSplatMesh();viewer.timingSensitiveUpdates();
          viewer.render();
          return {png_base64:renderer.domElement.toDataURL('image/png').split(',')[1],
            splats_sorted:viewer.lastSplatSortCount,splats_visible:viewer.splatRenderCount};
        }, command);
        process.stdout.write(JSON.stringify(response)+'\n');
      }
    }
  } finally { lines.close();process.stdin.pause();await browser.close(); }
  process.exit(0);
})().catch(error => {process.stderr.write(error.stack+'\n');process.exit(1)});
