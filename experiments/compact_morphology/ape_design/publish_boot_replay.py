"""Publish the five-seed CAD boot transfer comparison as a local review page."""

from __future__ import annotations

from html import escape
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-full-boot-control-20261002"
PRIOR = ROOT / "artifacts/ape-sole-policy-replay-20261002"
CASES = (
    ("baseline", "Original training foot", "Original 90 × 60 × 24 mm box", PRIOR),
    ("boot_geometry_locked", "Full CAD boot · same mass", "CAD collision hull; original foot inertia", OUT),
    ("boot_cad_mass", "Full CAD boot · estimated mass", "CAD collision hull; source-based foot mass", OUT),
    ("boot_trained150", "Learned boot walker · 150 updates", "Warm-started upstream PPO; 4,096 parallel worlds", OUT),
    ("boot_trained299", "Learned boot walker · 299 updates", "Selected checkpoint; fresh closed-loop replay", OUT),
)


def row(key: str, title: str, detail: str, folder: Path) -> str:
    reports = [json.loads((folder / "runs" / f"{key}-{seed}.json").read_text())
               for seed in range(142, 147)]
    worlds = [report["worlds"][0] for report in reports]
    survived = sum(w["ended_by_timeout"] for w in worlds)
    crossed = sum(w["first_meter"] is not None for w in worlds)
    straight = sum(w["ended_by_timeout"] and w["forward_m"] >= 1
                   and abs(w["lateral_m"]) <= 0.5 and abs(w["final_yaw_rad"]) <= 0.5
                   and w["hand_force_n"] < 1 for w in worlds)
    stats = "".join(
        f"<tr><td>{seed}</td><td>{w['duration_s']:.2f}</td>"
        f"<td>{w['forward_m']:.2f}</td><td>{w['lateral_m']:+.2f}</td>"
        f"<td>{w['final_yaw_rad']:+.2f}</td>"
        f"<td>{sum(w['contact_slip_m_left_right']):.2f}</td>"
        f"<td>{w['hand_force_n']:.1f}</td></tr>"
        for seed, w in zip(range(142, 147), worlds)
    )
    video = os.path.relpath(folder / "runs" / f"{key}-142.mp4", OUT)
    return f"""
      <section class="case"><div class="casehead"><div><h2>{escape(title)}</h2><small>{escape(detail)}</small></div>
      <div class="tally"><b>{survived}/5</b> finish 8 s <b>{crossed}/5</b> cross 1 m
      <b>{straight}/5</b> straight and controlled</div></div>
      <video src="{video}" controls muted playsinline preload="metadata"></video>
      <table><thead><tr><th>Seed</th><th>Time s</th><th>Forward m</th><th>Side m</th>
      <th>Yaw rad</th><th>Slip m</th><th>Hand N</th></tr></thead><tbody>{stats}</tbody></table></section>"""


def main() -> None:
    model = json.loads((OUT / "model.json").read_text())
    cards = "".join(row(*case) for case in CASES)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CAD boot gait transfer</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f1ec;color:#25342f;font:15px/1.42 system-ui,sans-serif}}
header{{padding:28px max(24px,5vw);background:#263b34;color:#fff}}h1{{margin:0;font-size:clamp(28px,3.2vw,42px)}}
header p{{max-width:920px;margin:8px 0 0;color:#e2e8dd}}main{{padding:26px max(24px,5vw) 65px;max-width:1550px;margin:auto}}
.hero{{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.8fr);gap:22px;margin-bottom:24px}}
#cad{{min-height:410px;border-radius:16px;background:#e5dfd5;overflow:hidden;position:relative}}
#cad canvas{{display:block;width:100%;height:100%}}.overlay{{position:absolute;left:16px;top:16px;display:flex;gap:8px;flex-wrap:wrap}}
button{{border:1px solid #b3c3b8;background:#fcfcf7;color:#243c31;border-radius:8px;padding:8px 11px;cursor:pointer}}
.note{{background:#fff;border:1px solid #d7ded6;border-radius:16px;padding:20px}}.note p{{margin:9px 0}}.case{{background:#fff;border:1px solid #d7ded6;border-radius:16px;overflow:hidden;margin:20px 0}}
.casehead{{padding:16px 20px;display:flex;justify-content:space-between;gap:20px;align-items:center}}h2{{font-size:21px;margin:0}}small{{color:#63786d}}
.tally{{font-size:13px;display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;color:#475f52}}.tally b{{font-size:22px;color:#266148}}
video{{display:block;width:100%;max-height:560px;background:#192627}}table{{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13px}}td,th{{padding:5px 14px;text-align:right;border-bottom:1px solid #edf0ea}}th{{color:#52645c}}
a{{color:#22654d}}code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:900px){{.hero{{grid-template-columns:1fr}}.casehead{{display:block}}table{{font-size:11px}}td,th{{padding:5px}}}}
</style><script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.171.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.171.0/examples/jsm/"}}}}</script></head>
<body><header><h1>Can the learned ape walk on its CAD boot?</h1><p>Yes, after one warm-started PPO run: the selected checkpoint finishes eight seconds and crosses one metre in all five fresh seeds; four stay within the stricter straight-walk bounds. The first three rows isolate geometry and mass with the original policy, then the last two show training progress on the CAD boot. All use the same 0.35 m/s command and 2 ms physics.</p></header>
<main><div class="hero"><div id="cad"><div class="overlay"><button id="contact">Show contact hull</button><button id="front">Front</button><button id="three">Boot close-up</button><button id="full">Full ape</button></div></div>
<div class="note"><h2>What changed</h2><p>The ivory boot is the approved rounded shell. The coral overlay is the actual contact mesh used by the gym; it is the convex hull of the shell and a 90 × 60 × 2.5 mm outsole. Its volume is <b>{model['contact_hull_extra_volume_fraction']*100:.1f}%</b> above the exact CAD union, so the collision is still a measured approximation.</p>
<p>CAD-derived foot mass: <b>{model['cad_foot_mass_kg']*1000:.0f} g</b> versus <b>{model['retained_foot_mass_kg']*1000:.0f} g</b> in the walking reference. The estimate uses a 2 mm PLA shell, TPU outsole and a 74.5 g motor case. Shaft, mount, wiring, tread compliance and friction remain unmeasured.</p>
<p>Independent visual review: the rounded shell hides the motor, but the current wide rectangular outsole reads like a platform rather than the sketch's continuous paw. That shape needs a separate design pass after the gait transfer is understood.</p>
<p><strong>Physical caveat:</strong> the selected walker reaches a leg torque clamp in about 96% of frames in these replays. The servo response and continuous-duty headroom remain hypotheses. The result establishes control transfer in this nominal gym, not a build-ready gait.</p>
<p><a href="model.json">Mass, sources and hashes</a> · <a href="models/boot_cad_mass.xml">Gym MJCF</a> · <a href="../ape-sketch-shell-v2-20261002/index.html?view=three">Full assembled CAD viewer</a></p>
<p><a href="checkpoints/model_299.pt">Selected policy</a> · <a href="training/params/env.yaml">Resolved gym config</a> · <a href="training/params/agent.yaml">PPO config</a> · <a href="pose-audit.json">Stand/step/squat fit screen</a> · <a href="evidence.json">All evidence hashes</a></p></div></div>
<p>The “straight and controlled” count requires eight seconds alive, at least one metre forward, no more than 0.5 m sideways and 0.5 rad yaw, and no significant hand-ground force. It is a review criterion, not a new training reward. Contact slip is cumulative across both feet.</p>{cards}
<p>Original checkpoint SHA-256 <code>39e316e0346cb538692832aaf300b451f2dec0f101f2cca1c7542d2ffe9f7fda</code> · selected boot checkpoint SHA-256 <code>685c0e2c2d5287117d09707397a5afe59acc635a7a529403ac247d6fa75f1cbd</code>. Videos for the trained policy and CAD-mass pretrain are rendered from the saved no-video simulation qpos, so playback does not perturb the measured rollout. An additional video-enabled replay diverged despite the same initial-state and first-action hashes; it is preserved under <a href="diagnostics/live-render-142.json">diagnostics</a>. All model XML/STL, five-seed JSON, trajectories, training config and logs are preserved locally; no 10 GB base-model weights were downloaded.</p></main>
<script type="module">
import * as THREE from 'three';import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';import {{STLLoader}} from 'three/addons/loaders/STLLoader.js';
const host=document.getElementById('cad'),renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;host.prepend(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe5dfd5);scene.add(new THREE.AmbientLight(0xffffff,.85));
for(const [p,n] of [[[-.5,-.7,.8],1.8],[[.6,.5,.8],.65]]){{const l=new THREE.DirectionalLight(0xffffff,n);l.position.set(...p);scene.add(l)}}
const camera=new THREE.PerspectiveCamera(29,1,.01,10);camera.up.set(0,0,1);camera.position.set(.27,-.28,.20);
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.target.set(0,0,.075);
const root=(await new GLTFLoader().loadAsync('../ape-sketch-shell-v2-20261002/sketch-06-shell.glb')).scene;root.scale.setScalar(.001);scene.add(root);
const palette={{porcelain:0xe9e3d5,warm:0xede7d9,face:0xf0d6b4,shadow:0x625e57,eye:0x232727,glint:0xffffff,accent:0xe5764b}};
const manifest=await fetch('../ape-sketch-shell-v2-20261002/manifest.json').then(r=>r.json());const names=new Map(manifest.parts.map(p=>[p.name,p.color]));
root.traverse(o=>{{if(o.isMesh){{o.material=new THREE.MeshStandardMaterial({{color:palette[names.get(o.name)]||0xe9e3d5,roughness:.82,metalness:0,side:THREE.DoubleSide}})}}}});
const geom=await new STLLoader().loadAsync('models/boot_contact_hull.stl');geom.computeVertexNormals();const contacts=[];
for(const side of [-1,1]){{const mesh=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({{color:0xea7650,transparent:true,opacity:.42,depthWrite:false,side:THREE.DoubleSide}}));mesh.position.set(0,side*.041,.032);mesh.visible=false;scene.add(mesh);contacts.push(mesh)}}
document.getElementById('contact').onclick=()=>{{const on=!contacts[0].visible;contacts.forEach(m=>m.visible=on);document.getElementById('contact').textContent=on?'Hide contact hull':'Show contact hull'}};
document.getElementById('front').onclick=()=>{{camera.position.set(.37,0,.14);orbit.target.set(0,0,.075);orbit.update()}};
document.getElementById('three').onclick=()=>{{camera.position.set(.27,-.28,.20);orbit.target.set(0,0,.075);orbit.update()}};
document.getElementById('full').onclick=()=>{{camera.position.set(.67,-.54,.36);orbit.target.set(0,0,.17);orbit.update()}};
function resize(){{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}}
new ResizeObserver(resize).observe(host);resize();(function frame(){{requestAnimationFrame(frame);orbit.update();renderer.render(scene,camera)}})();
</script></body></html>"""
    (OUT / "index.html").write_text(html)
    source_paths = [
        ROOT / "experiments/compact_morphology/ape_design" / name
        for name in ("sketch_shell.py", "make_boot_model.py", "audit_boot_poses.py",
                     "run_boot_replay.sh", "run_trained_boot_replay.sh", "publish_boot_replay.py")
    ] + [ROOT / "experiments/compact_morphology/mjlab_ape" / name
         for name in ("train_boot.py", "prepare_boot_warmstart.py")]
    source_paths.append(ROOT / "experiments/compact_morphology/keyframes/render_ape_qpos.py")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        "pollen_commit": "0bf989701f14d4008c866ffd6f09067870f99379",
        "runpod_pod_id": "nesm8u90qamduj",
        "training": "300 upstream PPO updates; 4096 worlds x 24 steps; 2 ms physics; 50 Hz action; warm-started 20-to-22 joint mapping",
        "selected_checkpoint": "checkpoints/model_299.pt",
        "selected_checkpoint_sha256": digest(OUT / "checkpoints/model_299.pt"),
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "artifact_files": {str(path.relative_to(OUT)): digest(path)
                           for path in sorted(OUT.rglob("*"))
                           if path.is_file() and path.name != "evidence.json"},
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
