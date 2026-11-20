"""Render obstacle-world evaluation traces: robot (solid) and its reference (translucent ghost) on chosen worlds and frames.
usage: render_world.py EVAL_DIR SEGMENT OUT.png --worlds 2 12 --frames 0 50 100"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'control/wbc'))
import parcour, tracking
from mjlab.terrains.terrain_generator import TerrainGenerator
p = argparse.ArgumentParser()
p.add_argument('eval'); p.add_argument('segment'); p.add_argument('out')
p.add_argument('--worlds', type=int, nargs='+'); p.add_argument('--frames', type=int, nargs='+'); p.add_argument('--size', type=int, default=480)
a = p.parse_args()
d = Path(a.eval); rep = json.loads((d / 'report.json').read_text())
t = np.load(d / f'{a.segment}-nominal-teacher.npz')
spec = mujoco.MjSpec()
spec.visual.global_.offwidth = spec.visual.global_.offheight = a.size
TerrainGenerator(parcour.PARCOUR_TERRAIN).compile(spec)
def body():
    r = mujoco.MjSpec.from_file(str(tracking.MODEL / 'cove50-rigid.xml'))
    return tracking.add_fists(tracking.scale_arms(tracking.scale_feet(r, (1.3, 1.3)), (1.6, 1.27)), .03, .01)
spec.worldbody.add_frame().attach_body(body().worldbody.first_body(), 'robot/', '')
ghost = body()
for g in ghost.geoms:
    g.rgba = (.2, .6, 1., .35); g.contype = g.conaffinity = 0
spec.worldbody.add_frame().attach_body(ghost.worldbody.first_body(), 'ghost/', '')
spec.worldbody.add_light(pos=(0, 0, 12), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(.6, .6, .6))
model = spec.compile(); data = mujoco.MjData(model)
model.vis.headlight.ambient[:] = (.45, .45, .45); model.vis.map.zfar = 200.
nq = 7 + 22
tiles = []
with mujoco.Renderer(model, a.size, a.size) as r:
    for w in a.worlds:
        for f in a.frames:
            q = t['qpos'][f, w]
            ghost_q = np.concatenate([t['ref_anchor_pos'][f, w], t['ref_anchor_quat'][f, w], t['ref_joint_pos'][f, w]])
            data.qpos[:nq], data.qpos[nq:2 * nq] = q, ghost_q
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera(); cam.lookat[:] = q[:3]; cam.distance, cam.azimuth, cam.elevation = 1.1, 90., -8.
            r.update_scene(data, cam); im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((8, 8), f"world {w} {rep['worlds'][w]['tile']} d={rep['worlds'][w]['difficulty']} t={f/50:.1f}s", fill='white')
            tiles.append(im)
cols = len(a.frames); sheet = Image.new('RGB', (cols * a.size, len(a.worlds) * a.size))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % cols) * a.size, (i // cols) * a.size))
sheet.save(a.out); print(a.out)
