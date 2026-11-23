"""Close-up of the shin back cover's side mounts (roll-flip, owner 2026-11-18; thigh_shell.py shin-cover).

Medial view of the cut cover on the shin frame, the section at the upper screw height, and a zoom of the medial screw.

  .venv/bin/python -B robot/body/hip-shell/sections/shin_cover_mounts_view.py
Writes drawing-sheets/roll-flip/shin-cover-mounts.png (derived; recreate with this command).
"""
from pathlib import Path
import json, sys, numpy as np, trimesh, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=str(Path(__file__).resolve().parents[4])+'/'
m=json.load(open(ROOT+'artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json'))
E={p['name']:p for p in m['parts']}
def L(n):
    f=ROOT+E[n]['mesh']
    if f.endswith('.npz'): d=np.load(f); return trimesh.Trimesh(d['vertices'],d['faces'],process=True)
    return trimesh.load(f)
C={'R_shin_back_hard_cover':'#b4b4b4','R_shin_combined':'#6f8fb0','R_ankle_pitch_ST3025_case':'#3a3a3a','R_shin_front_hard_cover':'#d8d8d8'}
C.update({f'R_shin_back_cover_PA2x8_{i}':'#e0a020' for i in range(4)})
M={n:L(n) for n in C}
light=np.array([0.5,0.4,0.75]); light/=np.linalg.norm(light)
fig,axs=plt.subplots(1,3,figsize=(21,8))
def shade(ax,w,h,v,hs,names,title):
    polys,cols,dep=[],[],[]
    for n in names:
        mm=M[n]; tri=mm.triangles; nr=mm.face_normals; idx=np.nonzero(nr@w>0)[0]
        P=tri[idx][:,:,[h,v]].copy(); P[:,:,0]*=hs
        rgb=np.array(matplotlib.colors.to_rgb(C[n])); sh=0.35+0.65*np.clip(nr[idx]@light,0,1)
        polys+=list(P); cols+=list(np.clip(rgb[None]*sh[:,None],0,1)); dep+=list(tri[idx].mean(1)@w)
    o=np.argsort(dep)
    ax.add_collection(PolyCollection([polys[i] for i in o],facecolors=[cols[i] for i in o],edgecolors='none'))
    ax.autoscale(); ax.set_aspect('equal'); ax.set_title(title,fontsize=11)
shade(axs[0],np.array([-1,0,0.]),1,2,-1,['R_shin_combined','R_shin_back_hard_cover']+[f'R_shin_back_cover_PA2x8_{i}' for i in range(4)],
      'Medial side (from inside the legs): grey back cover, 2 screw heads in counterbores')
axs[0].set_xlabel('y (mm, back to the left)'); axs[0].set_ylabel('z (mm)')
# section at the upper screw height
z0=81.0
for n,c in C.items():
    s=M[n].section(plane_origin=[0,0,z0],plane_normal=[0,0,1])
    if s is None: continue
    for d in s.discrete: axs[1].fill(d[:,0],d[:,1],color=c,alpha=0.85,lw=0.3,ec='k')
axs[1].set_aspect('equal'); axs[1].set_title(f'Section at z {z0} (top view): cover wall + boss -> PA2x8 -> frame side wall; ankle motor (dark) inside',fontsize=11)
axs[1].set_xlabel('x (mm)'); axs[1].set_ylabel('y (mm, back up)'); axs[1].set_ylim(0,32)
# zoom of the medial section
for n,c in C.items():
    s=M[n].section(plane_origin=[0,0,z0],plane_normal=[0,0,1])
    if s is None: continue
    for d in s.discrete: axs[2].fill(d[:,0],d[:,1],color=c,alpha=0.85,lw=0.4,ec='k')
axs[2].set_xlim(14,34); axs[2].set_ylim(5,20); axs[2].set_aspect('equal'); axs[2].grid(alpha=0.3)
axs[2].set_title('Zoom, medial screw: counterbore (oblique entry chamfer) in the slanted skin, boss seated on the frame wall,\nPA2x8 into a dia 1.7 pilot in the frame side wall, ankle motor beyond',fontsize=10)
fig.suptitle('Shin back cover: new side mounts (right leg; left mirrors). Blue-grey: shin frame; light grey: back cover; gold: PA2x8 screws',fontsize=13)
out=Path(__file__).resolve().parent/'drawing-sheets/roll-flip/shin-cover-mounts.png'; out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out,dpi=90,bbox_inches='tight'); print(out)
