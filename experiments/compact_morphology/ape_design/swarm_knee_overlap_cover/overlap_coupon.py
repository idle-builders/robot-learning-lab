"""Bounded tongue + rear-gore trial for the 50 cm offset knee.

The existing c48e952 cuff is the fixed 2 mm lip. A 2 mm moving tongue is
rooted inside the source shin and extends toward the displaced pitch axis.
The rest of the circumference is spanned by a separate folded fabric arc.
The meshes are screening surfaces: any point inside an exact hard B-rep is a
definite rejection; a zero count would still need continuous collision work.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import numpy as np
import trimesh
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_knee_flex_closure.closure_coupon import (
    AXIS, hard_parts, rotate,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import turn
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

ANGLES = (0, 60, 115)
TONGUE_START_DEG = -145
TONGUE_END_DEG = 65
TONGUE_TIP_Z = 148.0
TONGUE_ROOT_Z = 111.0
TONGUE_THICKNESS = 2.0
GORE_PLEATS = 8
GORE_MATERIAL_THICKNESS = 0.229  # nominal PU/nylon 9UN, before stiffeners


def strip_mesh(rings, color):
    rings = np.asarray(rings)
    nr, nc, _ = rings.shape
    faces = []
    for j in range(nr-1):
        for i in range(nc-1):
            a, c = j*nc+i, j*nc+i+1
            d, e = (j+1)*nc+i, (j+1)*nc+i+1
            faces.extend(((a,c,d),(c,e,d)))
    tri = trimesh.Trimesh(vertices=rings.reshape((-1,3)), faces=faces,
                          process=False)
    tri.visual.vertex_colors = color
    return tri


def tongue_local():
    phi = np.deg2rad(np.linspace(TONGUE_START_DEG, TONGUE_END_DEG, 85))
    z = np.linspace(TONGUE_ROOT_Z, TONGUE_TIP_Z, 38)
    rings = []
    for zz in z:
        if zz <= 118.07:
            t = (zz-111)/(118.07-111)
            cx = 60.94+(60.0-60.94)*t
            cy = 5.18+(4.286-5.18)*t
            rx = 23.7+(20.86-23.7)*t
            ry = 25.06+(22.3-25.06)*t
        else:
            cx,cy = 60.,4.286
            rx = 20.86-0.18*(zz-118.07)
            ry = 22.3-0.18*(zz-118.07)
        rings.append(np.stack((cx+rx*np.cos(phi),
                               cy+ry*np.sin(phi),
                               np.full_like(phi,zz)),axis=1))
    return np.asarray(rings)


def tongue_mesh(angle):
    outer = rotate(tongue_local().reshape((-1,3)), angle).reshape((38,85,3))
    # The inner surface is a 2 mm radial inset from the outer sheet.
    local = tongue_local()
    phi = np.deg2rad(np.linspace(TONGUE_START_DEG,TONGUE_END_DEG,85))
    normal = np.stack((np.cos(phi),np.sin(phi),np.zeros_like(phi)),axis=1)
    inner_local = local - TONGUE_THICKNESS*normal[None,:,:]
    inner = rotate(inner_local.reshape((-1,3)),angle).reshape((38,85,3))
    out = strip_mesh(outer,(227,218,200,255))
    back = strip_mesh(inner[::-1],(227,218,200,255))
    # The full 2 mm ribbon is exported as one mesh for opaque inspection.
    tri = trimesh.util.concatenate((out,back))
    return tri, outer, inner


def rear_loops(phi, angle):
    fixed = np.stack((60+18.5*np.cos(phi), 4.286+21.2*np.sin(phi),
                      np.full_like(phi,125.8)),axis=1)
    moving_local = np.stack((60.94+23.7*np.cos(phi),
                             5.18+25.06*np.sin(phi),
                             np.full_like(phi,111.0)),axis=1)
    moving = rotate(moving_local,angle)
    fixed_n = np.stack((np.cos(phi),np.sin(phi),np.zeros_like(phi)),axis=1)
    moving_n = rotate(moving_local+fixed_n,angle)-moving
    return fixed,moving,fixed_n,moving_n


def gore_mesh(angle, phi, stock):
    fixed,moving,fixed_n,moving_n = rear_loops(phi,angle)
    d = np.linalg.norm(moving-fixed,axis=1)
    amplitude = np.sqrt(np.maximum(0,(stock/(2*GORE_PLEATS))**2
                                 -(d/(2*GORE_PLEATS))**2))
    rings = []
    for j in range(2*GORE_PLEATS+1):
        t = j/(2*GORE_PLEATS)
        n = (1-t)*fixed_n+t*moving_n
        n /= np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-9)
        fold = -amplitude if j%2 else np.zeros_like(amplitude)
        rings.append((1-t)*fixed+t*moving
                     +(-2*math.sin(math.pi*t)+fold)[:,None]*n)
    return strip_mesh(rings,(48,55,57,255)),np.asarray(rings),d,amplitude


def inside_counts(samples, hard, labels):
    """Classify sampled material points and locate the failing material zone."""
    hits = {}
    for name, shape in hard.items():
        total = 0
        failed_indices = []
        for solid in shape.solids():
            bb = solid.bounding_box()
            lo = np.array((bb.min.X,bb.min.Y,bb.min.Z))
            hi = np.array((bb.max.X,bb.max.Y,bb.max.Z))
            indices = np.flatnonzero(np.all((samples>=lo)&(samples<=hi),axis=1))
            if not len(indices):
                continue
            cl = BRepClass3d_SolidClassifier(solid.wrapped)
            for index in indices:
                xyz = samples[index]
                cl.Perform(gp_Pnt(*map(float,xyz)),1e-4)
                if cl.State()==TopAbs_IN:
                    total += 1
                    failed_indices.append(index)
        if total:
            zone = labels[failed_indices]
            hits[name] = dict(points=int(total),
                              local_z_or_gore_fraction=[round(float(zone[:,0].min()),2),
                                                        round(float(zone[:,0].max()),2)],
                              azimuth_deg=[round(float(zone[:,1].min()),1),
                                           round(float(zone[:,1].max()),1)])
    return hits


def main():
    fixed,moving,hardware = hard_parts()
    # A short fixed lip is already embodied by the existing cuff surface.
    # Its lower edge is at Z=112.93; the front/lateral tongue slides inside it.
    phi = np.deg2rad(np.linspace(TONGUE_END_DEG,
                                  TONGUE_START_DEG+360,61))
    f115,m115,_,_ = rear_loops(phi,115)
    stock = np.linalg.norm(m115-f115,axis=1)+1.5
    rows = []
    for angle in ANGLES:
        print(f"Checking {angle} degrees",flush=True)
        tongue,outer,inner = tongue_mesh(angle)
        gore,rings,d,amp = gore_mesh(angle,phi,stock)
        posed = {n:turn(s,tuple(AXIS),angle) for n,s in moving.items()}
        # The proximal tongue root is deliberately inside its parent shin.
        # Sample its free rigid part above local Z=118.5, at 1/3 density.
        tongue_samples = np.concatenate((outer[8::3,::3].reshape((-1,3)),
                                          inner[8::3,::3].reshape((-1,3))))
        sampled_z = np.linspace(TONGUE_ROOT_Z,TONGUE_TIP_Z,38)[8::3]
        sampled_phi = np.linspace(TONGUE_START_DEG,TONGUE_END_DEG,85)[::3]
        tongue_labels_one = np.stack((np.repeat(sampled_z,len(sampled_phi)),
                                      np.tile(sampled_phi,len(sampled_z))),axis=1)
        tongue_labels = np.concatenate((tongue_labels_one,tongue_labels_one))
        fixed_hard = {**fixed,**hardware,
                      **{n:posed[n] for n in ("yoke","bridge","shin_cross_pad")}}
        tongue_hits = inside_counts(tongue_samples,fixed_hard,tongue_labels)
        # Fabric anchors are allowed to sit underneath the lips, but its free
        # field must avoid all exact fixed and moving hard solids.
        gore_samples = rings[1:-1:2,::3].reshape((-1,3))
        gore_j = np.arange(1,2*GORE_PLEATS,2)/(2*GORE_PLEATS)
        gore_phi = np.rad2deg(phi[::3])
        gore_labels = np.stack((np.repeat(gore_j,len(gore_phi)),
                                np.tile(gore_phi,len(gore_j))),axis=1)
        gore_hard = {**fixed,**hardware,**posed}
        gore_hits = inside_counts(gore_samples,gore_hard,gore_labels)
        gore_path = np.linalg.norm(np.diff(rings,axis=0),axis=2).sum(axis=0)
        # Existing hard geometry is unchanged, so the c48e952 cut retains
        # its 0.5 mm cutter margin. Report any newly introduced tongue hits.
        scene = trimesh.Scene()
        for name,shape in fixed.items():
            mesh(scene,name,shape,(230,222,207,255))
        for name in ("R_shin_front","R_shin_back"):
            mesh(scene,name,posed[name],(230,222,207,255))
        scene.add_geometry(tongue,geom_name="moving_shin_tongue",
                           node_name="moving_shin_tongue")
        scene.add_geometry(gore,geom_name="rear_flex_gore",
                           node_name="rear_flex_gore")
        scene.export(HERE/f"overlap-{angle}.glb")
        rows.append(dict(angle_deg=angle,
                         tongue_free_samples=len(tongue_samples),
                         tongue_inside_hard=tongue_hits,
                         gore_free_samples=len(gore_samples),
                         gore_inside_hard=gore_hits,
                         gore_anchor_distance_mm=[round(float(d.min()),2),round(float(d.max()),2)],
                         gore_fold_peak_mm=[round(float(amp.min()),2),round(float(amp.max()),2)],
                         gore_actual_path_mm=[round(float(gore_path.min()),2),
                                              round(float(gore_path.max()),2)],
                         gore_max_path_change_pct=round(float(np.max(np.abs(100*(gore_path/stock-1)))),2)))
    report = dict(source_commits=["c48e952","42f2ddc","928cc11","f61bbff"],
                  geometry="existing 2 mm cuff as fixed lip; 2 mm shin-rooted front/lateral tongue; separate rear PU/nylon gore",
                  hard_part_cutter_margin_mm=0.5,
                  offset_axis_mm=list(map(float,AXIS)),
                  tongue_local_z_mm=[TONGUE_ROOT_Z,TONGUE_TIP_Z],
                  tongue_azimuth_deg=[TONGUE_START_DEG,TONGUE_END_DEG],
                  rear_gore_azimuth_deg=[TONGUE_END_DEG,TONGUE_START_DEG+360],
                  rear_gore_cut_length_mm=[round(float(stock.min()),2),
                                           round(float(stock.max()),2)],
                  rear_gore_ideal_flat_stack_mm=round(2*GORE_PLEATS*GORE_MATERIAL_THICKNESS,2),
                  fixed_skin_solids={n:len(s.solids()) for n,s in fixed.items()},
                  rows=rows,
                  verdict="REJECT: tongue penetrates cuff/rear thigh/belt; rear gore penetrates cuff at stand and yoke/bridge when bent, while leaving a large 115-degree rear opening",
                  qualification="point-in-solid screen of sampled surface points, not continuous triangle intersection, mechanical fastening, fold volume, fatigue or whole-robot validation")
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
