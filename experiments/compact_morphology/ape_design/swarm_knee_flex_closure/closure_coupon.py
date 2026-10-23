"""Reproducible 50 cm offset-knee flex-cover *trial*, not a qualified gaiter.

Run from the repository root with `.venv/bin/python path/to/closure_coupon.py`.
The hard parts are rebuilt from the committed c48e952 source, then an explicitly
anchored ten-convolution membrane is sampled at three poses. The report marks
penetrations and fabric-path limitations; a pretty mesh is not a pass.
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
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import (
    integration_shapes, turn,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

AXIS = np.array((66.0, 12.0, 158.0))
ANGLES = (0, 60, 115)
PANEL_COUNT = 10
PHI_COUNT = 96
MARGIN = 0.5
FABRIC_THICKNESS = 0.229  # Dynatect 9UN PU/nylon nominal 0.009 in


def overlap(a, c):
    ab, cb = a.bounding_box(), c.bounding_box()
    if any(getattr(ab.min, axis) > getattr(cb.max, axis)
           or getattr(cb.min, axis) > getattr(ab.max, axis) for axis in "XYZ"):
        return 0.0
    common = a & c
    return 0.0 if common is None else sum(s.volume for s in common.solids())


def hard_parts():
    """Copy the c48e952 exact cutter choices without writing in its scope."""
    case = integration_shapes(500)
    shift = b.Pos(0, 10, 12)
    fixed_hw = {n: shift * s for n, s in case["fixed_knee_hardware"].items()}
    moving = {
        "yoke": shift * case["moving_knee_hardware"]["yoke"],
        "bridge": b.Pos(61, 11, 116) * b.Box(6, 12, 12),
        "shin_cross_pad": case["moving_knee_hardware"]["shin_cross_pad"],
        **{n: case["moving_shin_skin"][n]
           for n in ("R_shin_front", "R_shin_back")},
    }
    moving_cutters = {n: b.offset(moving[n], amount=MARGIN)
                      for n in ("yoke", "bridge")}
    fixed_cutters = {n: b.offset(fixed_hw[n], amount=MARGIN)
                     for n in ("motor", "support_B")}
    skin = {}
    for name, source in case["fixed_knee_skin"].items():
        cut = source
        for cutter in fixed_cutters.values():
            if overlap(cut, cutter) > 0.05:
                cut = cut - cutter
        for angle in (0, 15, 30, 60, 90, 115):
            for cutter in moving_cutters.values():
                posed = turn(cutter, tuple(AXIS), angle)
                if overlap(cut, posed) > 0.05:
                    cut = cut - posed
        skin[name] = cut
    return skin, moving, fixed_hw


def rotate(points, angle):
    a = math.radians(angle)
    r = np.array(((1, 0, 0), (0, math.cos(a), -math.sin(a)),
                  (0, math.sin(a), math.cos(a))))
    return (points - AXIS) @ r.T + AXIS


def loops(phi, angle):
    """Two 2 mm-inset fixing loops below the existing rigid lips.

    Fixed loop: ellipsoid cuff at z=125.8, 2 mm below its outer surface.
    Moving loop: source shin loft at z=111, about 7 mm below its upper rim.
    These are geometric seam datums, not a fastener or adhesive design.
    """
    fixed = np.stack((60 + 18.5*np.cos(phi),
                      4.286 + 21.2*np.sin(phi),
                      np.full_like(phi, 125.8)), axis=1)
    moving_local = np.stack((60.94 + 23.7*np.cos(phi),
                             5.18 + 25.06*np.sin(phi),
                             np.full_like(phi, 111.0)), axis=1)
    moving = rotate(moving_local, angle)
    fixed_n = np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=1)
    moving_n = rotate(moving_local + fixed_n, angle) - moving
    return fixed, moving, fixed_n, moving_n


def cover_mesh(angle, phi, material_length):
    fixed, moving, fixed_n, moving_n = loops(phi, angle)
    d = np.linalg.norm(moving - fixed, axis=1)
    rings = []
    # Every peak-to-valley edge uses the same developed length at all poses.
    # The panel stock length is cut separately at each azimuth.
    amplitude = np.sqrt(np.maximum(0, (material_length/(2*PANEL_COUNT))**2
                                 - (d/(2*PANEL_COUNT))**2))
    for j in range(2*PANEL_COUNT + 1):
        t = j/(2*PANEL_COUNT)
        base = (1-t)*fixed + t*moving
        normal = (1-t)*fixed_n + t*moving_n
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-9)
        # A shallow inward trough controls the center; every second ring is
        # the folded peak. This is an optimistic kinematic shape, not FEA.
        inward = -2.0*math.sin(math.pi*t)
        fold = -amplitude if j % 2 else np.zeros_like(amplitude)
        rings.append(base + (inward + fold)[:, None]*normal)
    vertices = np.concatenate(rings)
    faces = []
    for j in range(2*PANEL_COUNT):
        for i in range(len(phi)):
            k = (i+1) % len(phi)
            a, c = j*len(phi)+i, j*len(phi)+k
            b_, d_ = (j+1)*len(phi)+i, (j+1)*len(phi)+k
            faces.extend(((a, c, b_), (c, d_, b_)))
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.visual.vertex_colors = (42, 47, 50, 255)
    return tri, d, amplitude


def inside_count(mesh_, hard):
    # First-pass material/solid screen: vertices and triangle centroids.
    # A zero result still would not certify whole-triangle separation.
    vertices = mesh_.vertices
    centroids = mesh_.triangles_center
    samples = np.concatenate((vertices, centroids))
    # The first and last rows intentionally tuck under their anchor lips.
    # Evaluate the free sheet only, 5% <= path <= 95%.
    z = len(vertices)//(2*PANEL_COUNT+1)
    mask = np.ones(len(samples), dtype=bool)
    mask[:z] = False
    mask[2*PANEL_COUNT*z:(2*PANEL_COUNT+1)*z] = False
    for j in (0, 2*PANEL_COUNT-1):
        mask[len(vertices)+j*2*z:len(vertices)+(j+1)*2*z] = False
    samples = samples[mask][::8]
    hits = {}
    for name, shape in hard.items():
        classifiers = []
        for s in shape.solids():
            bb = s.bounding_box()
            lo = np.array((bb.min.X, bb.min.Y, bb.min.Z))
            hi = np.array((bb.max.X, bb.max.Y, bb.max.Z))
            candidate = samples[np.all((samples >= lo) & (samples <= hi), axis=1)]
            if len(candidate):
                classifiers.append((BRepClass3d_SolidClassifier(s.wrapped), candidate))
        count = 0
        for cl, candidate in classifiers:
            for x, y, z_ in candidate:
                cl.Perform(gp_Pnt(float(x),float(y),float(z_)),1e-4)
                if cl.State() == TopAbs_IN:
                    count += 1
        if count:
            hits[name] = count
    return hits, len(samples)


def main():
    fixed, moving, hardware = hard_parts()
    phi = np.arange(PHI_COUNT)*2*math.pi/PHI_COUNT
    f0, m0, _, _ = loops(phi, 0)
    f115, m115, _, _ = loops(phi, 115)
    d0 = np.linalg.norm(m0-f0, axis=1)
    d115 = np.linalg.norm(m115-f115, axis=1)
    # Each circumferential strip gets 1.5 mm end take-up after its 115° chord.
    # This is the most generous ten-pleat trial, not a certified pattern.
    stock = d115 + 1.5
    rows = []
    reference_widths = None
    for angle in ANGLES:
        print(f"Checking {angle} degrees", flush=True)
        gaiter, distance, amplitude = cover_mesh(angle, phi, stock)
        posed_moving = {n: turn(s, tuple(AXIS), angle) for n, s in moving.items()}
        hard = {**fixed, **hardware, **posed_moving}
        hits, sample_n = inside_count(gaiter, hard)
        rings = gaiter.vertices.reshape((2*PANEL_COUNT+1, PHI_COUNT, 3))
        developed = np.linalg.norm(np.diff(rings, axis=0), axis=2).sum(axis=0)
        widths = np.linalg.norm(np.roll(rings, -1, axis=1)-rings, axis=2)
        if reference_widths is None:
            reference_widths = widths
        width_change = 100*(widths/reference_widths-1)
        baseline_hits = []
        for n in ("R_shin_front", "R_shin_back", "yoke", "bridge"):
            for fixed_n, fixed_s in fixed.items():
                v = overlap(posed_moving[n], fixed_s)
                if v > 0.05:
                    baseline_hits.append((n, fixed_n, round(v, 3)))
        scene = trimesh.Scene()
        for n, s in fixed.items():
            mesh(scene, n, s, (230, 222, 207, 255))
        for n in ("R_shin_front", "R_shin_back"):
            mesh(scene, n, posed_moving[n], (230, 222, 207, 255))
        scene.add_geometry(gaiter, geom_name="trial_flex_cover",node_name="trial_flex_cover")
        scene.export(HERE / f"trial-{angle}.glb")
        rows.append(dict(angle_deg=angle,
                         endpoint_distance_mm=[round(float(distance.min()),2),
                                               round(float(distance.max()),2)],
                         fold_peak_height_mm=[round(float(amplitude.min()),2),
                                              round(float(amplitude.max()),2)],
                         actual_developed_path_mm=[round(float(developed.min()),2),
                                                   round(float(developed.max()),2)],
                         maximum_strip_length_change_pct=round(float(np.max(np.abs(100*(developed/stock-1)))),2),
                         maximum_ring_edge_change_pct=round(float(np.max(np.abs(width_change))),2),
                         baseline_rigid_skin_hits=baseline_hits,
                         sampled_free_sheet_points=sample_n,
                         material_inside_hard_parts=hits))
    report = dict(source_commits=["c48e952", "42f2ddc"],
                  hard_margin_mm=MARGIN, fabric_nominal_mm=FABRIC_THICKNESS,
                  topology="closed ten-pleat PU/nylon ring, ruled between two inset loops",
                  fixed_anchor="cuff ellipsoid at z125.8, radial inset 2 mm",
                  moving_anchor="shin loft at local z111, radial inset 2 mm",
                  cut_strip_length_mm=[round(float(stock.min()),2),
                                       round(float(stock.max()),2)],
                  neutral_anchor_span_mm=[round(float(d0.min()),2),
                                          round(float(d0.max()),2)],
                  ideal_flat_layer_stack_mm=round(2*PANEL_COUNT*FABRIC_THICKNESS,2),
                  baseline_skin_solids={n:len(s.solids()) for n,s in fixed.items()},
                  rows=rows,
                  verdict="REJECT: free membrane penetrates rigid skin and moving mechanism at 60/115 degrees",
                  qualification="visual/point-sample trial only; no sewn flange, stiffener thickness, fold radius, tensile or fatigue calculation, whole-triangle collision, self-intersection, or full-body motion gate")
    (HERE / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
