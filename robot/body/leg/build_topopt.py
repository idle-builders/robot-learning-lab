"""Build Family 2: Blank-Space 3D Topology-Optimized Lower-Leg Connectors (BRep CAD Reconstruction).

Reconstructs the 3D SIMP topology-optimized load-path skeletons (`topopt/topopt-fields.npz`)
into smooth, watertight, analytical OpenCascade BRep solids with exact cylindrical bores,
counterbores, and pressed-in stainless steel compression sleeves for all 3 lower-leg stages:
  1. Stage 1 (Shin): R/L_shin_topopt + 4 steel compression sleeves
  2. Stage 2 (Ankle Coupler): R/L_ankle_coupler_topopt + 10 steel compression sleeves
  3. Stage 3 (Foot Mount & Sole): R/L_foot_monocoque_topopt + 4 steel compression sleeves
"""
from pathlib import Path
import math
import numpy as np
from build123d import (
    Axis,
    Box,
    Circle,
    Compound,
    Cylinder,
    Edge,
    Face,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Solid,
    Vector,
    Wire,
    export_step,
    export_stl,
    extrude,
)
from common import (
    ART,
    MATERIALS,
    PRINTED,
    ALT_PRINTED,
    rel,
    sha,
    write_json,
    ref,
    fillet_2d_face,
    bevel_3d_edges,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    add_shin_shell_mounts,
)
from build_truss_box import box_pt, cyl_x, cyl_y, cyl_z, strut_3d, poly_prism_x, poly_prism_y


def capsule_strut_3d(p0, p1, radius_y: float, height_z: float):
    """Create a smooth organic rounded-end 3D strut between p0 and p1."""
    s = strut_3d(p0, p1, 2.0 * radius_y, height_z)
    c0 = cyl_z(radius_y, p0[2] - 0.5 * height_z, p0[2] + 0.5 * height_z, p0[0], p0[1])
    c1 = cyl_z(radius_y, p1[2] - 0.5 * height_z, p1[2] + 0.5 * height_z, p1[0], p1[1])
    return s + c0 + c1


def build_shin_topopt():
    """Reconstruct Stage 1 Shin from the 3D SIMP topology-optimized load paths with 100% symmetric circular hubs,
    4 screws per interface (16 steel sleeves, zero figure-8 holes), and 3D edge bevels."""
    z = np.load(ART / "topopt" / "topopt-fields.npz")
    rho = z["shin_rho"]
    high_fraction = float((rho > 0.45).sum() / max(1, (rho > 0).sum()))

    # 1. Symmetric V-branch cheek skeletons on Inner (X in [39.25, 41.25]) and Outer (X in [78.75, 81.05]) cheeks
    ch_poly_t = Polygon(
        (-9.5, 66.0),
        (17.5, 66.0),
        (17.5, 98.0),
        (14.5, 110.0),
        (14.5, 123.0),
        (-6.5, 123.0),
        (-6.5, 110.0),
        (-9.5, 98.0),
        align=None,
    )
    ch_union_t = (ch_poly_t + Pos(4.0, 123.0) * Circle(10.5)).clean().faces()[0]
    ch_face_in = fillet_2d_face(ch_union_t, 2.0, skip_xy=[(-9.5, 66.0), (17.5, 66.0), (-9.5, 98.0)], r_inner=3.0)
    ch_face_in = (ch_face_in - Pos(4.0, 81.0) * Circle(4.2).faces()[0]).clean().faces()[0]
    ch_face_out = fillet_2d_face(ch_union_t, 2.0, skip_xy=[(-9.5, 66.0), (17.5, 66.0), (-9.5, 98.0)], r_inner=3.0)
    ch_face_out = (ch_face_out - Pos(4.0, 81.0) * Circle(4.2).faces()[0]).clean().faces()[0]
    inner_cheek = extrude_xy_to_x(ch_face_in, 39.25, 41.25)
    outer_cheek = extrude_xy_to_x(ch_face_out, 78.75, 81.05)

    # 2. Integral cylindrical motor ear bosses for ALL 8 PA2 bolts (4 inner cheek + 4 outer cheek)
    standoffs = (
        cyl_x(3.0, 39.25, 44.0, -6.25, 69.30)
        + cyl_x(3.0, 39.25, 44.0, -6.25, 93.75)
        + cyl_x(3.0, 39.25, 44.0, 14.25, 69.30)
        + cyl_x(3.0, 39.25, 44.0, 14.25, 93.75)
        + cyl_x(3.0, 76.0, 81.05, -6.25, 69.30)
        + cyl_x(3.0, 76.0, 81.05, -6.25, 90.00)
        + cyl_x(3.0, 76.0, 81.05, 14.25, 69.30)
        + cyl_x(3.0, 76.0, 81.05, 14.25, 90.00)
    )

    # 3. Blank-Space Topology-Optimized 3D Organic Saddle Arch across X in [41.25, 78.75]
    ab_outer = Polygon((41.25, -9.5), (78.75, -9.5), (78.75, -2.4), (41.25, -2.4), align=None).faces()[0]
    ab_w1 = Pos(52.0, -5.95) * Circle(2.0).faces()[0]
    ab_w2 = Pos(68.0, -5.95) * Circle(2.0).faces()[0]
    arch_block = extrude_xy_to_z((ab_outer - ab_w1 - ab_w2).clean().faces()[0], 96.8, 108.4)

    # Anterior (-Y) organic wishbone arch with 2D-filleted windows
    wb_outer = Polygon((41.25, 72.2), (78.75, 72.2), (78.75, 108.4), (41.25, 108.4), align=None).faces()[0]
    wb_c1 = fillet_2d_face(Polygon((46.5, 75.6), (73.5, 75.6), (60.0, 95.0), align=None).faces()[0], 2.5)
    wb_c2 = Pos(49.0, 88.5) * Circle(4.2).faces()[0]
    wb_c3 = Pos(71.0, 88.5) * Circle(4.2).faces()[0]
    front_wishbone = extrude_xy_to_y((wb_outer - wb_c1 - wb_c2 - wb_c3).clean().faces()[0], -9.5, -8.55)

    ear_blend_in_f = box_pt((41.25, -9.5, 67.2), (44.0, -4.5, 75.4))
    ear_blend_out_f = box_pt((76.0, -9.5, 67.2), (78.75, -4.5, 75.4))
    rt_f = fillet_2d_face(
        Polygon((-9.2, 96.8), (-2.4, 96.8), (-2.4, 112.0), (-6.0, 112.0), align=None).faces()[0],
        1.5,
        skip_xy=[(-9.2, 96.8), (-2.4, 96.8)],
    )
    root_in = extrude_xy_to_x(rt_f, 39.25, 42.2)
    root_out = extrude_xy_to_x(rt_f, 77.8, 81.05)

    body = (
        inner_cheek
        + outer_cheek
        + standoffs
        + arch_block
        + front_wishbone
        + ear_blend_in_f
        + ear_blend_out_f
        + root_in
        + root_out
    )

    # 4. Exact functional cutouts, bores, and 16 steel sleeves
    body -= cyl_x(4.0, 38.0, 83.0, 4.0, 123.0)
    body -= cyl_x(11.0, 38.0, 83.0, 4.0, 61.0)
    body -= box_pt((44.0, -8.55, 50.5), (76.0, 16.55, 96.5))
    body -= box_pt((41.25, -3.5, 50.5), (78.75, 11.5, 96.5))

    sleeves = []
    for y_k, z_k in [(4.0, 116.0), (4.0, 130.0), (-3.0, 123.0), (11.0, 123.0)]:
        body -= cyl_x(2.25, 39.1, 41.4, y_k, z_k)
        body -= cyl_x(2.25, 78.6, 80.85, y_k, z_k)
        body -= cyl_x(3.0, 80.75, 82.0, y_k, z_k)
        s_in = cyl_x(2.25, 39.25, 41.25, y_k, z_k) - cyl_x(1.6, 39.1, 41.4, y_k, z_k)
        s_out = cyl_x(2.25, 78.75, 80.75, y_k, z_k) - cyl_x(1.6, 78.6, 80.9, y_k, z_k)
        sleeves.extend([s_in, s_out])

    for y_c, z_c in [(-6.25, 69.30), (-6.25, 93.75), (14.25, 69.30), (14.25, 93.75)]:
        body -= cyl_x(1.80, 39.1, 44.1, y_c, z_c)
        s_in_c = cyl_x(1.80, 39.25, 44.0, y_c, z_c) - cyl_x(1.1, 39.1, 44.1, y_c, z_c)
        sleeves.append(s_in_c)

    for y_c, z_c in [(-6.25, 69.30), (-6.25, 90.00), (14.25, 69.30), (14.25, 90.00)]:
        body -= cyl_x(1.80, 75.9, 80.85, y_c, z_c)
        body -= cyl_x(2.25, 80.75, 82.0, y_c, z_c)
        s_out_c = cyl_x(1.80, 76.0, 80.75, y_c, z_c) - cyl_x(1.1, 75.9, 80.85, y_c, z_c)
        sleeves.append(s_out_c)

    # 5. Integrate the 2 hard-shell mounting posts, Ø3.4 mm through-bores, and captive M3 nut pockets (`shin_front` & `shin_back`)
    body = add_shin_shell_mounts(body)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves), high_fraction


def build_ankle_coupler_topopt():
    """Reconstruct Stage 2 Ankle Coupler from the 3D SIMP topology-optimized wishbone arch
    with 12 steel sleeves (4 Pitch Horn + 4 Pitch Idler + 4 Roll Horn), full multi-pose clearance
    inside `shin_front`/`shin_back` across -35..+35 deg, and 3D edge bevels."""
    z = np.load(ART / "topopt" / "topopt-fields.npz")
    rho = z["coupler_rho"]
    high_fraction = float((rho > 0.45).sum() / max(1, (rho > 0).sum()))

    # 1. Cup-Rimmed Sculpted Main Pitch Hub + Deep-Beam Lateral Tendon (X in [74.5, 81.2])
    mh_f = fillet_2d_face(
        (
            Polygon((-2.0, 50.5), (3.5, 33.6), (20.6, 33.6), (20.6, 50.0), (14.5, 55.5), (14.5, 61.0), (-6.5, 61.0), align=None)
            + Pos(4.0, 61.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(3.5, 33.6), (20.6, 33.6), (20.6, 50.0)],
        r_inner=2.5,
    )
    mh_f = (mh_f - Pos(10.0, 44.5) * Circle(2.4).faces()[0]).clean().faces()[0]
    main_hub = extrude_xy_to_x(mh_f, 74.5, 81.2) - cyl_x(9.90, 74.0, 79.2, 4.0, 61.0) - box_pt((74.0, -10.0, 48.0), (76.8, 16.65, 75.0))

    # 2. Cup-Rimmed Symmetric Idler Hub + Deep Neck Blend (X in [39.95, 45.0]) supporting all 4 M3 bolts
    ih_f = fillet_2d_face(
        (
            Polygon((-2.0, 50.5), (6.5, 41.0), (20.6, 41.0), (20.6, 50.0), (14.5, 55.5), (14.5, 61.0), (-6.5, 61.0), align=None)
            + Pos(4.0, 61.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(6.5, 41.0), (20.6, 41.0), (20.6, 50.0)],
        r_inner=2.5,
    )
    idler_hub = extrude_xy_to_x(ih_f, 39.95, 45.0) - cyl_x(9.90, 41.95, 45.3, 4.0, 61.0)

    # 3. Organic Bifurcated Wishbone Arch + Roll Hub (Y in [16.65, 20.6])
    aw_f = fillet_2d_face(
        (
            Polygon((39.95, 41.0), (64.5, 29.0), (64.5, 19.0), (81.2, 19.0), (81.2, 50.0), (39.95, 50.0), align=None)
            + Pos(75.0, 19.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(39.95, 41.0), (39.95, 50.0), (81.2, 50.0)],
        r_inner=3.0,
    )
    aw_f = (
        aw_f
        - Pos(74.2, 40.0) * Circle(2.8).faces()[0]
        - Pos(51.5, 45.2) * Circle(2.4).faces()[0]
        - Pos(62.5, 44.5) * Circle(2.8).faces()[0]
    ).clean().faces()[0]
    arch_truss = extrude_xy_to_y(aw_f, 16.65, 20.6) + cyl_y(10.5, 16.65, 20.8, 75.0, 19.0)

    body = main_hub + idler_hub + arch_truss

    # 4. Cut exact horn central windows, motor keep-outs, and 12 steel sleeve seats
    body -= cyl_y(9.90, 16.0, 18.0, 75.0, 19.0)
    body -= cyl_x(4.5, 37.0, 85.0, 4.0, 61.0)
    body -= cyl_y(4.5, 16.0, 23.5, 75.0, 19.0)
    body -= box_pt((45.0, -12.0, 46.5), (76.8, 16.65, 75.0))
    body -= box_pt((37.0, -12.0, 5.0), (74.5, 16.65, 34.5))
    body -= box_pt((74.5, -12.0, 5.0), (85.0, 16.65, 33.6))

    sleeves = []
    for y_p, z_p in [(4.0, 54.0), (4.0, 68.0), (-3.0, 61.0), (11.0, 61.0)]:
        body -= cyl_x(2.25, 74.0, 83.5, y_p, z_p)
        s = cyl_x(2.25, 79.2, 81.2, y_p, z_p) - cyl_x(1.6, 79.0, 83.5, y_p, z_p)
        sleeves.append(s)

    for y_i, z_i in [(4.0, 54.0), (4.0, 68.0), (-3.0, 61.0), (11.0, 61.0)]:
        body -= cyl_x(2.25, 38.0, 45.5, y_i, z_i)
        s = cyl_x(2.25, 39.95, 41.95, y_i, z_i) - cyl_x(1.6, 38.0, 42.1, y_i, z_i)
        sleeves.append(s)

    for x_r, z_r in [(68.0, 19.0), (82.0, 19.0), (75.0, 12.0), (75.0, 26.0)]:
        body -= cyl_y(2.25, 16.0, 22.5, x_r, z_r)
        s = cyl_y(2.25, 18.0, 20.8, x_r, z_r) - cyl_y(1.6, 17.8, 22.5, x_r, z_r)
        sleeves.append(s)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves), high_fraction


def build_foot_monocoque_topopt():
    """Reconstruct Stage 3 Foot Monocoque from the 3D SIMP topology-optimized branching root ribs with 3D edge bevels."""
    z = np.load(ART / "topopt" / "topopt-fields.npz")
    rho = z["foot_rho"]
    high_fraction = float((rho > 0.45).sum() / max(1, (rho > 0).sum()))

    sole_outer_full = extrude(RectangleRounded(60.0, 86.0, 22.0), 14.0).translate((67.0, -12.0, 2.0))

    # 1. Continuous flat bottom membrane (Z in [2.0, 3.4]) for uninterrupted TPU pad & sensor bonding
    bottom_membrane = extrude(RectangleRounded(60.0, 86.0, 22.0), 1.4).translate((67.0, -12.0, 2.0))

    # 2. Organic Branching Sole Web with Circular/Elliptical SIMP Relief Windows (Z in [2.0, 5.4])
    sole_2d_t = (Pos(67.0, -12.0) * RectangleRounded(60.0, 86.0, 22.0)).faces()[0]
    pockets_2d_t = [
        Pos(54.5, -38.5) * Circle(6.5).faces()[0],
        Pos(54.5, 10.5) * Circle(5.5).faces()[0],
        Pos(66.0, -1.5) * Circle(5.5).faces()[0],
        Pos(81.5, -13.0) * Circle(7.5).faces()[0],
    ]
    for pk in pockets_2d_t:
        sole_2d_t = (sole_2d_t - pk).clean().faces()[0]
    sole_organic = extrude_xy_to_z(sole_2d_t, 2.0, 5.4)

    # 3. Sculpted Organic Arch Saddles for Front & Rear Motor Case Ears + 2D-filleted buttress gussets + Medial Arch Spine
    fg_f = fillet_2d_face(
        Polygon((19.5, 4.0), (27.5, 4.0), (27.5, 6.0), (20.35, 13.5), (19.5, 13.5), align=None).faces()[0],
        1.5,
        skip_xy=[(19.5, 4.0), (27.5, 4.0), (19.5, 13.5), (20.35, 13.5)],
    )
    rg_f = fillet_2d_face(
        Polygon((-19.95, 4.0), (-28.5, 4.0), (-28.5, 6.0), (-22.75, 13.5), (-19.95, 13.5), align=None).faces()[0],
        1.5,
        skip_xy=[(-19.95, 4.0), (-28.5, 4.0), (-19.95, 13.5), (-22.75, 13.5)],
    )
    front_arch = (
        box_pt((38.2, 17.55, 2.0), (69.5, 20.35, 13.8))
        - cyl_y(4.2, 17.0, 21.0, 56.5, 14.8)
        + cyl_y(3.1, 14.80, 20.35, 46.0, 8.75)
        + cyl_y(3.1, 14.80, 20.35, 66.7, 8.75)
        + extrude_xy_to_x(fg_f, 38.5, 41.8)
        + extrude_xy_to_x(fg_f, 53.5, 58.5)
        + extrude_xy_to_x(fg_f, 61.2, 63.8)
    )
    rear_arch = (
        box_pt((38.2, -22.75, 2.0), (69.5, -19.95, 13.8))
        - cyl_y(4.2, -23.5, -19.0, 54.5, 14.8)
        + cyl_y(3.1, -22.75, -17.20, 42.25, 8.75)
        + cyl_y(3.1, -22.75, -17.20, 66.7, 8.75)
        + extrude_xy_to_x(rg_f, 46.0, 49.5)
        + extrude_xy_to_x(rg_f, 54.5, 58.8)
        + extrude_xy_to_x(rg_f, 61.2, 63.8)
    )
    medial_spine = box_pt((37.2, -22.75, 2.0), (39.6, 20.35, 13.0)) - cyl_x(3.0, 36.5, 40.5, -1.2, 8.2)

    body = (
        bottom_membrane
        + sole_organic
        + front_arch
        + rear_arch
        + medial_spine
    ) & sole_outer_full

    # 4. Cut exact motor horn clearance cylinder, motor case keep-outs, and 4 steel sleeves
    body -= cyl_y(11.2, -35.0, 35.0, 75.0, 19.0)
    body -= box_pt((39.75, -17.2, 6.3), (85.5, 14.8, 32.0))
    body -= box_pt((40.8, 14.8, 11.85), (78.5, 17.55, 28.0))
    body -= box_pt((45.3, -19.95, 11.85), (78.5, -17.2, 29.0))

    sleeves = []
    for x_f in [46.0, 66.7]:
        body -= cyl_y(1.9, 14.7, 19.65, x_f, 8.75)
        body -= cyl_y(2.4, 19.55, 29.0, x_f, 8.75)
        s = cyl_y(1.9, 14.80, 19.55, x_f, 8.75) - cyl_y(1.1, 14.7, 19.65, x_f, 8.75)
        sleeves.append(s)

    for x_r in [42.25, 66.7]:
        body -= cyl_y(1.9, -22.05, -17.1, x_r, 8.75)
        body -= cyl_y(2.4, -33.0, -21.95, x_r, 8.75)
        s = cyl_y(1.9, -21.95, -17.20, x_r, 8.75) - cyl_y(1.1, -22.05, -17.1, x_r, 8.75)
        sleeves.append(s)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves), high_fraction


def export_family():
    out_dir = ART / "topopt" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    shin_r, shin_sleeves_r, vf_shin = build_shin_topopt()
    coupler_r, coupler_sleeves_r, vf_coupler = build_ankle_coupler_topopt()
    foot_r, foot_sleeves_r, vf_foot = build_foot_monocoque_topopt()

    rho_poly = PRINTED["density_g_cm3"] * 1e-3
    rho_steel = MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3

    parts_meta = []
    for stage_key, base_name, link_suffix, role, poly_r, slv_r, vf in [
        ("stage1_shin", "shin_topopt", "shin", "frame", shin_r, shin_sleeves_r, vf_shin),
        ("stage2_ankle_coupler", "ankle_coupler_topopt", "ankle_pitch", "coupler", coupler_r, coupler_sleeves_r, vf_coupler),
        ("stage3_foot_monocoque", "foot_monocoque_topopt", "foot", "frame", foot_r, foot_sleeves_r, vf_foot),
    ]:
        for side, sgn in [("R", 1), ("L", -1)]:
            poly = poly_r if sgn == 1 else poly_r.mirror(Plane.YZ)
            slv = slv_r if sgn == 1 else slv_r.mirror(Plane.YZ)

            p_name = f"{side}_{base_name}"
            s_name = f"{side}_{base_name}_steel_sleeves"

            p_comp = Compound(children=list(poly.solids()))
            p_comp.label = p_name
            p_step = out_dir / f"{p_name}.step"
            p_stl = out_dir / f"{p_name}.stl"
            export_step(p_comp, p_step)
            export_stl(p_comp, p_stl, tolerance=0.04, angular_tolerance=0.1)

            s_comp = Compound(children=list(slv.solids()))
            s_comp.label = s_name
            s_step = out_dir / f"{s_name}.step"
            s_stl = out_dir / f"{s_name}.stl"
            export_step(s_comp, s_step)
            export_stl(s_comp, s_stl, tolerance=0.04, angular_tolerance=0.1)

            bb = p_comp.bounding_box()
            s_bb = s_comp.bounding_box()
            poly_vol = float(poly.volume)
            slv_vol = float(slv.volume)
            poly_mass_g = poly_vol * rho_poly
            slv_mass_g = slv_vol * rho_steel

            parts_meta.append({
                "stage": stage_key,
                "side": side,
                "name": p_name,
                "owning_link": f"{side}_{link_suffix}",
                "role": role,
                "native_step": rel(p_step),
                "native_step_sha256": sha(p_step),
                "mesh": rel(p_stl),
                "mesh_sha256": sha(p_stl),
                "native_valid": all(s.is_valid for s in p_comp.solids()),
                "native_solids": len(p_comp.solids()),
                "volume_mm3": round(poly_vol, 3),
                "simp_high_density_fraction": round(vf, 4),
                "polymer_mass_g_PAHT_CF": round(poly_mass_g, 3),
                "polymer_mass_g_PETG_CF": round(poly_vol * ALT_PRINTED["density_g_cm3"] * 1e-3, 3),
                "bounds_mm": [
                    [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)],
                    [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)],
                ],
                "sleeves": {
                    "name": s_name,
                    "native_step": rel(s_step),
                    "native_step_sha256": sha(s_step),
                    "mesh": rel(s_stl),
                    "mesh_sha256": sha(s_stl),
                    "count": len(s_comp.solids()),
                    "volume_mm3": round(slv_vol, 3),
                    "mass_g": round(slv_mass_g, 3),
                    "bounds_mm": [
                        [round(s_bb.min.X, 3), round(s_bb.min.Y, 3), round(s_bb.min.Z, 3)],
                        [round(s_bb.max.X, 3), round(s_bb.max.Y, 3), round(s_bb.max.Z, 3)],
                    ],
                },
                "total_stage_mass_g_PAHT_CF": round(poly_mass_g + slv_mass_g, 3),
            })

    summary_path = ART / "topopt" / "cad-manifest.json"
    write_json(summary_path, {
        "family": "cove50-lower-leg-fdm-topopt",
        "version": "r1",
        "description": "Blank-Space 3D SIMP Topology-Optimized Lower-Leg Connectors reconstructed as smooth BRep CAD solids with Steel Compression Sleeves",
        "topopt_solve_report": ref(ART / "topopt" / "topopt-solve-report.json"),
        "parts": parts_meta,
        "source_code": ref(Path(__file__)),
    })
    for row in parts_meta:
        if row["side"] == "R":
            print(
                f"{row['name']:32s} | valid={row['native_valid']} solids={row['native_solids']} "
                f"| poly={row['polymer_mass_g_PAHT_CF']:6.2f} g + sleeves={row['sleeves']['mass_g']:5.2f} g "
                f"= {row['total_stage_mass_g_PAHT_CF']:6.2f} g | bounds={row['bounds_mm']}"
            )


if __name__ == "__main__":
    export_family()
