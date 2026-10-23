"""First distal-joint coupon: a single moving thumb and a fixed palm.

This is a CAD fit experiment, not a production hand. The XL330 case and SKF
bearing sizes are published; the XL330 horn datum, printed mount and skin cut
still need validation before this can become the robot's gripper joint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import sketch_shell as shell


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-gripper-joint-coupon-20261003"
ANGLES = (10, 15, 20, 25, 30, 35)
# Pollen's XL330 reference mesh puts the horn axis 7.5 mm off the 34 mm
# body-center plane. Mounting it inverted places that axis below the case.
# The OEM drawing has a 23 mm body, 3 mm output boss, and optional 3 mm
# idler boss: 26 mm with output only or 29 mm with both installed.
MOTOR_CENTER = (0.5, 100, 165)
PIVOT = (15, 100, 157.5)


def volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    if shape is None:
        return 0.0
    return sum(part.volume for part in shape) if isinstance(shape, bd.ShapeList) else shape.volume


def cylinder_x(radius: float, length: float, x: float, y: float, z: float) -> bd.Shape:
    return bd.Pos(x, y, z) * bd.Rot(0, 90, 0) * bd.Cylinder(radius, length)


def cylinder_y(radius: float, length: float, x: float, y: float, z: float) -> bd.Shape:
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(radius, length)


def thumb(angle: float) -> bd.Shape:
    # The motor output carries one finger directly. The x-axis pivot sits in
    # the forearm; the tip emerges through the lower paw aperture.
    link = bd.Pos(19, 100, 145) * bd.Box(8, 6, 26)
    fingertip = shell.zloft([
        (107, 19, 100, 5, 7), (110, 19, 100, 9, 10),
        (122, 19, 100, 10, 12), (135, 19, 100, 9, 10),
    ])
    stop_tab = bd.Pos(24, 100, 143) * bd.Box(3, 3, 4)
    # A printed drive plate bolts to the standard XL330 output horn. ROBOTIS's
    # dimensioned drawing shows four 1.6 mm tapping holes on a 12 mm PCD.
    # This 2.4 mm cheek uses M2 clearance holes; angular clocking is set at
    # assembly and does not change the pattern.
    drive_plate = cylinder_x(9, 2.4, 17, 100, PIVOT[2])
    # A compact boss receives an M2 heat-set insert for an off-the-shelf
    # 3 x 8 mm shoulder screw. The screw's 3 mm shank then runs in the SKF
    # bearing; it is independent of the XL330 horn's central retaining screw.
    axle_anchor = cylinder_x(4.2, 5.2, 20.3, 100, PIVOT[2])
    straight = link.fuse(fingertip, stop_tab, drive_plate, axle_anchor)
    for i in range(4):
        az = math.pi / 4 + i * math.pi / 2
        y = 100 + 6 * math.cos(az)
        z = PIVOT[2] + 6 * math.sin(az)
        straight -= cylinder_x(1.1, 2.6, 17, y, z)
    # The center remains accessible for the horn-retaining M2 screw. The
    # outboard shaft/bearing interface is a separate piece, not this screw.
    straight -= cylinder_x(1.1, 2.6, 17, 100, PIVOT[2])
    straight -= cylinder_x(1.5, 3.2, 21.3, 100, PIVOT[2])
    straight -= cylinder_x(1.0, 4, 20.9, 100, PIVOT[2])
    return (bd.Pos(*PIVOT) * bd.Rot(angle, 0, 0) *
            bd.Pos(*(-x for x in PIVOT)) * straight)


def mesh(shape: bd.Shape, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(0.16, 0.16)
    result = trimesh.Trimesh(
        vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
        faces=np.asarray(faces), process=False,
    )
    result.merge_vertices()
    result.visual.vertex_colors = rgba
    return result


def add(scene: trimesh.Scene, name: str, shape: bd.Shape,
        color: tuple[int, int, int, int]) -> None:
    scene.add_geometry(mesh(shape, color), geom_name=name, node_name=name)


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    original_hand = next(part.shape for part in shell.PARTS if part.name == "left_hand_shell")
    forearm = next(part.shape for part in shell.PARTS if part.name == "left_forearm_shell")
    wrist_shadow = next(part.shape for part in shell.PARTS if part.name == "left_wrist_shadow")

    # Open a bottom/front object entry while keeping the back, side rim and
    # upper cap of the original paw. A narrow channel lets the finger emerge.
    aperture = bd.fillet((bd.Pos(17, 114, 98) * bd.Box(26, 42, 42)).edges(), 5)
    channel = bd.fillet((bd.Pos(18, 114, 132) * bd.Box(16, 29, 29)).edges(), 3)
    hand = original_hand - aperture - channel
    forearm_open = forearm - channel
    wrist_open = wrist_shadow - aperture - channel
    # Front cheek masks the straight load-bearing spine in normal views while
    # leaving the grasp mouth under the paw open. It ties into the hand and
    # forearm skins; the thumb runs behind its narrow sloped seam.
    front_cheek = shell.skin([
        (114, 29.5, 104, 6, 22), (122, 28.5, 106, 5, 34),
        (134, 29.5, 105, 6, 35), (144, 27.5, 102, 5, 27),
        (151, 25.5, 100, 4, 8),
    ], 1.7)
    front_cheek -= bd.Pos(24, 106, 145) * bd.Box(4, 12, 10)
    # A separate thin flexible cheek follows the approved outer silhouette.
    # Only the outboard crescent remains, leaving the underside grip mouth
    # clear. Its small intentional overlap with the rigid skin represents a
    # TPU snap/adhesive land, not a hard-part Boolean union.
    wrist_gaiter = shell.skin([
        (102, 26, 104, 4, 9), (108, 16, 104, 40, 35),
        (118, 8, 105, 54, 45),
        (130, 6, 104, 57, 48), (143, 5, 102, 55, 48),
        (153, 4, 101, 53, 47),
    ], 1.2).intersect(bd.Pos(60, 102, 130) * bd.Box(70.4, 80, 52))[0]

    motor_case = bd.Pos(*MOTOR_CENTER) * bd.Box(23, 20, 34)
    # Approximate horn protrusions from Pollen's source mesh. The clearance
    # encloses the entire 29 mm two-sided asset, not just the 26 mm output-side
    # assembly stated in the spec.
    motor = motor_case.fuse(
        cylinder_x(8, 3, 13.5, 100, PIVOT[2]),
        cylinder_x(8, 3, -12.5, 100, PIVOT[2]),
    )
    motor_clearance = bd.Pos(*MOTOR_CENTER) * bd.Box(29.6, 20.6, 34.6)
    # The fixed grip face is tied all the way to the motor clamp. A split and
    # real screw bosses are still required before this could be assembled.
    fixed_palm = bd.fillet((bd.Pos(17, 90, 106) * bd.Box(14, 8, 26)).edges(), 2)
    spine = bd.Pos(13, 87, 132) * bd.Box(8, 5, 54)
    collar = (bd.Pos(*MOTOR_CENTER) * bd.Box(33, 24, 38) - motor_clearance)
    # The case clamp has an output-side relief. The finger's root must pass
    # this wall; otherwise a visually clear tip hides a mechanical collision.
    collar -= cylinder_x(9.4, 3.5, 16, 100, PIVOT[2])
    collar -= bd.Pos(16, 104, 145) * bd.Box(3.2, 16, 20)
    outer_bridge = bd.Pos(0, 117.7, 165) * bd.Box(8, 15.4, 12)
    fixed_frame = fixed_palm.fuse(spine, collar, outer_bridge) - motor_clearance
    # A small outboard bearing takes the tip's bending load off the servo horn.
    # The seat has a 0.1 mm radial clearance and lips on BOTH axial faces;
    # the y=100 clamshell split allows the bearing to be laid into one half.
    # SKF supplies the published 3 x 10 x 4 mm envelope.
    bearing = cylinder_x(5, 4, 28, 100, PIVOT[2]) - cylinder_x(1.5, 4.1, 28, 100, PIVOT[2])
    bearing_seat = cylinder_x(7, 4.5, 28, 100, PIVOT[2]) - cylinder_x(5.1, 5, 28, 100, PIVOT[2])
    bearing_pocket = cylinder_x(5.1, 4.2, 28, 100, PIVOT[2])
    bearing_inner_lip = (cylinder_x(6.7, 1.1, 25.25, 100, PIVOT[2])
                         - cylinder_x(4.4, 1.2, 25.25, 100, PIVOT[2]))
    bearing_cover = (bd.fillet(
        cylinder_x(6.5, 2, 30.8, 100, PIVOT[2]).edges(), 0.7
    ) - cylinder_x(2.7, 2.2, 30.8, 100, PIVOT[2]))
    close_stop = bd.Pos(28.1, 99.2, 145) * bd.Box(8, 3, 12)
    open_stop = bd.Pos(28.1, 112.2, 145) * bd.Box(8, 3, 12)
    # Ondrives SHS2-8: 3 mm x 8 mm shoulder, M2 x 4 mm thread. The screw
    # head is represented by a low-profile 5 mm disc; its exact head profile
    # and socket are omitted from this fit envelope.
    shaft = (cylinder_x(1, 4, 20.9, 100, PIVOT[2])
             .fuse(cylinder_x(1.5, 8, 26.9, 100, PIVOT[2]),
                   cylinder_x(2.5, 1.4, 31.6, 100, PIVOT[2])))
    insert = cylinder_x(1.6, 3, 21.3, 100, PIVOT[2]) - cylinder_x(1, 3.1, 21.3, 100, PIVOT[2])
    # Two transverse clamp posts are below and above the motor case. The
    # outer side takes M2 clearance and a recessed head; the inner side has
    # a heat-set insert pilot opening at the seam. These posts also bridge
    # each printed half to its exterior skin rather than floating in a cavity.
    closure_datums = ((0, 102, 137, 48), (0, 97, 191, 47))
    closure_posts = [cylinder_y(3.6, span, x, y, z)
                     for x, y, z, span in closure_datums]
    # Reserve a separate 6 mm passage beside the upper screw post for the
    # XL330 lead to leave this coupon toward the elbow. Its connector and
    # strain relief remain part of the elbow integration, not guessed here.
    cable_port = bd.Pos(8, 100, 196.5) * bd.Cylinder(3, 27)
    rigid_forearm_palm = (fixed_frame.fuse(hand, forearm_open, wrist_open,
                                        front_cheek, bearing_seat,
                                        bearing_inner_lip, *closure_posts,
                                        bearing_cover, close_stop,
                                        open_stop) - bearing_pocket - cable_port)
    substantial_solids = [solid for solid in rigid_forearm_palm.solids() if solid.volume > 1]
    # A clamshell cut through the pivot plane permits inserting the case and
    # bearing before closing the forearm. Its fastener bosses still need the
    # verified XL330 mounting datum, so these are candidate print halves.
    inner_cut = substantial_solids[0].intersect(
        bd.Pos(0, -0.1, 150) * bd.Box(200, 199.8, 200)
    )
    outer_cut = substantial_solids[0].intersect(
        bd.Pos(0, 200.1, 150) * bd.Box(200, 199.8, 200)
    )
    inner_cut = max(inner_cut, key=lambda s: s.volume) if isinstance(inner_cut, bd.ShapeList) else inner_cut
    outer_cut = max(outer_cut, key=lambda s: s.volume) if isinstance(outer_cut, bd.ShapeList) else outer_cut
    for x, y, z, span in closure_datums:
        outer_cut = outer_cut.cut(cylinder_y(1.15, 60, x, 125, z))
        outer_cut = outer_cut.cut(cylinder_y(2.2, 12, x, y + span / 2 + 3, z))
        inner_cut = inner_cut.cut(cylinder_y(1.5, 4, x, 98, z))
    inner_solids = [s for s in (inner_cut if isinstance(inner_cut, bd.ShapeList)
                               else inner_cut.solids()) if s.volume > 1]
    outer_solids = [s for s in (outer_cut if isinstance(outer_cut, bd.ShapeList)
                               else outer_cut.solids()) if s.volume > 1]
    inner_half = bd.Compound(children=inner_solids)
    outer_half = bd.Compound(children=outer_solids)
    clamp_screws = bd.Compound(children=[
        cylinder_y(1, 25, 0, 110.5, 137).fuse(cylinder_y(2.1, 2, 0, 124, 137)),
        cylinder_y(1, 20, 0, 107.5, 191).fuse(cylinder_y(2.1, 2, 0, 118.5, 191)),
    ])
    cable_route = bd.Pos(8, 100, 196.5) * bd.Cylinder(2.5, 27)
    toy = bd.Pos(17, 109, 110) * bd.Sphere(10)
    bd.export_step(inner_half, OUT / "paw-forearm-inner-half.step")
    bd.export_step(outer_half, OUT / "paw-forearm-outer-half.step")
    bd.export_step(thumb(10), OUT / "paw-moving-thumb.step")
    bd.export_step(wrist_gaiter, OUT / "paw-flexible-wrist-cheek.step")
    # A one-degree geometric sweep catches collisions between the five-degree
    # poses used in the lightweight viewer. This is a CAD run, not a test suite.
    swept_overlap = {
        degree: round(volume(thumb(degree).intersect(rigid_forearm_palm)), 3)
        for degree in range(10, 36)
    }

    samples = []
    for angle in ANGLES:
        finger = thumb(angle)
        hits = {
            "hand_skin_mm3": round(volume(finger.intersect(hand)), 1),
            "forearm_skin_mm3": round(volume(finger.intersect(forearm_open)), 1),
            "wrist_skin_mm3": round(volume(finger.intersect(wrist_open)), 1),
            "fixed_frame_mm3": round(volume(finger.intersect(fixed_frame)), 1),
            "front_cheek_mm3": round(volume(finger.intersect(front_cheek)), 1),
            "rigid_forearm_palm_mm3": round(volume(finger.intersect(rigid_forearm_palm)), 1),
        }
        # At the tip, the finger center shifts laterally by 50*sin(angle).
        tip_y = 100 + 50 * np.sin(np.deg2rad(angle))
        tip_z = PIVOT[2] - 50 * np.cos(np.deg2rad(angle))
        aperture_mm = (tip_y - 5) - (90 + 4)
        samples.append({"angle_deg": angle, "tip_center_yz_mm": [round(tip_y, 2), round(tip_z, 2)],
                        "tip_to_fixed_face_gap_mm": round(aperture_mm, 2),
                        "20mm_target_vs_thumb_mm3": round(volume(toy.intersect(finger)), 1),
                        "intersections": hits})

        scene = trimesh.Scene()
        add(scene, "rigid_inner_half", inner_half, (236, 227, 211, 255))
        add(scene, "rigid_outer_half", outer_half, (233, 223, 207, 255))
        add(scene, "candidate_M2_clamp_screws", clamp_screws, (157, 165, 166, 255))
        add(scene, "flexible_wrist_cheek", wrist_gaiter, (224, 217, 201, 255))
        add(scene, "XL330_source_sized_case", motor, (67, 105, 150, 180))
        add(scene, "moving_thumb", finger, (229, 214, 190, 255))
        add(scene, "pivot_shaft_d3", shaft, (165, 174, 176, 255))
        add(scene, "M2_insert", insert, (193, 153, 72, 255))
        add(scene, "SKF_623_2Z_outboard_bearing", bearing, (206, 188, 104, 255))
        add(scene, "20mm_grasp_target", toy, (196, 125, 67, 130))
        scene.export(OUT / f"paw-{angle:02d}.glb")

    report = {
        "stage": "distal gripper geometry coupon; not a supported final joint",
        "reference_shell": "sketch_shell.py, left side at 35 cm; mirror for right",
        "source_motor": "ROBOTIS XL330-M288-T; 20x34x23 mm body plus 3 mm output boss (26 mm specified depth), 18 g, 0.52 Nm stall at 5 V",
        "source_motor_url": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
        "motor_drawing_crosscheck_url": "https://img2024.cnblogs.com/blog/2030440/202602/2030440-20260227180532067-2138822197.png",
        "motor_mesh_reference": "Pollen Robotics microduck_rl XL330 STL, 29x20x34 mm including both horn bosses; local horn axis 7.5 mm from body center, inverted for this paw",
        "motor_mesh_reference_url": "https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/robot/microduck/assets/xl330.stl",
        "motor_mesh_reference_sha256": "c9f6bf7dd9d8b307a4174b40ff12517b3ea470e5535319a5b7dcce4ba52cdfe1",
        "output_support": "Ondrives SHS2-8 3 mm shoulder screw through an SKF 623-2Z bearing, M2 thread into a 3 mm pilot heat-set insert pocket in the moving thumb; split printed seat and end lips retain the bearing axially in CAD",
        "source_shoulder_screw": "Ondrives SHS2-8; 3 mm diameter x 8 mm shoulder, M2 x 4 mm thread",
        "source_shoulder_screw_url": "https://ondrives.com/shoulder-screws/shs2-8",
        "source_insert": "M2 heat-set insert candidate; 3.2 mm OD, 3 mm length, 3 mm parent hole, recommended boss OD 6.4 mm",
        "source_insert_url": "https://engineeringhardware.com/fastener/heat-set-threaded-inserts-3/",
        "source_bearing": "SKF 623-2Z; 3 x 10 x 4 mm, 0.54 kN basic dynamic load",
        "source_bearing_url": "https://www.emarketplace.in.skf.com/deep-groove-ball-bearing/623-2z",
        "source_bearing_net_mass_g": 1,
        "motor_case_center_mm": list(MOTOR_CENTER),
        "motor_case_xyz_mm": [23, 20, 34],
        "motor_with_horns_envelope_xyz_mm": [29, 20, 34],
        "pollen_mesh_derived_output_pivot_mm": list(PIVOT),
        "output_datum_status": "cross-checked against Pollen XL330 mesh but not against inaccessible ROBOTIS STEP; exact horn bolt pattern and case mount holes remain unverified",
        "one_to_one_motor_to_finger_angle": True,
        "output_horn_pattern": "four M2 clearance holes in printed plate on 12 mm PCD, clocked at 45 degrees; source drawing specifies four 1.6 mm tapping holes on 12 mm PCD",
        "fixed_palm_center_mm": [17, 90, 106],
        "20mm_object_center_xyz_mm": [17, 109, 110],
        "20mm_object_open35_vs_fixed_mm3": round(volume(toy.intersect(rigid_forearm_palm)), 3),
        "20mm_object_open35_vs_thumb_mm3": round(volume(toy.intersect(thumb(35))), 3),
        "20mm_object_closed10_vs_thumb_mm3": round(volume(toy.intersect(thumb(10))), 3),
        "rigid_forearm_palm_print_volume_mm3": round(inner_half.volume + outer_half.volume, 1),
        "moving_thumb_print_volume_mm3": round(thumb(10).volume, 1),
        "flexible_wrist_cheek_print_volume_mm3": round(wrist_gaiter.volume, 1),
        "flexible_wrist_cheek_to_rigid_skin_overlap_mm3": round(volume(wrist_gaiter.intersect(rigid_forearm_palm)), 1),
        "flexible_wrist_cheek_to_thumb_sweep_max_overlap_mm3": round(max(volume(wrist_gaiter.intersect(thumb(degree))) for degree in range(10, 36)), 3),
        "solid_PLA_density_assumption_g_per_cm3": 1.24,
        "printed_parts_solid_PLA_mass_estimate_g": round((inner_half.volume + outer_half.volume + thumb(10).volume) / 1000 * 1.24, 1),
        "distal_assembly_nominal_mass_g_excluding_shaft_fasteners_cables": round((inner_half.volume + outer_half.volume + thumb(10).volume) / 1000 * 1.24 + 18 + 1, 1),
        "paw_skin_volume_original_mm3": round(original_hand.volume, 1),
        "paw_skin_volume_after_aperture_mm3": round(hand.volume, 1),
        "front_cheek_volume_mm3": round(front_cheek.volume, 1),
        "front_cheek_skin_attachment_mm3": round(volume(front_cheek.intersect(hand)) + volume(front_cheek.intersect(forearm_open)), 1),
        "motor_vs_original_forearm_skin_mm3": round(volume(motor.intersect(forearm)), 1),
        "motor_vs_open_forearm_skin_mm3": round(volume(motor.intersect(forearm_open)), 1),
        "motor_vs_paw_skin_mm3": round(volume(motor.intersect(hand)), 1),
        "motor_vs_joined_rigid_body_mm3": round(volume(motor.intersect(rigid_forearm_palm)), 3),
        "shaft_vs_joined_rigid_body_mm3": round(volume(shaft.intersect(rigid_forearm_palm)), 3),
        "insert_vs_moving_thumb_print_mm3": round(volume(insert.intersect(thumb(10))), 3),
        "bearing_vs_joined_rigid_body_mm3": round(volume(bearing.intersect(rigid_forearm_palm)), 3),
        "bearing_seat_to_forearm_skin_attachment_mm3": round(volume(bearing_seat.intersect(forearm_open)), 1),
        "bearing_cover_to_seat_attachment_mm3": round(volume(bearing_cover.intersect(bearing_seat)), 1),
        "bearing_inner_lip_to_seat_attachment_mm3": round(volume(bearing_inner_lip.intersect(bearing_seat)), 1),
        "bearing_axial_pocket_span_mm": [25.8, 30.1],
        "bearing_axial_envelope_span_mm": [26.0, 30.0],
        "bearing_retention_note": "Both end lips trap the outer ring after the y=100 clamshell is closed; 0.2 mm inboard and 0.1 mm outboard nominal axial clearance. Seal and inner-race rub need real fit validation.",
        "closure_post_datums_xyz_span_mm": [list(d) for d in closure_datums],
        "closure_fastener_cad": "two transverse M2 candidate clamp screws, outer clearance 2.3 mm and 4.4 mm recessed head, inner 3 mm heat-set pilot; approximate first M2x25, second M2x20, supplier and real screw/insert engagement unverified",
        "closure_posts_vs_motor_mm3": round(sum(volume(post.intersect(motor)) for post in closure_posts), 3),
        "closure_screws_vs_print_halves_mm3": round(volume(clamp_screws.intersect(inner_half)) + volume(clamp_screws.intersect(outer_half)), 3),
        "closure_screws_vs_flexible_cheek_mm3": round(volume(clamp_screws.intersect(wrist_gaiter)), 3),
        "cable_port_diameter_mm": 6,
        "cable_port_center_xy_mm": [8, 100],
        "cable_port_vs_motor_mm3": round(volume(cable_port.intersect(motor)), 3),
        "cable_route_5mm_diameter_vs_rigid_shell_mm3": round(volume(cable_route.intersect(rigid_forearm_palm)), 3),
        "fixed_frame_vs_skin_mm3": round(volume(fixed_frame.intersect(hand)) + volume(fixed_frame.intersect(forearm_open)) + volume(fixed_frame.intersect(wrist_open)), 1),
        "fixed_frame_solid_count": len(fixed_frame.solids()),
        "fixed_frame_and_skin_are_intended_one_rigid_print": True,
        "joined_rigid_body_valid_brep": rigid_forearm_palm.is_valid,
        "joined_rigid_body_substantial_solid_count": len(substantial_solids),
        "candidate_print_split_plane_y_mm": 100,
        "candidate_print_split_gap_mm": 0.4,
        "inner_half_substantial_solid_count": len(inner_solids),
        "outer_half_substantial_solid_count": len(outer_solids),
        "moving_thumb_sweep_10_to_35_deg_max_intersection_mm3": max(swept_overlap.values()),
        "moving_thumb_sweep_colliding_degrees": [degree for degree, overlap in swept_overlap.items() if overlap > 0],
        "printed_stop_overlap_9_deg_mm3": round(volume(thumb(9).intersect(close_stop)), 3),
        "printed_stop_overlap_36_deg_mm3": round(volume(thumb(36).intersect(open_stop)), 3),
        "printed_stop_to_forearm_attachment_mm3": round(volume(close_stop.intersect(forearm_open)) + volume(open_stop.intersect(forearm_open)), 1),
        "samples": samples,
        "grip_force_note": "At 50 mm tip lever, a hypothetical 0.10 Nm sustained output gives 2 N ideal tangential force, before gearbox, pad friction and finger flex. 0.10 Nm is the vendor's approximate 20%-of-stall guidance, not a measured continuous rating for this duty. The 0.52 Nm stall value must not be used as sustained pinch force.",
        "unresolved": ["physical fit and source validation of case clamp liner, M2 shell screws, horn screw and printed heat-set inserts", "physical bearing axial-retention and rubbing check", "cable connector and strain relief at elbow interface", "flexible cheek retention and deformation after printing", "full arm/floor reach and grasp-policy transfer"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("gripper_joint_viewer.html").read_text())
    return report


if __name__ == "__main__":
    result = build()
    print("paw 35 cm:", result["motor_vs_original_forearm_skin_mm3"],
          "mm3 motor/forearm shell overlap; open/closed tip gaps",
          result["samples"][0]["tip_to_fixed_face_gap_mm"],
          result["samples"][-1]["tip_to_fixed_face_gap_mm"])
