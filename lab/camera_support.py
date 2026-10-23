"""Profile-selected camera support candidates, mm, optical front at origin, +X out."""

import hashlib
import math
from pathlib import Path

from build123d import Box, Cylinder, Pos, Rot, RegularPolygon, extrude
import yaml

from .vision import load_vision


def specification(profile="drawing"):
    paths = {
        "drawing": "electronics/camera-support.yaml",
        "vendor_step": "electronics/camera-support-vendor.yaml",
    }
    if profile not in paths:
        raise ValueError(f"unknown camera support profile: {profile}")
    spec = yaml.safe_load(Path(paths[profile]).read_text())
    spec["profile"] = profile
    spec["manifest_path"] = paths[profile]
    if profile == "vendor_step":
        if (
            hashlib.sha256(Path(spec["source_vendor"]).read_bytes()).hexdigest()
            != spec["vendor_sha256"]
        ):
            raise ValueError("camera vendor geometry changed")
    if (
        hashlib.sha256(Path(spec["source_drawing"]).read_bytes()).hexdigest()
        != spec["source_sha256"]
    ):
        raise ValueError("camera support drawing changed")
    if (
        spec["schema_version"] != 1
        or not 0
        < spec["pcb_thickness_mm"]
        < spec["pcb_rear_depth_mm"]
        < spec["carrier_front_depth_mm"]
        < 15
    ):
        raise ValueError("invalid camera support depth stack")
    for key, value in spec.items():
        if key.endswith("_mm") and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"invalid support dimension: {key}")
    for key, value in spec["fastener"].items():
        if key.endswith("_mm") and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"invalid fastener dimension: {key}")
    if (
        spec["support_diameter_mm"] <= 2.2
        or spec["fastener"]["nominal_thread_mm"] >= 2.2
    ):
        raise ValueError("support requires annular stock and screw clearance")
    return spec


def centers():
    camera = load_vision().camera
    ox, oy = camera.optic_datum_drawing_mm
    return [(x - ox, y - oy) for x, y in camera.mounting_holes_drawing_mm]


def cylinder(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(radius, length)


def module_parts(spec=None):
    """Purchased vendor assembly or explicitly simplified drawing allocation."""
    spec = spec or specification()
    if spec["profile"] == "vendor_step":
        from .camera_vendor import model

        return {"camera_vendor_module": model(spec)[0]}
    cam = load_vision().camera
    width, height = cam.drawing_planform_mm
    ox, oy = cam.optic_datum_drawing_mm
    rear, thick = spec["pcb_rear_depth_mm"], spec["pcb_thickness_mm"]
    pcb = Pos(-rear + thick / 2, width / 2 - ox, height / 2 - oy) * Box(
        thick, width, height
    )
    for y, z in centers():
        pcb -= cylinder(
            -rear + thick / 2, y, z, cam.mounting_hole_diameter_mm / 2, thick + 2
        )
    front = rear - thick
    lens = Pos(-front / 2, 0, 0) * Box(
        front, spec["lens_width_mm"], spec["lens_width_mm"]
    )
    depth = spec["rear_component_depth_mm"] - rear
    # Drawing's upper connector; centered laterally is a stated interpretation.
    connector = Pos(
        -rear - depth / 2, 0, height - oy - spec["rear_connector_height_mm"] / 2
    ) * Box(depth, spec["rear_connector_width_mm"], spec["rear_connector_height_mm"])
    return dict(
        camera_pcb=pcb,
        camera_lens_allocation=lens,
        camera_rear_connector_allocation=connector,
    )


def carrier_parts(spec=None):
    from .camera_mount import cassette_local

    spec = spec or specification()
    base = cassette_local()
    plane = -spec["carrier_front_depth_mm"]
    carrier = base & (Pos(plane - 50, 0, 0) * Box(100, 100, 100))
    bezel = base & (Pos(plane + 50, 0, 0) * Box(100, 100, 100))
    bezel -= cylinder(2, 0, 0, spec["aperture_diameter_mm"] / 2, 6)
    length = spec["carrier_front_depth_mm"] - spec["pcb_rear_depth_mm"]
    for y, z in centers():
        post = cylinder(
            plane + length / 2, y, z, spec["support_diameter_mm"] / 2, length
        )
        post -= cylinder(plane + length / 2, y, z, 1.1, length + 2)
        carrier += post
    for name, shape in [("carrier", carrier), ("bezel", bezel)]:
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f"{name} must be one connected valid solid")
    return dict(camera_rear_carrier=carrier, camera_front_bezel=bezel)


def hardware_parts(spec=None):
    spec = spec or specification()
    f = spec["fastener"]
    seat = -spec["pcb_rear_depth_mm"] + spec["pcb_thickness_mm"]
    result = {}
    for i, (y, z) in enumerate(centers()):
        shaft = cylinder(
            seat - f["length_mm"] / 2, y, z, f["nominal_thread_mm"] / 2, f["length_mm"]
        )
        head = cylinder(
            seat + f["head_height_mm"] / 2,
            y,
            z,
            f["head_diameter_mm"] / 2,
            f["head_height_mm"],
        )
        result[f"camera_board_screw_{i}"] = shaft + head
        # Nut flat bearing face rests on the carrier's rear plane x=-15.
        nut = (
            Pos(-15 - f["nut_height_mm"], y, z)
            * Rot(0, 90, 0)
            * extrude(
                RegularPolygon(f["nut_across_flats_mm"] / 2, 6, major_radius=False),
                amount=f["nut_height_mm"],
            )
        )
        # Clearance-hole proxy: thread contact/engagement is checked analytically.
        nut -= cylinder(
            -15 - f["nut_height_mm"] / 2, y, z, 1.01, f["nut_height_mm"] + 2
        )
        result[f"camera_board_nut_{i}"] = nut
    return result


def combined_carrier(spec=None):
    parts = carrier_parts(spec)
    return parts["camera_rear_carrier"] + parts["camera_front_bezel"]


def interface(spec=None):
    s = spec or specification()
    f = s["fastener"]
    grip = 15 - s["pcb_rear_depth_mm"] + s["pcb_thickness_mm"]
    return dict(
        schema_version=1,
        owner="head",
        datum="optical front, +X outward, mm",
        drawing_to_head=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        depth_interpretation=s.get(
            "depth_basis", "official drawing dimension chain; unmeasured"
        ),
        thread_scope="nominal tip projection beyond nut; no thread runout or tolerance qualification",
        pcb_front_x_mm=-s["pcb_rear_depth_mm"] + s["pcb_thickness_mm"],
        pcb_rear_x_mm=-s["pcb_rear_depth_mm"],
        carrier_split_x_mm=-s["carrier_front_depth_mm"],
        support_length_mm=s["carrier_front_depth_mm"] - s["pcb_rear_depth_mm"],
        grip_length_mm=grip,
        nut_full_engagement_margin_mm=f["length_mm"] - grip - f["nut_height_mm"],
        camera_hole_centers_yz_mm=centers(),
        source=s,
        not_run=[
            "measured populated geometry and tolerances",
            "clamp torque/PCB strain",
            "thread and material strength",
            "vibration retention",
            "actual flex mating/bend",
            "new assembly mass and dynamics",
        ],
    )


def visual_recipes(spec=None):
    """Cheap primitive fallbacks; CAD uses the exact shared candidate recipes."""
    s = spec or specification()
    f = s["fastener"]
    cam = load_vision().camera
    rear = s["pcb_rear_depth_mm"]
    thick = s["pcb_thickness_mm"]
    front = rear - thick
    h = cam.drawing_planform_mm[1]
    cy = h / 2 - cam.optic_datum_drawing_mm[1]
    data = [
        (
            "camera_rear_carrier",
            (-11.65, 0, -2.4),
            (6.7, 46, 28.8),
            (0.28, 0.38, 0.36, 1),
        ),
        (
            "camera_front_bezel",
            (-4.95, 0, -2.4),
            (15.9, 46, 28.8),
            (0.055, 0.08, 0.085, 1),
        ),
        (
            "camera_pcb",
            (-rear + thick / 2, 0, cy),
            (thick, 25, h),
            (0.15, 0.45, 0.23, 1),
        ),
        (
            "camera_lens_allocation",
            (-front / 2, 0, 0),
            (front, s["lens_width_mm"], s["lens_width_mm"]),
            (0.07, 0.08, 0.09, 1),
        ),
        (
            "camera_rear_connector_allocation",
            (
                -(s["rear_component_depth_mm"] + rear) / 2,
                0,
                h - cam.optic_datum_drawing_mm[1] - s["rear_connector_height_mm"] / 2,
            ),
            (
                s["rear_component_depth_mm"] - rear,
                s["rear_connector_width_mm"],
                s["rear_connector_height_mm"],
            ),
            (0.65, 0.58, 0.42, 1),
        ),
    ]
    if s["profile"] == "vendor_step":
        module = module_parts(s)["camera_vendor_module"]
        box = module.bounding_box()
        data = data[:2] + [
            (
                "camera_vendor_module",
                tuple(box.center()),
                tuple(box.size),
                (0.12, 0.23, 0.17, 1),
            )
        ]
    # Carrier depth changes with the selected support plane.
    rear_center = -(15 + rear) / 2
    data[0] = (data[0][0], (rear_center, 0, -2.4), (15 - rear, 46, 28.8), data[0][3])
    for i, (y, z) in enumerate(centers()):
        data += [
            (
                f"camera_board_screw_{i}",
                (-front + (f["head_height_mm"] - f["length_mm"]) / 2, y, z),
                (
                    f["length_mm"] + f["head_height_mm"],
                    f["head_diameter_mm"],
                    f["head_diameter_mm"],
                ),
                (0.55, 0.56, 0.58, 1),
            ),
            (
                f"camera_board_nut_{i}",
                (-15 - f["nut_height_mm"] / 2, y, z),
                (
                    f["nut_height_mm"],
                    f["nut_across_flats_mm"] * 1.155,
                    f["nut_across_flats_mm"] * 1.155,
                ),
                (0.55, 0.56, 0.58, 1),
            ),
        ]
    return data


def part_local(name, spec=None):
    if name in ("camera_rear_carrier", "camera_front_bezel"):
        return carrier_parts(spec)[name]
    if name in (
        "camera_vendor_module",
        "camera_pcb",
        "camera_lens_allocation",
        "camera_rear_connector_allocation",
    ):
        return module_parts(spec)[name]
    return hardware_parts(spec)[name]


def aperture_check(bezel=None):
    """Conservative circular diagonal-FOV envelope; not calibrated optical rays."""
    from build123d import Cone
    from .leg_mechanism import overlap

    camera = load_vision().camera
    radius = 2.375  # Half of the drawing's 4.75 mm front opening.
    slope = math.hypot(
        math.tan(math.radians(camera.intrinsics.horizontal_fov_deg / 2)),
        math.tan(math.radians(camera.intrinsics.vertical_fov_deg / 2)),
    )
    length = 3.01
    beam = (
        Pos(length / 2, 0, 0)
        * Rot(0, 90, 0)
        * Cone(radius, radius + length * slope, length)
    )
    if bezel is None:
        bezel = carrier_parts()["camera_front_bezel"]
    intrusion = overlap(beam, bezel)
    return dict(
        status="pass" if intrusion < 0.02 else "fail",
        overlap_mm3=intrusion,
        entrance_radius_mm=radius,
        front_radius_mm=radius + 3 * slope,
        scope="assumed entrance disk at optical-front datum plus circular diagonal-FOV envelope; not physical vignetting calibration",
    )
