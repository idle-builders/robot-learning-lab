"""Candidate ST3215 clamp-to-torso load path; millimetres in torso coordinates.

Nominal CAD fit only. No strength, creep, preload or fall-survival qualification.
"""

from build123d import Box, Cylinder, Pos, Rot, RegularPolygon, extrude, Compound

CLAMP_FACE_MM = 15.365
UPPER_BOLT_Z_MM = 38.615
BEAM_THICKNESS_MM = 6
BEAM_HEIGHT_MM = 12


def bolt_positions(d):
    return [
        (sign * d.hip_spacing_m * 500 + dy, UPPER_BOLT_Z_MM - d.hip_drop_m * 1000)
        for sign in [-1, 1]
        for dy in [-8, 8]
    ]


def socket_cuts(d):
    return [
        Pos(0, sign * d.hip_spacing_m * 500, d.hip_socket_height_m * 500 - 1)
        * Box(
            d.hip_socket_depth_m * 1000,
            d.hip_socket_width_m * 1000,
            d.hip_socket_height_m * 1000 + 2,
        )
        for sign in [-1, 1]
    ]


def mounting_stock(d):
    z = UPPER_BOLT_Z_MM - d.hip_drop_m * 1000
    beam = Pos(-CLAMP_FACE_MM - BEAM_THICKNESS_MM / 2, 0, z) * Box(
        BEAM_THICKNESS_MM, d.hip_spacing_m * 1000 + 30, BEAM_HEIGHT_MM
    )
    # Two ribs join the crossbeam to the front shell, between the motor openings.
    end = d.torso_depth_m * 500
    start = -CLAMP_FACE_MM - 1
    for y in [-16, 16]:
        beam += Pos((start + end) / 2, y, z) * Box(end - start, 5, BEAM_HEIGHT_MM)
    for y, z in bolt_positions(d):
        beam -= (
            Pos(-CLAMP_FACE_MM - BEAM_THICKNESS_MM / 2, y, z)
            * Rot(0, 90, 0)
            * Cylinder(1.65, 10)
        )
    return beam


def fit_front_shell(front, outer, d, centered=True):
    if not d.hip_socket_width_m:
        return front
    offset = Pos(0, 0, -d.torso_height_m * 500 if centered else 0)
    for cut in socket_cuts(d):
        front -= offset * cut
    # Rear access to upper screw heads; capture nuts in the front clamp halves.
    for y, z in bolt_positions(d):
        front -= offset * Pos(-46.75, y, z) * Rot(0, 90, 0) * Cylinder(4, 66.5)
    front += (offset * mounting_stock(d)) & outer
    if not front.is_valid or len(front.solids()) != 1:
        raise ValueError("pelvis stock is not connected to front shell")
    return front


def clamp_stack_specs(is_hip):
    """Two stacks per height, at lateral coordinates -8 and +8 mm."""
    for z in [UPPER_BOLT_Z_MM, -13.615]:
        yield (
            z,
            6 if is_hip and z > 0 else 0 if is_hip else 3,
            (40 if z > 0 else 35) if is_hip else 40,
            0,
        )


def front_nut_pockets():
    """Load M3 nuts before inserting the clamped motors in the torso."""
    hexagon = extrude(RegularPolygon(2.85, 6, major_radius=False), amount=3)
    return Compound(
        [
            Pos(CLAMP_FACE_MM - 2.6, y, UPPER_BOLT_Z_MM) * Rot(0, 90, 0) * hexagon
            for y in [-8, 8]
        ]
    )


def rear_tool_spaces(d):
    # Conservative 6 mm diameter tool shaft; cap/board assembly removed.
    tip = -CLAMP_FACE_MM - BEAM_THICKNESS_MM - 0.5 - 3
    return [
        Pos((-80 + tip) / 2, y, z) * Rot(0, 90, 0) * Cylinder(3, tip + 80)
        for y, z in bolt_positions(d)
    ]


def clamp_hardware(is_hip):
    """Four nominal M3 stacks, in a joint's parent-aligned zero frame.

    Threaded interfaces overlap intentionally inside each compound stack.
    Screw head/washer/nut envelopes are sourced; threads and drive sockets omitted.
    """
    parts = []
    for z, plate, length, spacer in clamp_stack_specs(is_hip):
        if is_hip and z > 0:
            face = -CLAMP_FACE_MM - BEAM_THICKNESS_MM
            under_head = face - 0.5
            shank = (
                Pos(under_head + length / 2, 0, 0)
                * Rot(0, 90, 0)
                * Cylinder(1.5, length)
            )
            head = Pos(under_head - 1.5, 0, 0) * Rot(0, 90, 0) * Cylinder(2.75, 3)
            washer = Cylinder(3.5, 0.5) - Cylinder(1.6, 1)
            nut = extrude(RegularPolygon(2.75, 6, major_radius=False), amount=2.4)
            nut -= Pos(0, 0, 1.2) * Cylinder(1.25, 4)
            stack = [
                shank,
                head,
                Pos(face - 0.25, 0, 0) * Rot(0, 90, 0) * washer,
                Pos(CLAMP_FACE_MM - 2.6, 0, 0) * Rot(0, 90, 0) * nut,
            ]
            for y in [-8, 8]:
                parts.extend((Pos(0, y, z) * p).solids() for p in stack)
            continue
        front = CLAMP_FACE_MM + plate
        under_head = front + 0.5
        shank = (
            Pos(under_head - length / 2, 0, 0) * Rot(0, 90, 0) * Cylinder(1.5, length)
        )
        head = Pos(under_head + 1.5, 0, 0) * Rot(0, 90, 0) * Cylinder(2.75, 3)
        washer = Cylinder(3.5, 0.5) - Cylinder(1.6, 1)
        stack = [shank, head, Pos(front + 0.25, 0, 0) * Rot(0, 90, 0) * washer]
        rear = -CLAMP_FACE_MM - spacer
        stack.append(Pos(rear - 0.25, 0, 0) * Rot(0, 90, 0) * washer)
        if spacer:
            ring = Cylinder(3.5, spacer) - Cylinder(1.65, spacer + 1)
            stack.append(Pos(-CLAMP_FACE_MM - spacer / 2, 0, 0) * Rot(0, 90, 0) * ring)
        nut = extrude(RegularPolygon(2.75, 6, major_radius=False), amount=2.4)
        nut -= Pos(0, 0, 1.2) * Cylinder(1.25, 4)
        stack.append(Pos(rear - 0.5 - 2.4, 0, 0) * Rot(0, 90, 0) * nut)
        for y in [-8, 8]:
            parts.extend((Pos(0, y, z) * p).solids() for p in stack)
    return Compound([s for group in parts for s in group])


def hardware_bom(robot, production_units=None):
    """Partial clamp procurement, derived from the same stack recipe as CAD."""
    from collections import Counter
    from pathlib import Path
    import yaml

    if (
        not robot.config.dimensions.hip_socket_width_m
        or robot.config.morphology != "wheeled"
    ):
        raise ValueError("clamp BOM requires the wheeled pelvis candidate")
    catalog = yaml.safe_load(Path("robot/fastener-catalog.yaml").read_text())
    counts = Counter()
    for link in robot.links:
        if link.name not in {
            f"{s}_{j}" for s in ["left", "right"] for j in ["thigh", "shin", "wheel"]
        }:
            continue
        for _, _, length, spacer in clamp_stack_specs(link.name.endswith("thigh")):
            counts[f"screw_m3_{length}"] += 2
            counts["washer_m3"] += (
                2 if link.name.endswith("thigh") and length == 40 else 4
            )
            counts["nut_m3"] += 2
            if spacer:
                counts["spacer_m3_3"] += 2
    if robot.config.dimensions.shoulder_mount_thickness_m:
        from .shoulder_mount import bolt_positions as shoulder_bolts

        quantity = len(shoulder_bolts(robot.config.dimensions))
        for part in ["screw_m3_40", "washer_m3", "nut_m3"]:
            counts[part] += quantity
    if robot.config.dimensions.neck_mount_thickness_m:
        from .neck_mount import hardware_counts

        counts.update(hardware_counts())
    units = production_units or robot.config.production_units
    rows = []
    for part, qty in sorted(counts.items()):
        record = catalog["parts"][part]
        eligible = [
            price
            for threshold, price in record["tiers_gbp"]
            if qty * units >= threshold
        ]
        price = eligible[-1] if eligible else None
        rows.append(
            dict(
                part=part,
                per_robot=qty,
                order_quantity=qty * units,
                unit_price_gbp=price,
                per_robot_gbp=round(qty * price, 4) if price is not None else None,
                **record,
            )
        )
    return dict(
        status="partial",
        scope=catalog["scope"],
        production_units=units,
        price_basis=catalog["price_basis"],
        retrieved=catalog["retrieved"],
        rows=rows,
        known_subtotal_per_robot_gbp=round(
            sum(r["per_robot_gbp"] or 0 for r in rows), 4
        ),
        unpriced_parts=[r["part"] for r in rows if r["unit_price_gbp"] is None],
        exclusions="remaining robot hardware, custom structure, assembly, shipping, duties and taxes; no USD conversion or whole-product cost update",
    )
