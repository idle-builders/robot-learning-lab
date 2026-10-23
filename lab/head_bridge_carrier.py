"""Removable printed nut/washer positioning aid; millimetre study geometry."""

from build123d import Box, Pos


def carrier(spec):
    length = spec["head_length_mm"]
    width = spec["head_width_mm"]
    z = spec["nut_seat_z_mm"]
    floor = spec["floor_thickness_mm"]
    nut_top = spec["nut_guide_top_z_mm"]
    washer_top = spec["washer_guide_top_z_mm"]
    if not (
        0 < floor
        and 0
        < spec["screw_slot_width_mm"]
        < spec["nut_slot_width_mm"]
        < spec["washer_slot_width_mm"]
        < width
        and z < nut_top < washer_top
    ):
        raise ValueError("Carrier dimensions do not leave nested support/guides")
    body = Pos(0, 0, z - floor / 2) * Box(length, width, floor)
    # Open front (+X) lets the tool slide rearward after the screw engages.
    for bottom, top, slot, back in [
        (z, nut_top, spec["nut_slot_width_mm"], spec["nut_slot_back_x_mm"]),
        (
            nut_top,
            washer_top,
            spec["washer_slot_width_mm"],
            spec["washer_slot_back_x_mm"],
        ),
    ]:
        layer = Pos(0, 0, (bottom + top) / 2) * Box(length, width, top - bottom)
        front = length / 2 + 1
        layer -= Pos((back + front) / 2, 0, (bottom + top) / 2) * Box(
            front - back, slot, top - bottom + 2
        )
        body += layer
    front = length / 2 + 1
    back = -spec["screw_slot_width_mm"] / 2
    body -= Pos((back + front) / 2, 0, z - floor / 2) * Box(
        front - back, spec["screw_slot_width_mm"], floor + 2
    )
    handle = spec["handle_length_mm"]
    height = spec["handle_thickness_mm"]
    body += Pos(-length / 2 - handle / 2 + 0.1, 0, z - floor + height / 2) * Box(
        handle + 0.2, spec["handle_width_mm"], height
    )
    if not body.is_valid or len(body.solids()) != 1:
        raise ValueError("Carrier is not one connected valid solid")
    return body
