"""Candidate four-post buck-board mount, in torso millimetres.

The mechanical interface is checked against the compiled board before use.
Posts contain heat-insertion pilot holes; printed material/retention unqualified.
"""

import numpy as np
from decimal import Decimal
from pathlib import Path
import yaml
from build123d import Cylinder, Pos, Rot


BOARD_OUTLINE_MM = [38, 34]
HOLES_PCB_MM = [[3, 3], [35, 3], [3, 31], [35, 31]]
POST_RADIUS_MM = 4.0
# Ruthex RX-M2.5x5.7; datasheet linked in docs/research/logic-mount-hardware.md.
INSERT_DIAMETER_MM = 4.6
INSERT_LENGTH_MM = 5.7
PILOT_DIAMETER_MM = 4.0
# Supplier minimum L+1; add 0.2 mm nominal tip space as a project assumption.
PILOT_DEPTH_MM = 6.9
SCREW_LENGTH_MM = 6.0
SCREW_HEAD_DIAMETER_MM = 4.5
SCREW_HEAD_HEIGHT_MM = 2.5
PCB_THICKNESS_MM = 1.6


def board_config(d):
    return {
        "schema_version": 1,
        "normal": "rearward",
        "position_torso_mm": [
            d.logic_board_x_m * 1000,
            d.logic_board_y_m * 1000,
            d.logic_board_z_m * 1000,
        ],
    }


def centers(d):
    x, z = d.logic_board_x_m * 1000, d.logic_board_z_m * 1000
    return [
        [x, d.logic_board_y_m * 1000 + 19 - px, z + 17 - py] for px, py in HOLES_PCB_MM
    ]


def validate_board(contract):
    if (
        contract["outline_mm"] != BOARD_OUTLINE_MM
        or sorted(contract["mounting_holes_mm"]) != sorted(HOLES_PCB_MM)
        or not np.isclose(contract["mounting_hole_diameter_mm"], 2.7)
        or not np.isclose(contract["thickness_mm"], PCB_THICKNESS_MM)
    ):
        raise ValueError("compiled board differs from four-post mounting interface")


def cylinder_x(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(radius, length)


def fit_front_shell(front, outer, d):
    if not d.logic_mount_enabled:
        return front
    offset = Pos(0, 0, -d.torso_height_m * 500)
    end = d.torso_depth_m * 500
    for x, y, z in centers(d):
        stock = cylinder_x((x + end) / 2, y, z, POST_RADIUS_MM, end - x)
        front += (offset * stock) & outer
        pilot = cylinder_x(
            x + PILOT_DEPTH_MM / 2 - 0.01,
            y,
            z,
            PILOT_DIAMETER_MM / 2,
            PILOT_DEPTH_MM + 0.02,
        )
        front -= offset * pilot
    if not front.is_valid or len(front.solids()) != 1:
        raise ValueError("logic mounting posts are not connected to front shell")
    return front


def hardware(d):
    """Nominal screw/head and maximum insert envelopes; no helical threads."""
    screws, inserts = [], []
    for x, y, z in centers(d):
        head_seat = x - PCB_THICKNESS_MM
        screw = cylinder_x(head_seat + SCREW_LENGTH_MM / 2, y, z, 1.25, SCREW_LENGTH_MM)
        screw += cylinder_x(
            head_seat - SCREW_HEAD_HEIGHT_MM / 2,
            y,
            z,
            SCREW_HEAD_DIAMETER_MM / 2,
            SCREW_HEAD_HEIGHT_MM,
        )
        insert = cylinder_x(
            x + INSERT_LENGTH_MM / 2, y, z, INSERT_DIAMETER_MM / 2, INSERT_LENGTH_MM
        )
        insert -= cylinder_x(x + INSERT_LENGTH_MM / 2, y, z, 1.25, INSERT_LENGTH_MM + 1)
        screws.append(screw)
        inserts.append(insert)
    return screws, inserts


def mount_bom(production_units, per_robot=4):
    if type(per_robot) is not int or per_robot <= 0:
        raise ValueError("positive hardware count required")
    catalog = yaml.safe_load(Path("robot/fastener-catalog.yaml").read_text())
    items = []
    for key in ("screw_m2p5_6", "insert_m2p5_5p7"):
        part = catalog["parts"][key]
        quantity = production_units * per_robot
        eligible = [(q, price) for q, price in part["tiers_gbp"] if q <= quantity]
        price = Decimal(str(max(eligible)[1])) if eligible else None
        items.append(
            {
                "part": part["procurement_candidate"],
                "per_robot": per_robot,
                "batch_quantity": quantity,
                "unit_price_gbp": float(price) if price is not None else None,
                "per_robot_gbp": float(per_robot * price)
                if price is not None
                else None,
                "source": part["source"],
            }
        )
    return {
        "production_units": production_units,
        "items": items,
        "complete_mount_cost": "unquoted",
        "exclusions": [
            "insert volume price",
            "post material/process",
            "insertion/assembly/test",
            "service tools",
            "VAT/logistics",
        ],
    }
