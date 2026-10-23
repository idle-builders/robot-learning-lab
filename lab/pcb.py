"""A small power-connector coupon proves local PCB checks and mechanical exchange.

This is explicitly NOT the robot controller board. No current rating is claimed.
"""

from pathlib import Path
import json
import subprocess
import uuid
from .electrical import find_tool


def uid(label):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "robot-learning-lab/" + label))


def board_source(robot, fault=False):
    d = robot.config.dimensions
    x = d.board_width_m * 1000
    y = d.board_length_m * 1000
    if min(x, y) < 30:
        raise ValueError("PCB coupon needs a board envelope of at least 30 mm per side")
    lines = [
        '(kicad_pcb (version 20241229) (generator "robot-learning-lab")',
        '(general (thickness 1.6)) (paper "A4")',
        '(layers (0 "F.Cu" signal) (31 "B.Cu" signal) (37 "F.SilkS" user "f.silkscreen") (44 "Edge.Cuts" user) (47 "F.CrtYd" user "f.courtyard"))',
        '(setup (pad_to_mask_clearance 0)) (net 0 "") (net 1 "VBAT") (net 2 "GND")',
    ]
    for i, (a, b) in enumerate(
        [((0, 0), (x, 0)), ((x, 0), (x, y)), ((x, y), (0, y)), ((0, y), (0, 0))]
    ):
        lines.append(
            f'(gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (stroke (width 0.05) (type default)) (layer "Edge.Cuts") (uuid "{uid(f"edge{i}")}"))'
        )
    for ref, px in [("J1", 10), ("J2", x - 10)]:
        lines.append(f'''(footprint "RobotLab:PowerCoupon_5.08" (layer "F.Cu") (uuid "{uid(ref)}") (at {px} 10)
          (property "Reference" "{ref}" (at 0 -3) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))
          (property "Value" "INTERFACE_TEST" (at 0 9) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))
          (attr through_hole)
          (fp_rect (start -2.5 -2.5) (end 2.5 7.58) (stroke (width 0.05) (type default)) (fill none) (layer "F.CrtYd"))
          (pad "1" thru_hole circle (at 0 0) (size 2.4 2.4) (drill 1.2) (layers "*.Cu" "*.Mask") (net 1 "VBAT"))
          (pad "2" thru_hole circle (at 0 5.08) (size 2.4 2.4) (drill 1.2) (layers "*.Cu" "*.Mask") (net 2 "GND")))''')
    for net, py in [(1, 10), (2, 15.08)]:
        lines.append(
            f'(segment (start 10 {py}) (end {x - 10} {py}) (width 1.5) (layer "F.Cu") (net {net}) (uuid "{uid(f"trace{net}")}"))'
        )
    if fault:
        lines.append(
            f'(segment (start 10 10) (end 10 15.08) (width 1.5) (layer "F.Cu") (net 1) (uuid "{uid("short")}"))'
        )
    lines.append(")")
    return "\n".join(lines)


def pcb_check(robot, output):
    output = Path(output).resolve()
    exe = find_tool("kicad-cli")
    if not exe:
        return {"status": "not_run", "reason": "KiCad CLI unavailable"}
    outcomes = {}
    for fault, name in [
        (False, "interface-coupon"),
        (True, "interface-coupon-shorted"),
    ]:
        board = output / f"{name}.kicad_pcb"
        board.write_text(board_source(robot, fault))
        report = output / f"{name}-drc.json"
        result = subprocess.run(
            [
                exe,
                "pcb",
                "drc",
                "--format",
                "json",
                "--severity-error",
                "--exit-code-violations",
                "-o",
                str(report),
                str(board),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        if not report.exists():
            raise RuntimeError(f"KiCad did not produce {report.name}: {result.stderr}")
        details = json.loads(report.read_text())
        errors = [
            v for v in details.get("violations", []) if v.get("severity") == "error"
        ]
        outcomes[name] = {
            "exit_code": result.returncode,
            "error_count": len(errors),
            "report": report.name,
        }
        # Exit 5 is KiCad's DRC violations code. Other failures must not count as detection.
        expected = (
            (result.returncode == 5 and len(errors) > 0)
            if fault
            else result.returncode == 0 and not errors
        )
        outcomes[name]["expected_result"] = expected
    exported = subprocess.run(
        [
            exe,
            "pcb",
            "export",
            "step",
            "-o",
            str(output / "interface-coupon.step"),
            str(output / "interface-coupon.kicad_pcb"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (output / "pcb-step.log").write_text(exported.stdout + exported.stderr)
    return {
        "status": "pass"
        if all(x["expected_result"] for x in outcomes.values())
        and exported.returncode == 0
        else "fail",
        "scope": "power connector test coupon only; not a controller PCB; error-level DRC only",
        "cases": outcomes,
        "step_export_exit_code": exported.returncode,
        "dimensions_mm": [
            robot.config.dimensions.board_width_m * 1000,
            robot.config.dimensions.board_length_m * 1000,
        ],
    }
