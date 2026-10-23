"""Retain competing spacer concepts, nominal fit, and sampled installation paths."""

import argparse
from pathlib import Path
import traceback
import numpy as np
from build123d import Compound, Pos, Cylinder, export_step, export_stl, import_step
from lab.yoke_spacers import candidate_parts, linked_spacers
from lab.leg_mechanism import servo_parts
from lab.joint_module import joint_parts
from lab.camera_vendor import assembly_overlap
from tools.check_camera_support import copy_source, sha, write


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", physical_status="not_run", scope=__doc__, source_hashes={}
    )
    files = [
        "lab/yoke_spacers.py",
        "lab/leg_mechanism.py",
        "lab/joint_module.py",
        "lab/camera_vendor.py",
        "tools/check_yoke_spacers.py",
        "robot/actuator-catalog.yaml",
        "docs/design/references/waveshare-st3215/ST3215.step",
    ]
    report["source_hashes"] = copy_source(output, {p: Path(p) for p in files})
    try:
        old, rings, screws, fused = candidate_parts()
        carriers = linked_spacers(rings)
        fixed, horns = servo_parts()
        clamp = joint_parts(clamped=True)
        parts = dict(
            yoke=old,
            loose_rings=rings,
            integrated_yoke=fused,
            carrier_rear=carriers[0],
            carrier_front=carriers[1],
            servo=fixed,
            horns=horns,
            screws=screws,
            clamp_lower=clamp["clamp_lower"],
            clamp_upper=clamp["clamp_upper"],
        )
        for name, shape in parts.items():
            export_step(shape, output / f"{name}.step")
            export_stl(
                shape, output / f"{name}.stl", tolerance=0.03, angular_tolerance=0.08
            )
        # Fresh STEP topology prevents coincident shared-source Boolean artifacts.
        read = {n: import_step(output / f"{n}.step") for n in parts}

        def volume(shape):
            return 0 if shape is None else sum(s.volume for s in shape.solids())

        original = Compound([read["yoke"], read["loose_rings"]])
        fusion_delta = dict(
            added_mm3=volume(read["integrated_yoke"] - original),
            missing_mm3=volume(original - read["integrated_yoke"]),
        )
        fitted = ["integrated_yoke", "carrier_rear", "carrier_front"]
        fit = []
        for n in fitted:
            for target in ["servo", "horns", "clamp_lower", "clamp_upper", "screws"]:
                fit.append(
                    dict(
                        part=n,
                        target=target,
                        overlap_mm3=assembly_overlap(read[n], read[target]),
                        gap_mm=read[n].distance_to(read[target]),
                    )
                )
        corridors = []
        for sign in [-1, 1]:
            for x, y in [(-7, 0), (7, 0), (0, -7), (0, 7)]:
                probe = Pos(x, y, sign * 21) * Cylinder(1.5, 9)
                corridors.append(
                    dict(
                        side=sign,
                        x_mm=x,
                        y_mm=y,
                        integrated_overlap_mm3=assembly_overlap(
                            probe, read["integrated_yoke"]
                        ),
                        carrier_overlap_mm3=assembly_overlap(
                            probe, read["carrier_rear" if sign < 0 else "carrier_front"]
                        ),
                    )
                )
        routes = []

        def check_route(label, moving, offsets, targets):
            for offset in offsets:
                posed = Pos(*offset) * moving
                for target in targets:
                    ov = assembly_overlap(posed, read[target])
                    routes.append(
                        dict(
                            route=label,
                            offset_mm=offset,
                            target=target,
                            overlap_mm3=ov,
                            status="fail" if ov > 1e-4 else "pass",
                        )
                    )

        check_route(
            "fused_yoke_slide_x",
            read["integrated_yoke"],
            [[float(x), 0, 0] for x in np.arange(64, -0.01, -2)],
            ["servo", "horns", "clamp_lower", "clamp_upper"],
        )
        for sign, name in [(-1, "carrier_rear"), (1, "carrier_front")]:
            check_route(
                f"{name}_axial_seat",
                read[name],
                [[0, 0, float(sign * z)] for z in np.arange(24, -0.01, -2)],
                ["servo", "horns", "clamp_lower", "clamp_upper"],
            )
        check_route(
            "yoke_after_carriers_slide_x",
            read["yoke"],
            [[float(x), 0, 0] for x in np.arange(64, -0.01, -2)],
            [
                "servo",
                "horns",
                "clamp_lower",
                "clamp_upper",
                "carrier_rear",
                "carrier_front",
            ],
        )
        tests = {
            label: (
                "fail"
                if any(r["status"] == "fail" for r in routes if r["route"] == label)
                else "pass"
            )
            for label in {r["route"] for r in routes}
        }
        report.update(
            status="partial",
            fusion_delta=fusion_delta,
            fit=fit,
            corridors=corridors,
            routes=routes,
            route_status=tests,
            candidates=[
                dict(
                    name="integrated_yoke",
                    printed_solids=len(read["integrated_yoke"].solids()),
                    status=tests["fused_yoke_slide_x"],
                ),
                dict(
                    name="linked_spacers",
                    printed_solids=len(read["yoke"].solids())
                    + sum(
                        len(read[n].solids()) for n in ["carrier_rear", "carrier_front"]
                    ),
                    status="pass"
                    if all(
                        tests[n] == "pass" for n in tests if n != "fused_yoke_slide_x"
                    )
                    else "fail",
                ),
            ],
            volumes_mm3={n: float(s.volume) for n, s in read.items()},
            added_carrier_volume_mm3=sum(
                read[n].volume for n in ["carrier_rear", "carrier_front"]
            )
            - read["loose_rings"].volume,
            not_run=[
                "continuous installation paths",
                "actual horn fastening and spline access",
                "screw threads/preload",
                "material, slicing, creep and strength",
                "full robot context and gripper integration",
                "physical assembly",
            ],
            limits=[
                "Vendor STEP is not verified exact C018 interchange.",
                "Horn seating contact is intentional; no interference threshold licenses physical compression.",
                "Module study preserves frames; robot variants and BOM remain unchanged.",
            ],
        )
        if any(v > 1e-4 for v in fusion_delta.values()):
            raise ValueError("Integrated candidate changed original occupancy")
        if any(r["overlap_mm3"] > 1e-4 for r in fit):
            raise ValueError("Seated interference")
        if any(
            r["integrated_overlap_mm3"] > 1e-4 or r["carrier_overlap_mm3"] > 1e-4
            for r in corridors
        ):
            raise ValueError("Shank corridor blocked")
    except Exception as exc:
        report.update(status="fail", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "report.json"
        }
        write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
