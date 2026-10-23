"""Supplemental small-intrusion controls from independently measured gap witnesses."""

from pathlib import Path
import argparse
import traceback
import numpy as np
from build123d import Pos, import_step
from lab.leg_mechanism import overlap
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(studies, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path("tools/check_head_cap_mount_detector.py"),
        Path("tools/check_camera_support.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("lab/leg_mechanism.py"),
    ]
    report = dict(
        status="fail",
        scope=__doc__,
        source_hashes=copy_source(output, {str(p): p for p in sources}),
        source_bindings={},
        cases=[],
    )
    try:
        for study in studies:
            prior = verify_report(study)
            report["source_bindings"][str(study / "report.json")] = sha(
                study / "report.json"
            )
            for row in prior["cases"]:
                case = dict(
                    study=str(study),
                    pose=row["pose"],
                    cover=row["cover"],
                    target=row["target"],
                    status="not_run",
                    probes=[],
                )
                report["cases"].append(case)
                if "distance_mm" not in row:
                    continue
                folder = study / row["pair_directory"]
                a, b = [import_step(folder / f"{n}.step") for n in ["stock", "target"]]
                for p in [
                    folder / "stock.step",
                    folder / "target.step",
                    folder / "distance.json",
                ]:
                    report["source_bindings"][str(p)] = sha(p)
                direction = np.array(row["target_point_head_mm"]) - np.array(
                    row["stock_point_head_mm"]
                )
                distance = float(np.linalg.norm(direction))
                if distance <= 1e-6:
                    raise ValueError(
                        "Positive separation required for intrusion control"
                    )
                for intrusion in [0.25, 0.5, 1.0]:
                    shift = direction / distance * (distance + intrusion)
                    volume = overlap(Pos(*shift) * a, b)
                    case["probes"].append(
                        dict(
                            intrusion_beyond_witness_mm=intrusion,
                            displacement_head_mm=shift.tolist(),
                            intersection_mm3=volume,
                            status="pass" if volume > 0.02 else "fail",
                        )
                    )
                case["status"] = (
                    "pass"
                    if any(p["status"] == "pass" for p in case["probes"])
                    else "fail"
                )
                print(
                    case["study"],
                    case["pose"],
                    case["target"],
                    case["status"],
                    flush=True,
                )
        report["status"] = (
            "pass"
            if report["cases"] and all(c["status"] == "pass" for c in report["cases"])
            else "fail"
        )
        report["limits"] = [
            "Each pair must detect at least one of the three deliberate small intrusions; this does not qualify installed-position clearance.",
            "Original motion-study controls and failures remain unchanged.",
            "Witnesses and distances are nominal CAD, not physical measurements.",
        ]
        write(output / "report.json", report)
        return report
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.studies, args.output)
    raise SystemExit(1 if outcome["status"] == "fail" else 0)
