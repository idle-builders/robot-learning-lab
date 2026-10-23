"""Present the bounded Scout posture comparison with retained dynamics and CAD."""

import argparse
import base64
from copy import deepcopy
import gzip
import html
import json
from pathlib import Path

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_forearm_accounted import (
    NORMAL_BALANCE,
    verify_archive,
    bind_recording_inputs,
)
from tools.review_integral_head_service import embedded_dataset


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def read(p):
        p = Path(p)
        bindings[str(p)] = sha(p)
        return json.loads(p.read_text())

    def verified(folder):
        folder = Path(folder)
        value = verify_report(folder)
        read(folder / "report.json")
        return value

    selection = verified("artifacts/scout-posture-selection-v1")
    static = verified("artifacts/scout-posture-cad-v1")
    observed = verified("artifacts/scout-posture-observed-cad-v1")
    verified("artifacts/shin-cover-review-v1")
    assembly = verified("artifacts/scout-shin-cover-composed-v2")
    verified("artifacts/scout-shin-covered-mass-v1")
    rows = []
    for profile, kind in [
        ("nominal", "normal"),
        ("lower", "bounds"),
        ("upper", "bounds"),
        ("nominal", "stress"),
    ]:
        new = Path(f"artifacts/scout-covered-rest-{profile}-{kind}-v1")
        old = Path(
            "artifacts/scout-covered-rest-nominal-baseline-stress-v1"
            if kind == "stress"
            else f"artifacts/scout-shin-covered-{profile}-balance-v1"
        )
        reports = [read(p / "balance-report.json") for p in [old, new]]
        expected = read(
            Path(f"artifacts/scout-shin-covered-mass-v1/{profile}/interfaces.json")
        )
        for folder, report in zip([old, new], reports, strict=True):
            verify_archive(folder, report, bindings)
            if (
                read(folder / "interfaces.json") != expected
                or report["mass_model"] != expected["mass_model"]
            ):
                raise ValueError("Mass/interface changed")
        for source in [
            "lab/balance.py",
            "lab/io.py",
            "lab/control.py",
            "lab/imu_sim.py",
            "lab/imu.py",
            "lab/session.py",
            "lab/hardware.py",
            "lab/transport.py",
        ]:
            if (
                reports[0]["source_hashes"][source]
                != reports[1]["source_hashes"][source]
            ):
                raise ValueError(f"Implementation changed: {source}")
        controllers = [deepcopy(r["controller"]) for r in reports]
        old_targets = controllers[0].pop("arm_rest_targets_rad")
        new_targets = controllers[1].pop("arm_rest_targets_rad")
        if (
            controllers[0] != controllers[1]
            or new_targets != selection["selected_pose_rad"]
        ):
            raise ValueError("Controller changed beyond selected posture")
        if {
            k
            for k in old_targets.keys() | new_targets.keys()
            if old_targets.get(k, 0) != new_targets.get(k, 0)
        } != {"left_upper_arm_joint", "right_upper_arm_joint"}:
            raise ValueError("Unexpected changed joints")
        for before, after in zip(reports[0]["cases"], reports[1]["cases"], strict=True):
            if before["case"] != after["case"] or before["input"] != after["input"]:
                raise ValueError("Case conditions differ")
            peaks = []
            for folder, c in [(old, before), (new, after)]:
                bind_recording_inputs(folder / c["case"], "balance", bindings)
                trace = folder / c["case"] / "trajectory.jsonl"
                bindings[str(trace)] = sha(trace)
                with trace.open() as stream:
                    traces = (json.loads(line) for line in stream if line.strip())
                    peaks.append(
                        max(
                            (
                                abs(r["pitch_rad"])
                                for r in traces
                                if r["time_s"] > c["input"]["duration_s"] - 2
                            ),
                            default=None,
                        )
                    )
            rows.append(
                dict(
                    profile=profile,
                    kind=kind,
                    case=after["case"],
                    normal=kind != "stress" and after["case"] in NORMAL_BALANCE,
                    before=before["status"],
                    after=after["status"],
                    before_hold_pitch_rad=peaks[0],
                    after_hold_pitch_rad=peaks[1],
                    hold_limit_rad=after["input"]["final_pitch_limit_rad"],
                    review=str(new / "review.html"),
                )
            )
    normal = [r for r in rows if r["normal"]]
    stress = [r for r in rows if r["kind"] == "stress"]
    counts = {
        label: dict(
            before=sum(r["before"] == "pass" for r in group),
            after=sum(r["after"] == "pass" for r in group),
            total=len(group),
        )
        for label, group in [("normal", normal), ("stress", stress)]
    }
    source_html = Path("artifacts/shin-cover-review-v1/index.html")
    bindings[str(source_html)] = sha(source_html)
    data = embedded_dataset(source_html)
    item = next(r for r in data["robots"] if r["name"] == "Scout · square shin covers")
    if {p["name"] for p in item["parts"]} != {p["name"] for p in assembly["parts"]}:
        raise ValueError("Displayed assembly inventory differs")
    robot = make_robot(
        RobotConfig.model_validate(
            read("artifacts/scout-shin-cover-composed-v2/resolved-config.json")
        )
    )
    item["samples"] = []
    for label, result in [("Command", static), ("Observed", observed)]:
        for row in result["rows"]:
            frames = robot.forward_kinematics(
                row["pose_rad"], enforce_limits=label == "Command"
            )
            flagged = [
                name
                for hit in row["hits"]
                for name in [hit["part"], *[h["part"] for h in hit["hits"]]]
            ]
            item["samples"].append(
                dict(
                    name=f"{label} · {row['name']}",
                    kind="pose",
                    phase="",
                    status=row["status"],
                    clearance_status=row["status"],
                    flagged=flagged,
                    pair=flagged or ["No overlap in checked arm/body pairs"],
                    gap_note=f"Minimum checked arm/leg gap {row['minimum_leg_gap_mm']:.3f} mm. Root-neutral self-clearance; same-arm internals and shoulder mating interfaces excluded. No ground, continuous-path or physical proof.",
                    frames=[frames[o].flatten().tolist() for o in item["owners"]],
                )
            )
    for p in item["parts"]:
        p.pop("download", None)
    item.update(
        source="artifacts/scout-shin-cover-composed-v2",
        inherited_status="fail",
        stock_summary="240 named groups · unchanged covered-shin CAD and assumed inertia · sampled arm/body screen only · physical NOT RUN",
    )
    data["robots"] = [item]
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    fragment = (
        Path("ui/scout-posture.fragment.html")
        .read_text()
        .replace("__HEAD_MOUNT_DATA__", encoded)
    )
    (output / "viewer.html").write_text(wrapper(fragment))
    (output / "review.css").write_bytes(Path("ui/product-review.css").read_bytes())

    def fmt(v):
        return "not reached" if v is None else f"{v:.5f}"

    table = "".join(
        f'<tr><td>{html.escape(r["profile"] + " / " + r["case"])}</td><td>{r["before"]}</td><td>{r["after"]}</td><td>{fmt(r["before_hold_pitch_rad"])}</td><td>{fmt(r["after_hold_pitch_rad"])}</td><td>{r["hold_limit_rad"]:.3f}</td><td><a href="../../{r["review"]}">Replay</a></td></tr>'
        for r in rows
    )
    doc = Path("docs/design/scout-rest-posture.md")
    bindings[str(doc)] = sha(doc)
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scout · balance posture</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:auto;padding:28px;background:#f4f3ee;color:#203332}}h1{{font-size:38px}}.cards{{display:flex;gap:20px;flex-wrap:wrap}}.cards p{{background:white;padding:20px;border-radius:16px}}iframe{{width:100%;height:780px;border:0;background:white;border-radius:16px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #cad1cc}}a{{color:#195f55}}.scroll{{overflow-x:auto}}</style></head><body><p>ROBOT LEARNING LAB · SCOUT</p><h1>A small rest-posture adjustment</h1><p>Move both upper arms back 0.06 rad (3.4°) while balancing. No added parts; gains, torque limits, mass assumptions and evaluation thresholds stay fixed.</p><div class="cards"><p><strong>Normal balance</strong><br>{counts["normal"]["before"]}/{counts["normal"]["total"]} → {counts["normal"]["after"]}/{counts["normal"]["total"]} pass</p><p><strong>Harder trials</strong><br>{counts["stress"]["before"]}/{counts["stress"]["total"]} → {counts["stress"]["after"]}/{counts["stress"]["total"]} pass</p><p><strong>Physical evidence</strong><br>NOT RUN</p></div><h2>Inspect commanded and recorded postures</h2><p>Four commanded poses and four peak-hold states from the new normal trials. The root is held upright in this self-clearance viewer; it does not depict floor contact or torso lean.</p><iframe src="viewer.html" title="Scout parts and sampled arm-body checks"></iframe><h2>What remains unresolved</h2><p>The 100 ms delay, larger initial tilt and quarter wheel-response stress cases still fail. The source assembly retains whole-robot failures. Same-arm internal pairs, shoulder mating interfaces, complete trajectories, harness, tolerances and strength remain outside this clearance screen. Dynamics use updated inertia with the earlier contact and visual shapes. Supported manipulation is separate and unchanged; this is a candidate driving rest posture, not a released controller or learned skill.</p><p><a href="../../docs/design/scout-rest-posture.md">Method, physical comparison and reproduction</a> · <a href="../scout-shin-cover-composed-v2/assembly.step">Complete covered Scout STEP</a> · <a href="../shin-dynamics-review-v2/index.html">Pip, Scout and supported placement</a> · <a href="report.json">Source-bound report</a></p><h2>All retained case results</h2><p>Hold pitch is the maximum absolute torso pitch during the final two seconds, in radians. A small pitch alone does not imply success: other task/fault criteria apply. Nine deliberately faulty nominal cases remain failed; early aborts may never reach the hold window.</p><div class="scroll"><table><thead><tr><th>Case</th><th>Before</th><th>After</th><th>Before hold</th><th>After hold</th><th>Limit</th><th>Evidence</th></tr></thead><tbody>{table}</tbody></table></div></body></html>"""
    )
    files = [
        Path(__file__).relative_to(Path.cwd()),
        Path("ui/scout-posture.fragment.html"),
        Path("ui/product-review.css"),
        Path("tools/build_product_review.py"),
        Path("tools/review_integral_head_service.py"),
        Path("tools/review_forearm_accounted.py"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation/source identity, not engineering qualification",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in files}),
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            counts=counts,
            rows=rows,
            geometry=dict(
                commanded=static["checked_geometry_status"],
                observed=observed["checked_geometry_status"],
            ),
            physical_status="not_run",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
