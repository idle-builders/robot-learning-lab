"""Present the bounded wheel/arm CAD study without claiming whole-robot fit."""

import argparse
import html
from pathlib import Path
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/balance-arm-clearance-complete-v1")
    prior = Path("artifacts/balance-arm-clearance-v5")
    d = verify_report(source)
    p = verify_report(prior)
    bindings = {str(f / "report.json"): sha(f / "report.json") for f in (source, prior)}
    rows, details = [], []
    for robot in d["robots"]:
        for pose in robot["poses"]:
            for pair in pose["pairs"]:
                if pair.get("reused_from"):
                    old = next(r for r in p["robots"] if r["robot"] == robot["robot"])
                    oldpose = next(x for x in old["poses"] if x["pose"] == "zero")
                    match = next(
                        x
                        for x in oldpose["pairs"]
                        if x["wheel_part"] == pair["wheel_part"]
                        and x["arm_part"] == pair["arm_part"]
                    )
                    for k in ("intersection_volume_mm3", "distance_mm"):
                        if match[k] != pair[k]:
                            raise ValueError("Reused pair differs")
            failed = [x for x in pose["pairs"] if x["intersection_volume_mm3"] > 1e-6]
            touching = [x for x in pose["pairs"] if x["zero_margin_touching"]]
            minimum = min(x["distance_mm"] for x in pose["pairs"])
            label = "Zero" if pose["pose"] == "zero" else "Both elbows −0.4 rad"
            state = (
                "FAIL"
                if failed
                else "ZERO MARGIN"
                if touching
                else "NOMINAL PAIR CLEARANCE"
            )
            rows.append(
                f"<tr><td>{robot['robot'].title()}</td><td>{label}</td><td>{len(pose['pairs'])}</td><td>{len(failed)}</td><td>{minimum:.3f} mm</td><td>{state}</td></tr>"
            )
            hits = "".join(
                f"<li>{html.escape(x['wheel_part'])} / {html.escape(x['arm_part'])}: {x['intersection_volume_mm3']:.3f} mm³ overlap</li>"
                for x in failed
            )
            details.append(
                f"<h3>{robot['robot'].title()} · {label}</h3><ul>{hits or '<li>No overlap in these pairs.</li>'}</ul>"
            )
    content = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wheel and arm clearance</title><style>body{font:17px/1.5 system-ui;background:#f4f2ec;color:#183933;margin:36px auto;max-width:1100px;padding:0 24px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#006957}.note{padding:20px;background:#fff1cc;border-radius:12px}.scroll{overflow:auto}</style><h1>Why the arms need a declared rest pose</h1><p>Exact retained CAD checks compare same-side wheel groups against groups named forearm_* and gripper_* in the complete Pip and Scout compositions. Source/configuration/STEP identity is checked separately from clearance.</p><div class="note">This is a two-pose pair study. It excludes thigh interactions, separately named cover groups, opposite-side pairs, motion between poses, tolerances, wiring and strength. It does not qualify the complete arm or robot. Scout's simplified forearm/thigh contact remains a separate unresolved check.</div><div class="scroll"><table><tr><th>Robot</th><th>Pose</th><th>Pairs</th><th>Overlaps</th><th>Minimum distance</th><th>Result</th></tr>"""
        + "".join(rows)
        + """</table></div><h2>Retained intersections</h2>"""
        + "".join(details)
        + """<p><a href="../integral-head-composed-review-v2/index.html">Inspect the complete neutral CAD and part identities</a> · <a href="../composed-dynamics-review-v2/index.html">Compare control trials</a> · <a href="../balance-arm-clearance-complete-v1/report.json">All pair measurements</a></p><p>The neutral CAD inspector does not display the parked pose; use the declared joint values and detailed pair report for that comparison. Physical print and fixture checks remain not run.</p></html>"""
    )
    (output / "index.html").write_text(content)
    bindings[str(Path(__file__).relative_to(Path.cwd()))] = sha(Path(__file__))
    write(
        output / "report.json",
        dict(
            status="partial",
            scope="Bounded CAD presentation; whole-robot fit not qualified",
            source_bindings=bindings,
            output_hashes={"index.html": sha(output / "index.html")},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
