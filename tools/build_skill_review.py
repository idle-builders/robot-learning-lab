"""Compare bounded skill candidates on both collar robots; retain every attempt."""

import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.recording import encoded, sha256
from lab.skills import Candidate, evaluate_skill, generate_candidates, verify_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    specs = [
        (
            Path("skills/definitions/supported-place.yaml"),
            "release_clearance_m",
            [0.01, 0.025],
            "Place the block in its tray.",
        ),
        (
            Path("skills/definitions/wheel-drive-hold.yaml"),
            "pitch_gain_m_s",
            [4.0],
            "Balance, roll forward and return.",
        ),
    ]
    results = []
    for definition, parameter, values, instruction in specs:
        proposals = output / "candidates" / definition.stem
        ids = generate_candidates(definition, parameter, values, instruction, proposals)
        for ident in ids:
            candidate_path = proposals / f"{ident}.json"
            candidate = Candidate.model_validate_json(candidate_path.read_text())
            for robot in ("pip", "scout"):
                name = f"{robot}-{definition.stem}-{getattr(candidate.parameters, parameter):g}"
                folder = output / "evaluations" / name
                model = Path(f"artifacts/{robot}-neck-collar-mass-v1/nominal")
                report = evaluate_skill(
                    definition,
                    candidate_path,
                    Path(f"variants/{robot}-neck-collar.yaml"),
                    model,
                    model / "mass-properties.json",
                    folder,
                )
                integrity = verify_result(folder)
                results.append(
                    dict(
                        name=name,
                        robot=robot,
                        skill_id=definition.stem,
                        candidate_id=ident,
                        parameters=candidate.parameters.model_dump(),
                        evaluation=str(folder.relative_to(output)),
                        manifest_sha256=sha256(folder / "manifest.json"),
                        evaluation_status=report["evaluation_status"],
                        positive_successes=report["positive_successes"],
                        positive_cases=report["positive_cases"],
                        negative_controls_detected=report["negative_controls_detected"],
                        evidence_scope=report["evidence_scope"],
                        integrity=integrity,
                    )
                )
                print(
                    name,
                    report["evaluation_status"],
                    f"{report['positive_successes']}/{report['positive_cases']}",
                    flush=True,
                )
                (output / "progress.json").write_text(encoded(results) + "\n")
    # Rejected candidates are research results. Integrity/runner failures fail this study.
    healthy = all(
        r["integrity"]["status"] == "pass"
        and all(
            c["runner_status"] == "completed"
            for c in json.loads((output / r["evaluation"] / "report.json").read_text())[
                "cases"
            ]
        )
        for r in results
    )
    registry = dict(
        schema_version=1,
        status="pass" if healthy else "fail",
        entries=results,
        scope="visible simulation parameter probes; no learned policy, physical compatibility or held-out final test",
        source_sha256=sha256(__file__),
    )
    (output / "registry.json").write_text(encoded(registry) + "\n")
    (output / "build_skill_review.py").write_bytes(Path(__file__).read_bytes())
    rows = "".join(
        f"<tr><td>{r['robot']}</td><td>{r['skill_id']}</td><td>{html.escape(str(r['parameters']))}</td>"
        f"<td>{r['positive_successes']}/{r['positive_cases']}</td><td>{r['negative_controls_detected']}</td>"
        f'<td>{r["evaluation_status"]}</td><td><a href="{r["evaluation"]}/report.json">Report</a> · '
        f'<a href="{r["evaluation"]}/contract.json">IO contract</a></td></tr>'
        for r in results
    )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Pip & Scout skill factory</title>'
        "<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;background:#f5f3ed;color:#213334}"
        "td,th{padding:12px;text-align:left;border-bottom:1px solid #ccc}td:nth-child(3){font-size:12px}</style>"
        "<h1>Pip & Scout · skill candidates</h1><p>Two named skills, bounded controller parameters and fixed evaluation. "
        "All candidates, failed attempts and negative controls remain inspectable. Both controllers own every joint exclusively.</p>"
        "<p>Placement uses a supported torso and privileged object/contact state. Balance uses simulated IMU/encoders. "
        "These are visible simulation probes, not trained policies or physical qualification.</p>"
        "<table><tr><th>Robot</th><th>Skill</th><th>Candidate parameters</th><th>Positive successes</th>"
        "<th>Negative controls detected</th><th>Evaluation</th><th>Evidence</th></tr>"
        + rows
        + "</table>"
        '<p><a href="registry.json">Versioned registry</a> · <a href="../../docs/design/skills.md">Contract and reproduction</a></p>'
    )
    if not healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
