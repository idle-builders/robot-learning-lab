"""Present source-bound passive-assist screening without claiming installation."""

import argparse
import json
from pathlib import Path
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.screen_shoulder_assist import assist


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/shoulder-assist-screen-v1")
    report = verify_report(source)
    raw = json.loads((source / "data.json").read_text())
    cases = []
    for case in raw["cases"]:
        curves = []
        for candidate in raw["settings"]["candidates"]:
            for domain in ["joint_limit_grid", "recorded_targets"]:
                rows = [r for r in case["samples"] if r["domain"] == domain]
                by_roll = {}
                for row in rows:
                    q = row["q"][0]
                    t = assist(q, candidate)[0]
                    by_roll.setdefault(q, []).append(row["gravity_nm"] - t)
                curves.append(
                    dict(
                        candidate=candidate["name"],
                        domain=domain,
                        points=[
                            dict(q=q, low=min(values), high=max(values))
                            for q, values in sorted(by_roll.items())
                        ],
                    )
                )
        cases.append(
            {**{k: v for k, v in case.items() if k != "samples"}, "curves": curves}
        )
    template = Path("ui/shoulder-assist.html")
    data = dict(settings=raw["settings"], cases=cases)
    (output / "index.html").write_text(
        template.read_text().replace(
            "__DATA__", json.dumps(data).replace("<", "\\u003c")
        )
    )
    builder = Path(__file__).relative_to(Path.cwd())
    (output / "builder.py").write_bytes(builder.read_bytes())
    sources = [
        Path("docs/design/shoulder-assist-study.md"),
        Path("docs/research/arm-passive-assist.md"),
        Path("docs/design/shoulder-assist-bench.json"),
        source / "report.json",
        source / "data.json",
        template,
        builder,
        Path("tools/screen_shoulder_assist.py"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation identity only; spring study remains partial",
            source_bindings={str(p): sha(p) for p in sources},
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            candidate_dynamics="not_run",
            physical="not_run",
            source_status=report["status"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
