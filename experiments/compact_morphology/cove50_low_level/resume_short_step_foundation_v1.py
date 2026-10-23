"""Resume frozen short-step cases after a pre-native EGL import failure."""

import json
import os
from pathlib import Path
import subprocess
import sys

from run_short_step_foundation_v1 import METHOD, OUT, ROOT, score, sha


def main():
    method = json.loads(METHOD.read_text())
    for entry in method["source"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen source changed: {entry['path']}")
    failed = OUT / "seed50101.log"
    if (not failed.is_file() or "AttributeError: 'NoneType' object has no attribute 'eglQueryString'"
            not in failed.read_text() or any((OUT / f"seed50101.{extension}").exists()
                                          for extension in ("json", "npz", "mp4"))):
        raise RuntimeError("Expected only the retained pre-native EGL failure")
    if (OUT / "report.json").exists():
        raise RuntimeError("Already complete")
    body = ROOT / method["source"]["body"]["path"]
    actor = ROOT / method["source"]["actor"]["path"]
    evaluator = ROOT / method["source"]["evaluator"]["path"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    rows = []
    for case in method["fresh_development_cases"]:
        label = f"seed{case['seed']}"
        report, trace, video, log = (OUT / f"{label}.{extension}" for extension in
                                     ("json", "npz", "mp4", "native.log"))
        if any(path.exists() for path in (report, trace, video, log)):
            raise RuntimeError(f"Original case already opened: {label}")
        command = [sys.executable, str(evaluator), str(actor),
                   "--task", method["replay"]["task"], "--body-xml", str(body),
                   "--physics-model", "frame50", "--initial-root-z", "0.231",
                   "--yaw-only-head", "--arms-down", "--worlds", "1",
                   "--seed", str(case["seed"]), "--velocity-x", "0.25",
                   "--velocity-y", "0", "--angular-z", "0",
                   "--motor-rail-v", str(case["motor_rail_v"]),
                   "--command-delay-ms", str(case["command_delay_ms"]),
                   "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
                   "--stop-at-s", str(case["stop_at_s"]), "--duration-s", "3.0",
                   "--out", str(report), "--diagnostic-trace", str(trace),
                   "--video", str(video)]
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=240, check=False)
        if result.returncode:
            raise RuntimeError(f"{label} native evaluator failed {result.returncode}; see {log}")
        row = score(case, report, trace, method)
        row["video_sha256"] = sha(video)
        rows.append(row)
        (OUT / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(label, row["rough_step_and_settle"], row["checks"], flush=True)
    result = {"schema": "robot-learning-lab.short-step-foundation-v1.result.v1",
              "method_sha256": sha(METHOD), "body_sha256": method["source"]["body"]["sha256"],
              "actor_sha256": method["source"]["actor"]["sha256"],
              "pre_native_failure": "seed50101 first import lacked libEGL.so.1; original log retained",
              "recovery_orchestrator_sha256": sha(Path(__file__)),
              "development_cases": len(rows),
              "rough_step_and_settle_passes": sum(row["rough_step_and_settle"] for row in rows),
              "optimizer_updates": 0,
              "claim_limit": "Existing saved actor only; fresh exploratory short-step CUDA screen, not a newly trained or independently robust policy.",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "passes": result["rough_step_and_settle_passes"]}))


if __name__ == "__main__":
    main()
