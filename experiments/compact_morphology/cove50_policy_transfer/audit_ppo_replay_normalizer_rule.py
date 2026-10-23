"""Record the installed PPO replay normalizer rule without running an actor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(".venv-control/lib/python3.12/site-packages")
NORMALIZATION = ROOT / PACKAGE / "rsl_rl/modules/normalization.py"
MODEL = ROOT / PACKAGE / "rsl_rl/models/mlp_model.py"
METADATA = ROOT / PACKAGE / "rsl_rl_lib-5.0.1.dist-info/METADATA"
SETUP = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/setup-remote.log"
PAYLOAD = ROOT / "artifacts/ape-policy-ppo-normalizers-20261013/report.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = ROOT / "artifacts/ape-policy-ppo-normalizers-20261013/runtime-rule.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = METADATA.read_text()
    normalizer = NORMALIZATION.read_text()
    model = MODEL.read_text()
    setup = SETUP.read_text()
    payload = json.loads(PAYLOAD.read_text())
    pointer = json.loads(POINTER.read_text())
    if not all((
        re.search(r"(?m)^Name: rsl-rl-lib$", metadata),
        re.search(r"(?m)^Version: 5\.0\.1$", metadata),
        "Successfully installed " in setup and "rsl-rl-lib-5.0.1" in setup,
        "self.obs_normalizer = EmpiricalNormalization(self.obs_dim)" in model,
        "latent = self.obs_normalizer(latent)" in model,
        "eps: float = 1e-2" in normalizer,
        "return (x - self._mean) / (self._std + self.eps)" in normalizer,
        payload["policy_count"] == 24,
        payload["policies_with_zero_std_features"] == 10,
        payload["source_sha256"]["pointer"] == sha(POINTER),
        pointer["selected_design_id"] is None,
    )):
        raise ValueError("Replay normalizer source, saved payload or design pointer changed")
    report = {
        "schema": "robot-learning-lab.ape-ppo-replay-normalizer-rule.v1",
        "role": "installed_replay_source_rule_no_actor_execution",
        "runtime": {"mjlab": "1.3.0", "rsl_rl_lib": "5.0.1"},
        "source_sha256": {
            "script": sha(Path(__file__)),
            "normalization": sha(NORMALIZATION),
            "mlp_model": sha(MODEL),
            "package_metadata": sha(METADATA),
            "remote_setup_log": sha(SETUP),
            "saved_payload_report": sha(PAYLOAD),
            "design_pointer": sha(POINTER),
        },
        "source_paths": {
            "normalization": str(NORMALIZATION.relative_to(ROOT)),
            "mlp_model": str(MODEL.relative_to(ROOT)),
            "package_metadata": str(METADATA.relative_to(ROOT)),
            "remote_setup_log": str(SETUP.relative_to(ROOT)),
        },
        "inference_rule": "(observation - saved_mean) / (saved_std + 0.01)",
        "zero_std_denominator": 0.01,
        "saved_ppo_policy_count": 24,
        "saved_policies_with_zero_std_features": 10,
        "qualification": (
            "Source rule for the installed MJLab 1.3.0 / RSL-RL 5.0.1 replay "
            "runtime. No first observation, normalized vector, actor action "
            "or selected-body rollout was measured. Recheck the actual runtime "
            "and checkpoint before new-body transfer."),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded installed replay normalizer rule: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
