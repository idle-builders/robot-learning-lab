"""Freeze prospective V35 native development replay and no-replacement ledger.

Local preparation only: no pod, native visit, optimizer or score reserve opening.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r4"
V31D = ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d"
SEEDS = ()
ROUTES = ()
V31D_PLAN = json.loads((V31D / "plan.json").read_text())
SOURCES = {
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "bc_checkpoint": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/model_bc_seed.pt",
    "v25_checkpoint": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/checkpoints/model_25.pt",
    "v25_plan": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/plan.json",
    "bank_manifest": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/bank-manifest.json",
    "fit_report": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/fit-report.json",
    "v34_closure": "artifacts/ape-cove50-stand-recovery/seated-trigger-v34/v34-closure.json",
    "unopened_rtx_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35/plan.json",
    "unopened_rtx_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35/attempt-ledger.json",
    "evaluator": "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_v35_a40_r4.py",
    "scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_bc_eval_v35_a40_r4.py",
    "runner": "experiments/compact_morphology/cove50_stand_recovery/run_seated_bc_eval_v35_a40_r4.py",
    "preparer": "experiments/compact_morphology/cove50_stand_recovery/prepare_seated_bc_eval_v35_a40_r4.py",
    "v26_scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_reference_cuda_v26.py",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "frame50": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "v31d_plan": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/plan.json",
    "v31d_report": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/report.json",
    "v31d_installed_closure": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/installed-closure.json",
    "v31d_cuda_sentinel": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/cuda-sentinel.json",
    "failed_a40_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/plan.json",
    "failed_a40_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/attempt-ledger.json",
    "failed_a40_runtime_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/first-runtime-failure.log",
    "failed_a40_r1_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/plan.json",
    "failed_a40_r1_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/attempt-ledger.json",
    "failed_a40_r1_sentinel_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/seed35100-sentinel.log",
    "failed_a40_r1_runtime_attestation": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/runtime-attestation.json",
    "failed_a40_r2_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/plan.json",
    "failed_a40_r2_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/attempt-ledger.json",
    "failed_a40_r2_timing_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35001-timing-bc.log",
    "failed_a40_r2_failure_summary": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/failure-summary.json",
    "failed_a40_r2_sentinel_report": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35300-sentinel.json",
    "failed_a40_r2_sentinel_native": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35300-sentinel.npz",
    "failed_a40_r2_runtime_attestation": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/runtime-attestation.json",
    "failed_a40_r3_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3/plan.json",
    "failed_a40_r3_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3/attempt-ledger.json",
    "failed_a40_r3_pod_attestation": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3/fresh-pod-attestation.json",
    "os_egl_loader": "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_os_egl_v35_a40_r4.py",
    "os_bootstrap": "experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v35_a40_r4.sh",
    "worker_launch": "experiments/compact_morphology/cove50_stand_recovery/launch_seated_bc_env_v35_a40_r4.sh",
    "a40_runtime_repair": "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v35_a40_r4.py",
}
SOURCES.update({f"v31d_material_{name}": rel
                for name, rel in V31D_PLAN["source_paths"].items()})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / rel) for name, rel in SOURCES.items()}


def verify_sources(plan: dict) -> None:
    if plan["source_paths"] != SOURCES or plan["source_sha256"] != source_hashes():
        raise ValueError("V35 source/body/checkpoint changed from frozen plan")
    bank = json.loads((ROOT / SOURCES["bank_manifest"]).read_text())
    fit = json.loads((ROOT / SOURCES["fit_report"]).read_text())
    closure = json.loads((ROOT / SOURCES["v34_closure"]).read_text())
    if (bank["train_families"] != list(range(35001, 35007))
            or bank["fresh_development_validation_seed_families"] != list(range(35101, 35109))
            or bank["separate_unopened_score_seed_families"] != list(range(35201, 35213))
            or fit["bank_manifest_sha256"] != sha(ROOT / SOURCES["bank_manifest"])
            or fit["checkpoint_sha256"] != sha(ROOT / SOURCES["bc_checkpoint"])
            or fit["chosen_epochs"] != 12
            or bank["v34_closure_sha256"] != sha(ROOT / SOURCES["v34_closure"])
            or closure["v34_closed"] is not True
            or plan["v31d_material_sha256"] != V31D_PLAN["source_sha256"]):
        raise ValueError("V35 fit, source closure or disjoint seeds differ")
    installed = json.loads((V31D / "installed-closure.json").read_text())
    if installed["content_sha256"] != plan["v31d_installed_closure_content_sha256"]:
        raise ValueError("V35 installed runtime closure differs from V31d")
    prior = json.loads((ROOT / SOURCES["unopened_rtx_ledger"]).read_text())
    if (prior["sentinel_attempts"] != 0 or
            any(route["attempts"] != 0 for family in prior["families"]
                for route in family["routes"].values()) or
            prior["plan_sha256"] != sha(ROOT / SOURCES["unopened_rtx_plan"])):
        raise ValueError("V35 RTX packet was opened; A40 cannot reuse these seed families")
    failed = json.loads((ROOT / SOURCES["failed_a40_ledger"]).read_text())
    if (failed["sentinel_attempts"] != 0 or failed["timing_attempts"] != 0
            or any(route["attempts"] != 0 for family in failed["families"]
                   for route in family["routes"].values())
            or failed["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_plan"])):
        raise ValueError("First A40 packet opened a native visit; fresh original method required")
    failed_r1 = json.loads((ROOT / SOURCES["failed_a40_r1_ledger"]).read_text())
    if (failed_r1["sentinel_seed"] != 35100 or failed_r1["sentinel_attempts"] != 1
            or failed_r1["sentinel_status"] != "failed_original_unscored"
            or failed_r1["timing_attempts"] != 0
            or any(route["attempts"] != 0 for family in failed_r1["families"]
                   for route in family["routes"].values())
            or failed_r1["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r1_plan"])):
        raise ValueError("A40-r1 stopped sentinel or development source differs")
    failed_r1_log = (ROOT / SOURCES["failed_a40_r1_sentinel_log"]).read_text()
    if "FileNotFoundError" not in failed_r1_log or "evaluate_seated_bc_v35_a40_r1.py" not in failed_r1_log:
        raise ValueError("A40-r1 original failure evidence differs")
    attestation = json.loads((ROOT / SOURCES["failed_a40_r1_runtime_attestation"]).read_text())
    if (attestation["installed_count"] != 227
            or attestation["status"] != "installed_source_attested_before_v35_cuda"
            or attestation["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r1_plan"])):
        raise ValueError("A40-r1 installed runtime success source differs")
    failed_r2 = json.loads((ROOT / SOURCES["failed_a40_r2_ledger"]).read_text())
    summary = json.loads((ROOT / SOURCES["failed_a40_r2_failure_summary"]).read_text())
    if (failed_r2["sentinel_seed"] != 35300 or failed_r2["sentinel_attempts"] != 1
            or failed_r2["sentinel_status"] != "passed_unscored"
            or failed_r2["timing_seed"] != 35001 or failed_r2["timing_attempts"] != 1
            or failed_r2["timing_status"] != "failed_original_unscored"
            or any(route["attempts"] != 0 for family in failed_r2["families"]
                   for route in family["routes"].values())
            or failed_r2["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_plan"])
            or summary["attempt_ledger_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_ledger"])
            or summary["timing_log_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_timing_log"])
            or summary["sentinel_report_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_sentinel_report"])
            or summary["sentinel_native_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_sentinel_native"])
            or summary["runtime_attestation_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_runtime_attestation"])):
        raise ValueError("A40-r2 consumed render failure source differs")
    if ("an OpenGL platform library has not been loaded" not in
            (ROOT / SOURCES["failed_a40_r2_timing_log"]).read_text()):
        raise ValueError("A40-r2 original headless-render failure differs")
    failed_r3 = json.loads((ROOT / SOURCES["failed_a40_r3_ledger"]).read_text())
    r3_pod = json.loads((ROOT / SOURCES["failed_a40_r3_pod_attestation"]).read_text())
    if (failed_r3["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r3_plan"])
            or failed_r3["sentinel_attempts"] != 0
            or failed_r3["render_preflight_attempts"] != 0
            or failed_r3["timing_attempts"] != 0
            or any(route["attempts"] != 0 for family in failed_r3["families"]
                   for route in family["routes"].values())
            or r3_pod["plan_sha256"] != failed_r3["plan_sha256"]
            or r3_pod["pod_id"] != "imzmr21jwlulvc"):
        raise ValueError("A40-r3 zero-attempt failure lineage differs")


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    plan_path, ledger_path = ART / "plan.json", ART / "attempt-ledger.json"
    if plan_path.exists() or ledger_path.exists():
        raise FileExistsError("V35 plan/ledger exists; method or original cannot be replaced")
    from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r4 as runtime
    runtime.static_smoke()
    plan = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r4.plan",
        "status": "prospective_os_egl_loader_repair_for_independent_review_before_paid_cuda",
        "claim_limit": "Environment readiness only: zero scored visits, optimizer steps, or learner qualification",
        "hardware_shift": "A40-r3 set EGL and graphics capability but the pod lacked Ubuntu libEGL.so.1; no native visits began. This packet installs pinned Ubuntu GLVND loader packages and attests them before ape_walk or MuJoCo import, then checks native render, MP4 and timing.",
        "failed_a40_plan_sha256": sha(ROOT / SOURCES["failed_a40_plan"]),
        "failed_a40_ledger_sha256": sha(ROOT / SOURCES["failed_a40_ledger"]),
        "failed_a40_runtime_log_sha256": sha(ROOT / SOURCES["failed_a40_runtime_log"]),
        "failed_a40_r1_plan_sha256": sha(ROOT / SOURCES["failed_a40_r1_plan"]),
        "failed_a40_r1_ledger_sha256": sha(ROOT / SOURCES["failed_a40_r1_ledger"]),
        "failed_a40_r1_sentinel_log_sha256": sha(ROOT / SOURCES["failed_a40_r1_sentinel_log"]),
        "failed_a40_r1_runtime_attestation_sha256": sha(ROOT / SOURCES["failed_a40_r1_runtime_attestation"]),
        "failed_a40_r1_sentinel_consumed": 35100,
        "failed_a40_r1_pod_id": "4bhiy36ik82msf",
        "failed_a40_r1_originals_unopened": list(range(35101, 35109)),
        "failed_a40_r2_plan_sha256": sha(ROOT / SOURCES["failed_a40_r2_plan"]),
        "failed_a40_r2_ledger_sha256": sha(ROOT / SOURCES["failed_a40_r2_ledger"]),
        "failed_a40_r2_timing_log_sha256": sha(ROOT / SOURCES["failed_a40_r2_timing_log"]),
        "failed_a40_r2_failure_summary_sha256": sha(ROOT / SOURCES["failed_a40_r2_failure_summary"]),
        "failed_a40_r2_sentinel_consumed": 35300,
        "failed_a40_r2_timing_consumed": 35001,
        "failed_a40_r2_pod_id": "a9osk1e2qoaat5",
        "failed_a40_r2_originals_unopened": list(range(35301, 35309)),
        "failed_a40_r3_plan_sha256": sha(ROOT / SOURCES["failed_a40_r3_plan"]),
        "failed_a40_r3_ledger_sha256": sha(ROOT / SOURCES["failed_a40_r3_ledger"]),
        "failed_a40_r3_pod_attestation_sha256": sha(ROOT / SOURCES["failed_a40_r3_pod_attestation"]),
        "failed_a40_r3_pod_id": "imzmr21jwlulvc",
        "failed_a40_r3_originals_unopened": list(range(35511, 35519)),
        "gpu_id": "NVIDIA A40", "cuda_device_name": "NVIDIA A40",
        "nvidia_smi_name": "NVIDIA A40",
        "unopened_rtx_plan_sha256": sha(ROOT / SOURCES["unopened_rtx_plan"]),
        "unopened_rtx_ledger_sha256": sha(ROOT / SOURCES["unopened_rtx_ledger"]),
        "seed_families": list(SEEDS), "routes": list(ROUTES),
        "fresh_seed_rationale": "A40-r1 consumed 35100; A40-r2 consumed 35300 and 35001; A40-r3 opened zero native visits. A40-r4 uses fresh paired one-step diagnostic 35700 and already-opened BC training-family 35002 for timing. No development or independent-score original is opened.",
        "sentinel_seed": 35700, "sentinel_role": "unscored one-step BC no-video exact CUDA and first-action/target parity",
        "render_preflight_seed": 35700,
        "render_preflight_role": "second one-step BC visit from same seed with exact MJLab 640x480 render, MP4 encode/decode and physical/action parity to no-video sentinel",
        "timing_seed": 35002,
        "timing_role": "unscored environment-only zero-action 500-step exact native task/render/MP4 path, explicit manual reset after done, actor outcome excluded",
        "timing_route": "environment_timing", "timing_max_elapsed_wall_s": 300,
        "timing_control_steps": 500, "timing_encoded_video_frames": 250,
        "timing_continue_after_done": "MJLab auto_reset=False; record terminal step, then explicit reset(env_ids=[0]) before the next step",
        "worker_evidence_volume": "/workspace/cove50-seated-bc-eval-v35-a40-r4",
        "worker_logged_stages": ["bootstrap", "os-egl-check", "checkout-attest", "checkout-runtime", "guard-relay-ready", "runner"],
        "v25_historical_hold_seeds_sealed": list(range(2240, 2252)),
        "v35_independent_score_seeds_sealed": list(range(35201, 35213)),
        "rare_sole_gap_tail_sealed": True,
        "device": "cuda:0", "duration_s": 10.0,
        "original_visits_max": 0, "sentinel_visits_max": 2, "timing_visits_max": 1,
        "os_packages": {"libegl1": "1.7.0-1build1", "libopengl0": "1.7.0-1build1"},
        "os_libraries": ["libEGL.so.1", "libOpenGL.so.0", "libEGL_nvidia.so.0"],
        "evaluator_env": {"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"},
        "render_preflight_frame_shape": [480, 640, 3],
        "render_preflight_min_pixel_std": 1.0,
        "render_preflight_physical_max_abs": 1e-4,
        "render_preflight_actor_max_abs": 1e-5,
        "no_posthoc_replacement": True,
        "max_route_wall_s": 360, "route_start_reserve_s": 20,
        "max_worker_wall_s": 1200, "max_pod_wall_s": 1800,
        "stop_reserve_s": 60,
        "gpu_listed_rate_usd_per_hour": 0.59,
        "gpu_listed_spend_cap_usd": 0.35,
        "guard_contract": "Before any ape_walk or MuJoCo import, install exact Ubuntu libegl1/libopengl0 and attest dpkg, ldconfig, ctypes, NVIDIA vendor JSON and library. Then require exact A40 CUDA closure and NVIDIA EGL context, paired one-step BC no-video/video parity, decoded nonblank 640x480 MP4, and unscored environment-only zero-action 500-step timing with explicit done resets and 250 decoded frames <=300 s. Capture immutable stage stdout/stderr and exit codes on /workspace. Zero scored visits or optimizer steps. One-shot stop guard and <=$0.35 listed GPU cap.",
        "first_action_parity_max_abs": 1e-5,
        "processed_target_parity_max_abs": 2e-6,
        "evaluation_common_step_counter": 624,
        "physical_gate": "none; environment-only readiness packet",
        "video": "unscored exact-task diagnostic MP4 from same native report and trace; every timing frame decoded and checked",
        "interpretation": {"readiness": "only the three source-matched diagnostic visits can pass; no learned-control score"},
        "source_paths": SOURCES, "source_sha256": source_hashes(),
        "v31d_material_sha256": V31D_PLAN["source_sha256"],
        "v31d_installed_closure_content_sha256": json.loads(
            (V31D / "installed-closure.json").read_text())["content_sha256"],
    }
    with plan_path.open("x") as output:
        output.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    ledger = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r4.attempts",
        "plan_sha256": sha(plan_path), "seed_families": list(SEEDS),
        "routes": list(ROUTES), "no_posthoc_replacement": True,
        "sentinel_seed": 35700, "sentinel_status": "not_attempted_unscored",
        "sentinel_attempts": 0,
        "render_preflight_seed": 35700, "render_preflight_status": "not_attempted_unscored",
        "render_preflight_attempts": 0,
        "timing_seed": 35002, "timing_route": "environment_timing",
        "timing_status": "not_attempted_unscored", "timing_attempts": 0,
        "families": [{"seed": seed, "routes": {
            route: {"status": "not_attempted", "attempts": 0}
            for route in ROUTES}}
            for seed in SEEDS],
    }
    with ledger_path.open("x") as output:
        output.write(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"Prepared V35 plan {sha(plan_path)}; zero native visits")


if __name__ == "__main__":
    main()
