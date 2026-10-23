"""Run one native Isaac Lab Mimic generation/replay trial on published G1 data."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name="robot-lab-arena-mimic", gpu=GpuGroup.ADA_24,
          workers=(0, 1), gpu_count=1, idle_timeout=5, min_cuda_version="13.0",
          execution_timeout_ms=1_800_000, template=PodTemplate(containerDiskInGb=100),
          dependencies=[],
          system_dependencies=["git", "libegl1", "libgl1", "libopengl0", "libvulkan1",
                               "build-essential", "ninja-build", "libxt6", "vulkan-tools"],
          env={"RUNPOD_API_KEY": "", "NVIDIA_DRIVER_CAPABILITIES": "all",
               "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "N"})
async def arena_mimic(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix="arena-mimic-"))
    evidence = root / "evidence"
    evidence.mkdir()
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    secrets = dict(storage_auth, hf_token=model_auth)
    started = time.monotonic()
    env = dict(os.environ, HF_TOKEN=model_auth, HF_HOME=str(root / "hf-cache"),
               UV_HTTP_TIMEOUT="60", UV_HTTP_RETRIES="1", UV_LINK_MODE="copy",
               HF_HUB_ENABLE_HF_TRANSFER="0", MAX_JOBS="4", PYTHONUNBUFFERED="1")
    report = {"status": "running", "task_success": "not_run", "stages": [],
              "data_source": "NVIDIA published annotated demonstration",
              "operation": "Isaac Lab Mimic augmentation and native replay; not RL",
              "pins": payload["pins"]}

    def run(stage, command, cwd, seconds):
        remaining = 1740 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Arena Mimic worker deadline")
        item = {"stage": stage, "command": command, "status": "running"}
        report["stages"].append(item)
        log = evidence / f"{stage}.log"
        begin = time.monotonic()
        with log.open("x") as stream:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=min(seconds, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                item.update(status="timeout", elapsed_s=time.monotonic() - begin)
                raise
        item.update(status="pass" if code == 0 else "fail", exit_code=code,
                    elapsed_s=time.monotonic() - begin)
        os.write(1, (stage + ": " + item["status"] + "\n").encode())
        if code:
            raise RuntimeError(stage + " failed")

    try:
        pins = payload["pins"]
        run_id = payload["run_id"]
        if not re.fullmatch(r"arena-mimic-[a-f0-9]{12}", run_id):
            raise ValueError("invalid run id")
        run("hardware", ["nvidia-smi"], root, 20)
        run("driver", ["bash", "-c",
            "test \"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)\" -ge 580"], root, 15)
        run("vulkan", ["vulkaninfo", "--summary"], root, 30)
        run("install-uv", [sys.executable, "-m", "pip", "install", "--retries", "1",
                           "--timeout", "30", "uv==0.12.17"], root, 90)

        checkout = root / "IsaacLab-Arena"
        run("clone", ["git", "clone", "--filter=blob:none",
                      "https://github.com/isaac-sim/IsaacLab-Arena.git", str(checkout)], root, 90)
        run("checkout", ["git", "checkout", pins["arena"]], checkout, 20)
        run("submodules", ["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
                           "submodule", "update", "--init", "--depth", "1"], checkout, 180)
        if subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=checkout / "submodules/IsaacLab", text=True).strip() != pins["isaaclab"]:
            raise ValueError("Isaac Lab revision mismatch")
        if hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest() != pins["arena_lock_sha256"]:
            raise ValueError("Arena lock mismatch")

        # Mimic swaps Arena's normal environment for G1MimicEnv and removes the success term before
        # manager construction. Restore the three pieces of Arena state used by that term, then
        # instantiate it so Mimic can call the resulting object directly.
        generator = checkout / "submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"
        source = generator.read_text()
        anchor = "    env.reset()\n\n    motion_planners = None"
        replacement = (
            "    env.reset()\n\n"
            "    from isaaclab.managers import ManagerTermBase\n"
            "    if isinstance(success_term.func, type) and issubclass(success_term.func, ManagerTermBase):\n"
            "        from isaaclab_arena.environments.arena_world import ArenaWorld\n"
            "        from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder\n"
            "        env.arena_world = ArenaWorld(env.scene)\n"
            "        env.progress_tracker = None\n"
            "        env.object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(\n"
            "            num_envs=env.num_envs, device=env.device\n"
            "        )\n"
            "        success_term.func = success_term.func(success_term, env)\n\n"
            "        env.progress_tracker = env._progress_tracker\n\n"
            "    motion_planners = None"
        )
        if source.count(anchor) != 1:
            raise RuntimeError("Mimic compatibility anchor changed")
        generator.write_text(source.replace(anchor, replacement))
        shutil.copyfile(generator, evidence / "generate_dataset-patched.py")
        replay_script = checkout / "submodules/IsaacLab/scripts/tools/replay_demos.py"
        replay_source = replay_script.read_text()
        keyboard_block = (
            "    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))\n"
            "    teleop_interface.add_callback(\"N\", play_cb)\n"
            "    teleop_interface.add_callback(\"B\", pause_cb)\n"
            "    print('Press \"B\" to pause and \"N\" to resume the replayed actions.')\n"
        )
        camera_anchor = "    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)\n"
        step_anchor = "                env.step(actions)\n\n                if state_validation_enabled:"
        reset_anchor = "    env.reset()\n\n    episode_names = list(dataset_file_handler.get_episode_names())"
        if (replay_source.count(keyboard_block) != 1 or
                replay_source.count("    teleop_interface.reset()\n") != 1 or
                replay_source.count(camera_anchor) != 1 or
                replay_source.count(step_anchor) != 1 or
                replay_source.count(reset_anchor) != 1):
            raise RuntimeError("Replay headless compatibility anchor changed")
        replay_source = replay_source.replace(keyboard_block, "").replace("    teleop_interface.reset()\n", "")
        replay_source = replay_source.replace(
            camera_anchor,
            camera_anchor + "    env_cfg.scene.robot_head_cam = None\n    env_cfg.observations.camera_obs = None\n",
        )
        replay_source = replay_source.replace(
            step_anchor,
            "                env.step(actions)\n"
            "                if success_term is not None:\n"
            "                    success_term.func(env, **success_term.params)\n\n"
            "                if state_validation_enabled:",
        )
        replay_source = replay_source.replace(
            reset_anchor,
            "    env.reset()\n\n"
            "    if success_term is not None:\n"
            "        from isaaclab.managers import ManagerTermBase\n"
            "        if isinstance(success_term.func, type) and issubclass(success_term.func, ManagerTermBase):\n"
            "            success_term.func = success_term.func(success_term, env)\n\n"
            "    episode_names = list(dataset_file_handler.get_episode_names())",
        )
        replay_script.write_text(replay_source)
        shutil.copyfile(replay_script, evidence / "replay_demos-patched.py")
        report["compatibility_patch"] = (
            "restore Arena success state in G1MimicEnv and run camera-free headless replay with stateful success validation"
        )
        uv = [sys.executable, "-m", "uv"]
        run("arena-install", [*uv, "sync", "--frozen", "--group", "gr00t-client"], checkout, 600)
        python = str(checkout / ".venv/bin/python")

        annotated = root / "annotated.hdf5"
        download = (
            "from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id={pins['dataset_repo']!r},repo_type='dataset',"
            f"revision={pins['dataset_revision']!r},filename='arena_g1_loco_manipulation_dataset_annotated.hdf5',"
            f"local_dir={str(root)!r}); import shutil; "
            f"shutil.copyfile({str(root / 'arena_g1_loco_manipulation_dataset_annotated.hdf5')!r},{str(annotated)!r})")
        run("download-annotated", [python, "-c", download], checkout, 180)

        # The camera/state episode can be hundreds of MB. Keep it on the worker for replay and
        # retain its digest/shape only; the reference source dataset is already pinned separately.
        generated = root / "generated-one-trial.hdf5"
        generate = [python,
            "submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
            "--enable_cameras", "--mimic", "--input_file", str(annotated),
            "--output_file", str(generated), "--generation_num_trials", "1",
            "--device", "cpu",
            "--external_callback", "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback",
            "--task", "galileo_g1_locomanip_pick_and_place", "--object", "brown_box",
            "--embodiment", "g1_wbc_pink"]
        run("mimic-generate", generate, checkout, 1200)
        if not generated.is_file():
            raise RuntimeError("Mimic produced no HDF5 file")

        summary_path = evidence / "hdf5-summary.json"
        summarize = (
            "import h5py,json,sys; f=h5py.File(sys.argv[1],'r'); "
            "data=f.get('data'); demos=sorted(data.keys()) if data else []; "
            "p=__import__('pathlib').Path(sys.argv[1]); "
            "out={'bytes':p.stat().st_size,'sha256':__import__('hashlib').file_digest(p.open('rb'),'sha256').hexdigest(),"
            "'demos':demos,'demo_count':len(demos)}; "
            "open(sys.argv[2],'w').write(json.dumps(out,indent=2)); print(out); f.close()")
        run("summarize", [python, "-c", summarize, str(generated), str(summary_path)], checkout, 30)
        summary = json.loads(summary_path.read_text())
        if summary["demo_count"] < 1:
            raise RuntimeError("Mimic completed without a successful generated episode")

        replay = [python, "submodules/IsaacLab/scripts/tools/replay_demos.py",
            "--device", "cpu", "--dataset_file", str(generated), "--validate_success_rate",
            "--external_callback", "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback",
            "--task", "galileo_g1_locomanip_pick_and_place", "--object", "brown_box",
            "--embodiment", "g1_wbc_pink"]
        run("native-replay", replay, checkout, 500)
        if "Successfully replayed: 1/1" not in (evidence / "native-replay.log").read_text(errors="replace"):
            raise RuntimeError("Replay exited without 1/1 task success")
        report.update(status="published_mimic_trial_replayed", task_success="pass",
                      generated=summary)
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    finally:
        report["elapsed_s"] = time.monotonic() - started
        for path in evidence.rglob("*"):
            if path.is_file() and path.suffix in {".log", ".json", ".txt", ".yaml"}:
                path.write_text(redact(path.read_text(errors="replace"), secrets))
        report = redact(report, secrets)
        (evidence / "report.json").write_text(json.dumps(report, indent=2))
    ref = storage.retain_tree(evidence, payload["run_id"])
    return {"status": report["status"], "report": report, "s3_evidence": ref}
