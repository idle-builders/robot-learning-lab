"""Bounded Isaac Sim 6.1 rigged-person observation run on Runpod compute."""

from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(
    name="robot-lab-isaac-people",
    gpu=GpuGroup.ADA_24,
    workers=(0, 1),
    gpu_count=1,
    idle_timeout=900,
    execution_timeout_ms=3_600_000,
    dependencies=["uv==0.12.17"],
    system_dependencies=["git", "ffmpeg", "libegl1", "libgl1", "libopengl0", "libvulkan1", "libxt6"],
    template=PodTemplate(containerDiskInGb=150),
    env={"NVIDIA_DRIVER_CAPABILITIES": "all",
         "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
         "PRIVACY_CONSENT": "N", "RUNPOD_API_KEY": ""},
)
async def people(payload: dict) -> dict:
    import base64
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    import zipfile

    root = Path(tempfile.mkdtemp(prefix="ape-isaac-people-"))
    evidence = root / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    scenario = payload.get("scenario", "away")
    if scenario not in ("away", "crossing", "active_look", "social_show"):
        raise ValueError(f"Unknown scenario {scenario}")
    response = payload.get("response")
    control_fov_deg = payload.get("control_fov_deg", 90)
    if control_fov_deg not in (70, 90):
        raise ValueError(f"Unsupported control FOV {control_fov_deg}")
    social_fov_deg = payload.get("social_fov_deg", control_fov_deg)
    if social_fov_deg not in (70, 90):
        raise ValueError(f"Unsupported social FOV {social_fov_deg}")
    clear_view_response = payload.get("clear_view_response", False)
    if not isinstance(clear_view_response, bool) or (clear_view_response and
            (scenario != "social_show" or response != "owner_show")):
        raise ValueError("clear_view_response requires social_show owner_show")
    staged_walk_show = payload.get("staged_walk_show", False)
    if not isinstance(staged_walk_show, bool) or (staged_walk_show and
            (scenario != "social_show" or response != "owner_show")):
        raise ValueError("staged_walk_show requires social_show owner_show")
    rtx_lidar = payload.get("rtx_lidar", False)
    if not isinstance(rtx_lidar, bool):
        raise ValueError("rtx_lidar must be a boolean")
    range_source = payload.get("range_source", "physx")
    if range_source not in ("physx", "rtx") or (range_source == "rtx" and not rtx_lidar):
        raise ValueError("range_source must be physx or rtx with rtx_lidar enabled")
    distractor_motion = payload.get("distractor_motion", "away")
    if distractor_motion not in ("away", "cross") or (distractor_motion == "cross" and scenario != "social_show"):
        raise ValueError("distractor_motion must be away or social_show cross")
    continuous_vision = payload.get("continuous_vision", False)
    fast_owner_flow = payload.get("fast_owner_flow", False)
    visual_odometry = payload.get("visual_odometry", False)
    odometry_scale = payload.get("odometry_scale", 1.0)
    route_owner = payload.get("route_owner", False)
    route_obstacle = payload.get("route_obstacle", False)
    route_range_mode = payload.get("route_range_mode", "both")
    frame_count = payload.get("frame_count")
    if not isinstance(continuous_vision, bool) or (continuous_vision and scenario not in ("active_look", "social_show")):
        raise ValueError("continuous_vision requires an active actor")
    if frame_count is not None and (not isinstance(frame_count, int) or not 1 <= frame_count <= 36):
        raise ValueError("frame_count must be between 1 and 36")
    person_detector = payload.get("person_detector", "grounding_dino")
    if person_detector not in ("grounding_dino", "ssdlite"):
        raise ValueError("person_detector must be grounding_dino or ssdlite")
    if not isinstance(fast_owner_flow, bool) or (fast_owner_flow and
            (not continuous_vision or person_detector != "ssdlite")):
        raise ValueError("fast_owner_flow requires continuous vision and ssdlite")
    if not isinstance(visual_odometry, bool) or (visual_odometry and
            scenario not in ("active_look", "social_show")):
        raise ValueError("visual_odometry requires an active actor")
    if not isinstance(odometry_scale, (int, float)) or not .5 <= odometry_scale <= 1.5:
        raise ValueError("odometry_scale must be between 0.5 and 1.5")
    if not isinstance(route_owner, bool) or (route_owner and
            (scenario not in ("active_look", "social_show") or
             not visual_odometry or range_source != "physx")):
        raise ValueError("route_owner requires an active actor, visual odometry and PhysX range")
    if not isinstance(route_obstacle, bool) or (route_obstacle and not route_owner):
        raise ValueError("route_obstacle requires route_owner")
    if route_range_mode not in ("both", "low", "mid") or (
            route_range_mode != "both" and not route_owner):
        raise ValueError("route_range_mode requires route_owner")
    if scenario == "social_show" and response not in ("owner_show", "bystander_show", "ignored"):
        raise ValueError(f"Unknown social response {response}")
    report = {"status": "fail", "stages": [], "isaac_sim": "6.1.0.0",
              "scenario": scenario, "response": response,
              "control_fov_deg": control_fov_deg,
              "social_fov_deg": social_fov_deg,
              "clear_view_response": clear_view_response,
              "staged_walk_show": staged_walk_show,
              "rtx_lidar": rtx_lidar,
              "range_source": range_source,
              "distractor_motion": distractor_motion,
              "continuous_vision": continuous_vision,
              "fast_owner_flow": fast_owner_flow,
              "visual_odometry": visual_odometry,
              "odometry_scale": odometry_scale,
              "route_owner": route_owner,
              "route_obstacle": route_obstacle,
              "route_range_mode": route_range_mode,
              "frame_count": frame_count,
              "person_detector": person_detector,
              "scope": ("sensor-driven high-level look/approach with kinematic camera proxy"
                        if scenario == "active_look" else
                        "sensor-driven person attention and new-show observation with kinematic camera proxy"
                        if scenario == "social_show" else
                        "two rigged people, low RGB-D Isaac scene smoke; no ape actor")}

    def run(name, command, timeout):
        remaining = min(timeout, 3300 - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError("worker total deadline")
        os.write(1, (json.dumps({"people_stage": name, "state": "starting"}) + "\n").encode())
        begin = time.monotonic()
        log = evidence / f"{name}.log"
        process = subprocess.Popen(
            ["bash", "-o", "pipefail", "-c",
             '"$@" 2>&1 | tee -a "$ROBOTLAB_STAGE_LOG"', "people-stage", *command],
            cwd=root, env={**os.environ, "ROBOTLAB_STAGE_LOG": str(log),
                           "UV_HTTP_TIMEOUT": "600", "UV_HTTP_RETRIES": "5"},
            stdout=1, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            report["stages"].append({"stage": name, "status": "timeout"})
            raise TimeoutError(f"{name} deadline")
        report["stages"].append({"stage": name, "exit_code": code,
                                 "elapsed_s": time.monotonic() - begin})
        os.write(1, (json.dumps({"people_stage": name, "exit_code": code}) + "\n").encode())
        if code:
            raise RuntimeError(f"{name} failed")

    try:
        source_dir = root / "source"
        source_dir.mkdir()
        source_names = ["run_isaac61_people.py", "isaac61_first_people.yaml"]
        if scenario in ("active_look", "social_show"):
            source_names.extend(("online_owner_actor.py", "fast_owner_flow.py",
                                 "grounding-dino-tiny-working-config.json"))
        if visual_odometry:
            source_names.append("rgbd_visual_odometry.py")
        if route_owner:
            source_names.extend(("fan_view_route.py", "rgbd_low_surface.py",
                                 "raw_swept_gate.py", "near_body_tof.py",
                                 "tof_intrusion_response.py"))
        if scenario == "social_show":
            source_names.extend(("social_show_actor.py", "taught-rgb-crop.png",
                                 "TomatoSauce.obj", "TomatoSauce.mtl", "TomatoSauce.jpg"))
        for name in source_names:
            record = payload["files"][name]
            data = base64.b64decode(record["base64"], validate=True)
            limit = (2_000_000 if name.startswith("TomatoSauce.") else
                     150_000 if name == "run_isaac61_people.py" else 100_000)
            if len(data) > limit or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"{name} source hash mismatch")
            (source_dir / name).write_bytes(data)
            (evidence / name).write_bytes(data)
            report.setdefault("input_sha256", {})[name] = record["sha256"]
        config = source_dir / "isaac61_first_people.yaml"
        old = "/workspace/ape-realistic-people/isaac61-first-people"
        changed = config.read_text().replace(old, str(evidence / "capture"))
        if old in changed or changed == config.read_text():
            raise ValueError("expected output directory missing in config")
        config.write_text(changed)

        run("hardware", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                         "--format=csv,noheader"], 20)
        python = Path("/tmp/robot-lab-isaac-env/bin/python")
        ready = python.parents[1] / ".robotlab-isaac-ready"
        if not ready.exists():
            if python.exists():
                report["stages"].append({"stage": "venv", "status": "reused_incomplete"})
            else:
                run("venv", [sys.executable, "-m", "uv", "venv", "--python", "3.12",
                             str(python.parents[1])], 60)
            run("torch", [sys.executable, "-m", "uv", "pip", "install", "--python", str(python),
                          "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu128"], 420)
            install = [sys.executable, "-m", "uv", "pip", "install", "--python",
                       str(python)]
            index = ["--extra-index-url", "https://pypi.nvidia.com",
                     "--index-strategy", "unsafe-best-match"]
            # Isaac's base experience needs extensions absent from the public
            # registry. The documented full cache is large; allow its first
            # worker install to finish, then reuse the warm environment.
            run("isaac-install", install + ["isaacsim[all,extscache]==6.1.0.0"] +
                index, 2400)
            ready.write_text("isaacsim 6.1.0.0\n")
        else:
            report["stages"].append({"stage": "isaac-install", "status": "cached"})
        if scenario in ("active_look", "social_show"):
            run("vision-deps", [sys.executable, "-m", "uv", "pip", "install",
                "--python", str(python), "transformers==5.5.4",
                "huggingface-hub==1.32.0"], 300)
            run("vision-torchvision-cu128", [sys.executable, "-m", "uv", "pip",
                "install", "--python", str(python), "--reinstall", "--no-deps",
                "torchvision==0.26.0+cu128", "--index-url",
                "https://download.pytorch.org/whl/cu128"], 180)
        run("capture", [str(python), str(source_dir / "run_isaac61_people.py"),
                        "--config", str(config), "--scene-info",
                        str(evidence / "scene-info.json"), "--output",
                        str(evidence / "capture"), "--scenario", scenario,
                        *(["--response", response] if response else []),
                        "--control-fov", str(control_fov_deg),
                        "--social-fov", str(social_fov_deg),
                        *(["--clear-view-response"] if clear_view_response else []),
                        *(["--staged-walk-show"] if staged_walk_show else []),
                        "--range-source", range_source,
                        "--distractor-motion", distractor_motion,
                        *(["--continuous-vision"] if continuous_vision else []),
                        *(["--fast-owner-flow"] if fast_owner_flow else []),
                        *(["--visual-odometry"] if visual_odometry else []),
                        "--odometry-scale", str(odometry_scale),
                        *(["--route-owner"] if route_owner else []),
                        *(["--route-obstacle"] if route_obstacle else []),
                        "--route-range-mode", route_range_mode,
                        *(["--frame-count", str(frame_count)] if frame_count is not None else []),
                        "--person-detector", person_detector,
                        *(["--rtx-lidar"] if rtx_lidar else [])],
            2400 if scenario in ("active_look", "social_show") else 480)
        if "ISAAC_RUN_FAILURE" in (evidence / "capture.log").read_text(errors="replace"):
            raise RuntimeError("Isaac coroutine reported failure; inspect capture.log")
        capture = evidence / "capture"
        files = [p for p in capture.rglob("*") if p.is_file()]
        report["capture_files"] = len(files)
        report["capture_bytes"] = sum(p.stat().st_size for p in files)
        scene = json.loads((evidence / "scene-info.json").read_text())
        report["rigged_character_count"] = len(scene["characters"])
        camera_dirs = [p for p in capture.iterdir() if p.is_dir() and (p / "rgb").is_dir()]
        if len(camera_dirs) < 2 or not all(list((p / "rgb").glob("*.png")) for p in camera_dirs):
            raise RuntimeError("capture returned without two RGB camera streams")
        for camera_dir in camera_dirs:
            run("encode-" + camera_dir.name[-5:], ["ffmpeg", "-v", "error", "-y",
                "-framerate", "4" if scenario == "crossing" else "10",
                "-i", str(camera_dir / "rgb/rgb_%05d.png"),
                "-vf", "scale=640:-2", "-c:v", "libx264", "-crf", "28",
                "-pix_fmt", "yuv420p",
                str(evidence / (camera_dir.name + ".mp4"))], 120)
        depth_code = """from pathlib import Path
import numpy as np
import sys
root, out = map(Path, sys.argv[1:3])
expected = int(sys.argv[3])
for camera in (path for path in root.iterdir() if path.is_dir()):
    files = sorted((camera / 'distance_to_camera').glob('*.npy'))
    if len(files) != expected:
        raise RuntimeError(f'{camera.name}: expected {expected} depth frames, got {len(files)}')
    depth = np.stack([np.load(path)[::3, ::3] for path in files]).astype(np.float32)
    np.savez_compressed(out / f'{camera.name}-depth-sequence-stride3.npz', depth_m=depth)
"""
        run("depth-sample", [str(python), "-c", depth_code, str(capture), str(evidence),
                             str(frame_count or (20 if scenario == "social_show" else 12))], 60)
        if len(list(evidence.glob("*-depth-sequence-stride3.npz"))) < 2:
            raise RuntimeError("capture returned without two metric depth samples")
        report["status"] = "captured_unreviewed"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (evidence / "report.json").write_text(json.dumps(report, indent=2))

    # The first probe has an intentionally small response. A complete raw
    # sequence must use durable transport after the volume API is healthy.
    # Keep the bounded response for sensor/decision evidence. Installation
    # logs and duplicate HOPE mesh source dominate the zip without changing
    # replay; retain capture.log only when a stage fails.
    duplicate_scene_assets = {"TomatoSauce.obj", "TomatoSauce.mtl",
                              "TomatoSauce.jpg", "TomatoSauce-converted.usd"}
    selected = [p for p in evidence.iterdir() if p.is_file() and
                p.name not in duplicate_scene_assets and
                (p.suffix != ".log" or (report["status"] != "captured_unreviewed" and
                                        p.name == "capture.log"))]
    capture = evidence / "capture"
    if capture.exists():
        by_camera = {}
        for path in sorted(p for p in capture.rglob("*") if p.is_file()):
            relative = path.relative_to(capture)
            camera = relative.parts[0]
            by_camera.setdefault(camera, []).append(path)
        for paths in by_camera.values():
            selected.extend(p for p in paths if p.suffix == ".mp4")
            # Social FOV comparisons need the *raw* small-item pixels; H.264
            # changes DINOv2 crop similarity on a 10-20 px can. Stride-3 depth
            # for every frame is already retained, so omit the bulky first
            # full-resolution depth sample to fit selected raw RGB views.
            for label in (("rgb", "camera_params") if scenario == "social_show"
                          else ("rgb", "distance_to_camera", "camera_params")):
                matches = [p for p in paths if label in p.parts or label in p.name]
                if matches:
                    selected.append(matches[0])
            if scenario == "social_show":
                selected.extend(p for p in paths if p.name in
                                ("rgb_00013.png", "rgb_00015.png"))
            selected.extend(p for p in paths if p.name == "object_detection.json")
    def pack(paths):
        stream = io.BytesIO()
        included = []
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for path in dict.fromkeys(paths):
                # The 15 Hz safety trace grows with inference duration. It
                # compresses well and is required to audit the mounted rays;
                # do not silently drop it at the per-file transport limit.
                if (path.stat().st_size > 5_000_000 and
                        path.name != "low-tof-trace.json"):
                    continue
                archive.write(path, path.relative_to(evidence))
                included.append(str(path.relative_to(evidence)))
        return stream.getvalue(), included

    blob, included = pack(selected)
    if len(blob) > 9_000_000:
        # Preserve videos, synchronized depth and traces before optional raw
        # object-view comparison frames when the inline service is tight.
        selected = [p for p in selected if p.name not in
                    ("rgb_00013.png", "rgb_00015.png")]
        blob, included = pack(selected)
        report["raw_comparison_frames_pruned_for_transport"] = True
    if len(blob) > 9_000_000 and report["status"] == "captured_unreviewed":
        # Half precision retains about millimetre resolution at the 1-3 m
        # interaction range while keeping every synchronized sampled pixel.
        # Run in Isaac's Python; the lightweight Flash handler has no numpy.
        depth_transport_code = """from pathlib import Path
import numpy as np
import sys
for path in Path(sys.argv[1]).glob('*-depth-sequence-stride3.npz'):
    with np.load(path) as data:
        depth = data['depth_m']
    if depth.dtype != np.float32:
        raise ValueError(f'unexpected depth dtype: {depth.dtype}')
    temporary = path.with_name(path.stem + '.half.npz')
    np.savez_compressed(temporary, depth_m=depth.astype(np.float16))
    temporary.replace(path)
"""
        subprocess.run([str(python), "-c", depth_transport_code, str(evidence)],
                       check=True, timeout=60, capture_output=True)
        blob, included = pack(selected)
        report["depth_transport_dtype"] = "float16"
    if len(blob) > 9_000_000:
        # The complete per-beam monitor remains available at capture time;
        # the replay/audit needs scan timing, nearest return and alarm state.
        monitor_path = evidence / "continuous-vision-monitor.json"
        if monitor_path.exists():
            monitor = json.loads(monitor_path.read_text())
            for frame in monitor.get("frames", []):
                for sample in frame.get("samples", []):
                    sample.pop("mid_range_m", None)
            monitor_path.write_text(json.dumps(monitor, separators=(",", ":")))
            blob, included = pack(selected)
            report["monitor_beam_arrays_pruned_for_transport"] = True
    if len(blob) > 9_000_000 and report["status"] == "captured_unreviewed":
        # A noisy rendered room can produce multi-megabyte 36-frame H.264
        # clips. Preserve both playable views at a smaller transport bitrate
        # before discarding synchronized sensor evidence.
        for video in evidence.glob("ape_social_*deg.mp4"):
            compact = video.with_name(video.stem + ".transport.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                            "-vf", "scale=480:-2", "-c:v", "libx264",
                            "-crf", "36", "-pix_fmt", "yuv420p", str(compact)],
                           check=True, timeout=120, capture_output=True)
            compact.replace(video)
        blob, included = pack(selected)
        report["video_transport_resolution"] = "480px wide, H.264 CRF 36"
    if (len(blob) > 9_000_000 and scenario == "active_look" and
            report["status"] == "captured_unreviewed"):
        # The route actor controls from 90°. Keep its raw first frame, both
        # videos, all traces and its depth sequence; the secondary raw first
        # frame/depth sequence are redundant for this route-only run.
        selected = [path for path in selected if not (
            ("ape_social_70deg" in path.parts and path.suffix in (".png", ".npy"))
            or path.name == "ape_social_70deg-depth-sequence-stride3.npz")]
        blob, included = pack(selected)
        report["secondary_raw_depth_pruned_for_transport"] = True
    if report["status"] == "captured_unreviewed":
        outcome_path = evidence / "social-actor-outcome.json"
        if outcome_path.exists():
            report["social_outcome_unreviewed"] = json.loads(outcome_path.read_text())
    report["inline_capture_files"] = included
    report["inline_file_bytes"] = {
        str(path.relative_to(evidence)): path.stat().st_size
        for path in dict.fromkeys(selected) if path.exists()
    }
    report["archive_bytes"] = len(blob)
    (evidence / "report.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(root, ignore_errors=True)
    if len(blob) > 9_000_000:
        return {"status": "fail", "report": report,
                "reason": "selected capture exceeds inline response limit"}
    return {"status": report["status"], "report": report,
            "archive_sha256": hashlib.sha256(blob).hexdigest(),
            "archive_base64": base64.b64encode(blob).decode()}
