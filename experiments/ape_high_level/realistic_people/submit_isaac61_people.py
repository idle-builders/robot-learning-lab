"""Submit the bounded two-person Isaac observation job and verify its files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[3]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8898")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "isaac61_first_people.yaml")
    parser.add_argument("--scenario", choices=("away", "crossing", "active_look", "social_show"), default="away")
    parser.add_argument("--response", choices=("owner_show", "bystander_show", "ignored"))
    parser.add_argument("--control-fov", type=int, choices=(70, 90), default=90)
    parser.add_argument("--social-fov", type=int, choices=(70, 90))
    parser.add_argument("--clear-view-response", action="store_true")
    parser.add_argument("--staged-walk-show", action="store_true")
    parser.add_argument("--rtx-lidar", action="store_true")
    parser.add_argument("--range-source", choices=("physx", "rtx"), default="physx")
    parser.add_argument("--distractor-motion", choices=("away", "cross"), default="away")
    parser.add_argument("--continuous-vision", action="store_true")
    parser.add_argument("--fast-owner-flow", action="store_true")
    parser.add_argument("--visual-odometry", action="store_true")
    parser.add_argument("--route-owner", action="store_true")
    parser.add_argument("--route-obstacle", action="store_true")
    parser.add_argument("--route-range-mode", choices=("both", "low", "mid"),
                        default="both")
    parser.add_argument("--odometry-scale", type=float, default=1.0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--person-detector", choices=("grounding_dino", "ssdlite"),
                        default="grounding_dino")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/ape-realistic-people/isaac61-first-people-flash")
    args = parser.parse_args()
    if args.scenario in ("active_look", "social_show") and "Simple_Room/simple_room.usd" not in args.config.read_text():
        raise ValueError("active scenarios require the Simple Room config with the range stream")
    if args.scenario == "social_show" and args.response is None:
        raise ValueError("social_show requires --response")
    if args.scenario != "social_show" and args.response is not None:
        raise ValueError("--response belongs to social_show")
    if args.scenario != "social_show" and args.social_fov is not None:
        raise ValueError("--social-fov belongs to social_show")
    if args.clear_view_response and (args.scenario != "social_show" or
                                     args.response != "owner_show"):
        raise ValueError("clear-view response requires social_show owner_show")
    if args.staged_walk_show and (args.scenario != "social_show" or
                                  args.response != "owner_show"):
        raise ValueError("staged walk show requires social_show owner_show")
    if args.range_source == "rtx" and not args.rtx_lidar:
        raise ValueError("--range-source rtx requires --rtx-lidar")
    if args.distractor_motion != "away" and args.scenario != "social_show":
        raise ValueError("--distractor-motion cross belongs to social_show")
    if args.continuous_vision and args.scenario not in ("active_look", "social_show"):
        raise ValueError("continuous vision requires an active actor")
    if args.fast_owner_flow and (not args.continuous_vision or args.person_detector != "ssdlite"):
        raise ValueError("fast owner flow requires continuous vision and ssdlite")
    if args.visual_odometry and args.scenario not in ("active_look", "social_show"):
        raise ValueError("visual odometry requires an active actor")
    if not .5 <= args.odometry_scale <= 1.5:
        raise ValueError("odometry scale must be between 0.5 and 1.5")
    if args.route_owner and (args.scenario not in ("active_look", "social_show") or
                             not args.visual_odometry or args.range_source != "physx"):
        raise ValueError("route owner requires an active actor, visual odometry and PhysX range")
    if args.route_obstacle and not args.route_owner:
        raise ValueError("route obstacle requires route owner")
    if args.route_range_mode != "both" and not args.route_owner:
        raise ValueError("route range mode requires route owner")
    if args.frame_count is not None and not 1 <= args.frame_count <= 36:
        raise ValueError("frame count must be between 1 and 36")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    source_names = ["run_isaac61_people.py", "isaac61_first_people.yaml"]
    if args.scenario in ("active_look", "social_show"):
        source_names.extend(("online_owner_actor.py", "fast_owner_flow.py",
                             "grounding-dino-tiny-working-config.json"))
    if args.visual_odometry:
        source_names.append("rgbd_visual_odometry.py")
    if args.route_owner:
        source_names.extend(("fan_view_route.py", "rgbd_low_surface.py",
                             "raw_swept_gate.py", "near_body_tof.py",
                             "tof_intrusion_response.py"))
    if args.scenario == "social_show":
        source_names.extend(("social_show_actor.py", "taught-rgb-crop.png",
                             "TomatoSauce.obj", "TomatoSauce.mtl", "TomatoSauce.jpg"))
    for name in source_names:
        if name.endswith(".yaml"):
            path = args.config
        elif name == "taught-rgb-crop.png":
            path = ROOT / "artifacts/ape-robocasa-kitchen-pilot/taught-rgb-crop.png"
        elif name.startswith("TomatoSauce."):
            path = ROOT / ".cache/ape-high-level/hope_meshes_eval" / name
        else:
            path = Path(__file__).parent / name
        data = path.read_bytes()
        files[name] = {"sha256": sha(data), "base64": base64.b64encode(data).decode()}
    worker = ROOT / "workers/serverless/isaac_people.py"
    identity = {"worker_sha256": sha(worker.read_bytes()),
                "source_config": str(args.config.resolve()),
                "scenario": args.scenario, "response": args.response,
                "control_fov_deg": args.control_fov,
                "social_fov_deg": args.social_fov or args.control_fov,
                "clear_view_response": args.clear_view_response,
                "staged_walk_show": args.staged_walk_show,
                "rtx_lidar": args.rtx_lidar,
                "range_source": args.range_source,
                "distractor_motion": args.distractor_motion,
                "continuous_vision": args.continuous_vision,
                "fast_owner_flow": args.fast_owner_flow,
                "visual_odometry": args.visual_odometry,
                "odometry_scale": args.odometry_scale,
                "route_owner": args.route_owner,
                "route_obstacle": args.route_obstacle,
                "route_range_mode": args.route_range_mode,
                "frame_count": args.frame_count,
                "person_detector": args.person_detector,
                "input_sha256": {name: record["sha256"] for name, record in files.items()}}
    (args.output / "inputs.json").write_text(json.dumps(identity, indent=2) + "\n")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/isaac_people/runsync",
        data=json.dumps({"input": {"payload": {"files": files,
                                                 "scenario": args.scenario,
                                                 "response": args.response,
                                                 "control_fov_deg": args.control_fov,
                                                 "social_fov_deg": args.social_fov or args.control_fov,
                                                 "clear_view_response": args.clear_view_response,
                                                 "staged_walk_show": args.staged_walk_show,
                                                 "rtx_lidar": args.rtx_lidar,
                                                 "range_source": args.range_source,
                                                 "distractor_motion": args.distractor_motion,
                                                 "continuous_vision": args.continuous_vision,
                                                 "fast_owner_flow": args.fast_owner_flow,
                                                 "visual_odometry": args.visual_odometry,
                                                 "odometry_scale": args.odometry_scale,
                                                 "route_owner": args.route_owner,
                                                 "route_obstacle": args.route_obstacle,
                                                 "route_range_mode": args.route_range_mode,
                                                 "frame_count": args.frame_count,
                                                 "person_detector": args.person_detector}}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3700) as response:
            result = json.load(response)
        output = result.get("output", {})
        archive_b64 = output.pop("archive_base64", None)
        (args.output / "response.json").write_text(json.dumps(result, indent=2) + "\n")
        if archive_b64 is not None:
            raw = base64.b64decode(archive_b64, validate=True)
            if sha(raw) != output["archive_sha256"] or len(raw) > 9_000_000:
                raise ValueError("Archive identity or size mismatch")
            (args.output / "capture.zip").write_bytes(raw)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                infos = archive.infolist()
                if sum(item.file_size for item in infos) > 100_000_000:
                    raise ValueError("Unpacked evidence too large")
                for item in infos:
                    path = PurePosixPath(item.filename)
                    if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                        raise ValueError("Unsafe archive path")
                evidence = args.output / "evidence"
                evidence.mkdir()
                archive.extractall(evidence)
        if result.get("status") != "COMPLETED" or output.get("status") != "captured_unreviewed":
            raise RuntimeError("Worker did not capture; inspect response and retained logs")
    except Exception as error:
        (args.output / "client-error.json").write_text(
            json.dumps({"error_type": type(error).__name__, "message": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
