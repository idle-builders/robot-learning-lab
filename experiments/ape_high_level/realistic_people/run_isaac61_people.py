"""Capture rigged people and optionally run a sensor-driven camera proxy.

The active-look scenario closes perception, range, and kinematic camera
feedback. It does not execute an ape body or qualified physical skill.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-info", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", choices=("away", "crossing", "active_look", "social_show"), default="away")
    parser.add_argument("--response", choices=("owner_show", "bystander_show", "ignored"))
    parser.add_argument("--control-fov", type=int, choices=(70, 90), default=90)
    parser.add_argument("--social-fov", type=int, choices=(70, 90),
                        help="object show view; defaults to the owner control view")
    parser.add_argument("--clear-view-response", action="store_true",
                        help="probe a native held-item hand reach after a clear-view request")
    parser.add_argument("--staged-walk-show", action="store_true",
                        help="owner walks into the social view after the request, then picks up the item")
    parser.add_argument("--rtx-lidar", action="store_true")
    parser.add_argument("--range-source", choices=("physx", "rtx"), default="physx")
    parser.add_argument("--distractor-motion", choices=("away", "cross"), default="away")
    parser.add_argument("--continuous-vision", action="store_true",
                        help="advance people and sample range while vision runs")
    parser.add_argument("--fast-owner-flow", action="store_true",
                        help="track a detector-bound owner for at most three fresh RGB-D frames")
    parser.add_argument("--visual-odometry", action="store_true",
                        help="estimate the eye pose from calibrated RGB-D and gate later translation")
    parser.add_argument("--route-owner", action="store_true",
                        help="plan a short known-free viewpoint route using pose and two-height fan")
    parser.add_argument("--route-obstacle", action="store_true",
                        help="add a visible colliding obstacle to challenge that route")
    parser.add_argument("--route-range-mode", choices=("both", "low", "mid"),
                        default="both", help="which height planes reach the route actor")
    parser.add_argument("--odometry-scale", type=float, default=1.0,
                        help="deliberate command-distance bias in the pose estimator")
    parser.add_argument("--frame-count", type=int,
                        help="cap the sampled views in a timing experiment")
    parser.add_argument("--person-detector", choices=("grounding_dino", "ssdlite"),
                        default="grounding_dino")
    args = parser.parse_args()
    args.social_fov = args.social_fov or args.control_fov
    if args.scenario == "social_show" and args.response is None:
        parser.error("social_show requires --response")
    if args.clear_view_response and (args.scenario != "social_show" or
                                     args.response != "owner_show"):
        parser.error("clear-view response requires social_show owner_show")
    if args.staged_walk_show and (args.scenario != "social_show" or
                                  args.response != "owner_show"):
        parser.error("staged walk show requires social_show owner_show")
    if args.range_source == "rtx" and not args.rtx_lidar:
        parser.error("--range-source rtx requires --rtx-lidar")
    if args.distractor_motion == "cross" and args.scenario != "social_show":
        parser.error("--distractor-motion cross requires social_show")
    if args.frame_count is not None and not 1 <= args.frame_count <= 36:
        parser.error("--frame-count must be between 1 and 36")
    if args.fast_owner_flow and (not args.continuous_vision or args.person_detector != "ssdlite"):
        parser.error("--fast-owner-flow requires continuous vision and ssdlite")
    if args.visual_odometry and args.scenario not in ("active_look", "social_show"):
        parser.error("--visual-odometry requires an active actor")
    if not .5 <= args.odometry_scale <= 1.5:
        parser.error("--odometry-scale must be between 0.5 and 1.5")
    if args.route_owner and (args.scenario not in ("active_look", "social_show") or
                             not args.visual_odometry or args.range_source != "physx"):
        parser.error("--route-owner requires an active actor, visual odometry and PhysX range")
    if args.route_obstacle and not args.route_owner:
        parser.error("--route-obstacle requires --route-owner")
    if args.route_range_mode != "both" and not args.route_owner:
        parser.error("--route-range-mode requires --route-owner")
    launch_config_text = Path(args.config).read_text()
    early_asset_converter = (
        "robotlab_early_asset_converter: enabled" in launch_config_text)
    path_tracing_probe = "robotlab_path_tracing: enabled" in launch_config_text
    if early_asset_converter and args.scenario != "social_show":
        parser.error("Early can conversion extension requires social_show")
    if path_tracing_probe and args.scenario != "social_show":
        parser.error("Path-traced comparison requires social_show")

    from isaacsim import SimulationApp

    launch_settings = {"headless": True}
    if path_tracing_probe:
        launch_settings.update({"renderer": "PathTracing",
                                "samples_per_pixel_per_frame": 4})
    simulation_app = SimulationApp(launch_settings)
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.replicator.agent.core")
        if early_asset_converter:
            enable_extension("omni.kit.asset_converter")
        if args.rtx_lidar:
            enable_extension("isaacsim.sensors.experimental.rtx")
        simulation_app.update()

        async def run() -> None:
            import carb
            if path_tracing_probe:
                render_settings = carb.settings.get_settings()
                render_settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
                render_settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)
            import numpy as np
            import omni.anim.behavior.core as behavior
            import omni.anim.navigation.core as navigation
            import omni.kit.app
            import omni.kit.commands
            import omni.physx
            import omni.replicator.core as rep
            import omni.timeline
            import omni.usd
            from isaacsim.replicator.agent.core import api as ira
            from isaacsim.storage.native import get_assets_root_path
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
            from PIL import Image

            config_path = Path(args.config)
            config_text = config_path.read_text()
            dynamic_low_cube_mode = (
                "continuous_fast" if "robotlab_dynamic_low_cube: continuous_0p24mps" in config_text
                else "continuous" if "robotlab_dynamic_low_cube: continuous_0p15mps" in config_text
                else "instant" if "robotlab_dynamic_low_cube: frame9_xplus0p20" in config_text
                else "creep_fast" if "robotlab_dynamic_low_cube: frame9_xplus0p03_to18" in config_text
                else "creep" if "robotlab_dynamic_low_cube: frame9_xplus0p02_to18" in config_text
                else None)
            low_motion_guard_enabled = "robotlab_low_motion_guard: enabled" in config_text
            path_timed_guard_enabled = "robotlab_path_timed_guard: enabled" in config_text
            person_fallback_enabled = (
                "robotlab_person_fallback: grounding_dino" in config_text)
            force_primary_person_miss = (
                "robotlab_force_primary_person_miss: frame5" in config_text)
            force_one_person_miss = "robotlab_force_person_miss: frame29" in config_text
            route_proxy_radius_m = (.20 if
                "robotlab_route_proxy_radius: 0p20" in config_text else .12)
            route_planning_margin_m = (
                .12 if "robotlab_route_planning_margin: 0p12" in config_text else
                .04 if "robotlab_route_planning_margin: 0p04" in config_text else 0.)
            require_height_coverage = (
                "robotlab_require_height_coverage: enabled" in config_text)
            rgbd_low_hit_fusion_enabled = (
                "robotlab_rgbd_low_hit_fusion: enabled" in config_text)
            raw_swept_gate_enabled = (
                "robotlab_raw_swept_gate: enabled" in config_text)
            held_raw_monitor_enabled = (
                "robotlab_held_raw_monitor: enabled" in config_text)
            low_tof_probe_enabled = (
                "robotlab_low_tof_probe: enabled" in config_text)
            tof_escape_enabled = "robotlab_tof_escape: enabled" in config_text
            tof_escape_distance_m = (.28 if
                "robotlab_tof_escape_distance: 0p28" in config_text else .16)
            low_blind_mode = (
                "all_from_frame9" if
                "robotlab_low_range_blind_sector: all_from_frame9" in config_text
                else "left_70deg" if
                "robotlab_low_range_blind_sector: left_70deg" in config_text
                else None)
            route_mid_room_start = "robotlab_route_mid_room_start: enabled" in config_text
            held_pose_probe = "robotlab_held_pose_probe: enabled" in config_text
            held_head_look_probe = (
                "robotlab_held_head_look_probe: enabled" in config_text)
            verify_gaze_conflict_probe = (
                "robotlab_verify_gaze_conflict_probe: enabled" in config_text)
            verify_gaze_conflict_bypass = (
                "robotlab_verify_gaze_conflict_bypass: enabled" in config_text)
            near_social_view = "robotlab_near_social_view: enabled" in config_text
            clean_social_capture = "robotlab_clean_social_capture: enabled" in config_text
            near_cube_probe = "robotlab_near_cube_probe: enabled" in config_text
            far_cube_probe = "robotlab_far_cube_probe: enabled" in config_text
            far_right_cube_probe = (
                "robotlab_far_right_cube_probe: enabled" in config_text)
            if dynamic_low_cube_mode and not (args.route_owner and args.route_obstacle and
                                              args.continuous_vision):
                raise ValueError("dynamic low cube requires active routed obstacle and continuous vision")
            if low_motion_guard_enabled and (not args.route_owner or
                                             args.route_range_mode == "mid"):
                raise ValueError("low motion guard requires routed low range")
            if path_timed_guard_enabled and (not args.route_owner or
                    args.route_range_mode == "mid" or low_motion_guard_enabled):
                raise ValueError("path-timed guard requires routed low range and excludes scalar guard")
            if low_blind_mode and not (args.route_owner and args.route_obstacle):
                raise ValueError("low blind-sector fault requires routed obstacle")
            if person_fallback_enabled and args.person_detector != "ssdlite":
                raise ValueError("Grounding DINO fallback requires SSDLite primary")
            if force_primary_person_miss and not person_fallback_enabled:
                raise ValueError("Primary miss fixture requires the fallback detector")
            if rgbd_low_hit_fusion_enabled and not args.route_owner:
                raise ValueError("RGB-D low hit fusion requires the viewpoint route")
            if raw_swept_gate_enabled and (not args.route_owner or
                                           args.route_range_mode == "mid" or
                                           args.range_source != "physx"):
                raise ValueError("Raw swept gate requires routed PhysX low range")
            if held_raw_monitor_enabled and (not args.continuous_vision or
                    not args.route_owner or args.route_range_mode == "mid" or
                    args.range_source != "physx"):
                raise ValueError("Held raw monitor requires continuous routed PhysX low range")
            if (held_pose_probe or near_cube_probe or far_cube_probe or
                    far_right_cube_probe) and not (
                    held_raw_monitor_enabled and route_mid_room_start and
                    args.route_obstacle):
                raise ValueError("Held-pose cube probe requires mid-room routed low monitor")
            if held_head_look_probe and not (held_pose_probe and
                    tof_escape_enabled and args.scenario == "social_show"):
                raise ValueError("Held head look requires a social-show ToF fixed-body fixture")
            if verify_gaze_conflict_probe and not (held_head_look_probe and
                    args.response == "owner_show"):
                raise ValueError("Verification gaze conflict requires the owner-show held-head fixture")
            if verify_gaze_conflict_bypass and not verify_gaze_conflict_probe:
                raise ValueError("Verification gaze bypass requires the diagnostic conflict")
            if near_social_view and not (held_head_look_probe and
                    far_right_cube_probe and args.route_owner):
                raise ValueError("Near social view requires the right-cube held-head fixture")
            if clean_social_capture and args.scenario != "social_show":
                raise ValueError("Extra render subframes belong to social-show capture")
            if sum((near_cube_probe, far_cube_probe, far_right_cube_probe)) > 1:
                raise ValueError("Choose one fixed-eye cube start")
            if low_tof_probe_enabled and not (held_pose_probe and
                    (near_cube_probe or far_cube_probe or far_right_cube_probe)):
                raise ValueError("Low ToF probe currently requires a fixed-eye cube fixture")
            if tof_escape_enabled and not low_tof_probe_enabled:
                raise ValueError("ToF escape screen requires the low ToF stream")
            simple_room = "Isaac/Environments/Simple_Room/simple_room.usd"
            room_selected = simple_room in config_text
            # The official room's main floor mesh is translated to z=-0.8154 m.
            # Keep the people, NavMesh include volume and ape-height eye in
            # that same stage frame; the warehouse floor is at z=0.
            floor_z = -0.8154 if room_selected else 0.0
            range_heights_m = (0.15, 0.45) if args.route_owner else (0.25, 0.80)
            if room_selected:
                overlay_path = Path(args.output).parent / "simple-room-navmesh.usda"
                source_url = get_assets_root_path().rstrip("/") + "/" + simple_room
                overlay_stage = Usd.Stage.CreateNew(str(overlay_path))
                # Stage metadata belongs to the root layer; sublayer metadata
                # does not replace its default centimetre units.
                UsdGeom.SetStageMetersPerUnit(overlay_stage, 1.0)
                UsdGeom.SetStageUpAxis(overlay_stage, UsdGeom.Tokens.z)
                overlay_stage.SetTimeCodesPerSecond(60)
                overlay_stage.SetStartTimeCode(0)
                overlay_stage.SetEndTimeCode(10000)
                overlay_stage.GetRootLayer().subLayerPaths.append(source_url)
                overlay_stage.GetRootLayer().Save()
                del overlay_stage
                await omni.usd.get_context().open_stage_async(str(overlay_path))
                prepared_stage = omni.usd.get_context().get_stage()
                ok, created = omni.kit.commands.execute(
                    "CreateNavMeshVolumeCommand",
                    parent_prim_path=Sdf.Path("/Root"), volume_type=0,
                    position=Gf.Vec3d(0.6, 0.2, floor_z + 1.0))
                if not ok:
                    raise RuntimeError(f"Simple Room NavMesh volume creation failed: {created}")
                volumes = [prim for prim in prepared_stage.Traverse()
                           if prim.GetTypeName() == "NavMeshVolume"]
                if not volumes:
                    raise RuntimeError("Simple Room overlay has no NavMesh volume")
                # CreateNavMeshVolumeCommand authors a unit-cube extent,
                # so this scale gives an 8 x 8 x 3 m include volume.
                UsdGeom.XformCommonAPI(volumes[-1]).SetScale(Gf.Vec3f(8, 8, 3))
                prepared_stage.GetRootLayer().Save()
                prepared_config = Path(args.output).parent / "prepared-simple-room.yaml"
                prepared_config.write_text(config_text.replace(simple_room, str(overlay_path)))
                config_path = prepared_config
                print("PREPARED_SIMPLE_ROOM", json.dumps({"source": source_url,
                    "overlay": str(overlay_path),
                    "navmesh_volume": str(volumes[-1].GetPath())}), flush=True)
                import importlib
                import inspect

                for module_name in ("omni.metropolis.pipeline.simulation_util",
                                    "isaacsim.replicator.agent.core.scene_assembly.environment_loader"):
                    try:
                        module = importlib.import_module(module_name)
                        lines = inspect.getsource(module).splitlines()
                        selected = sorted({near for index, line in enumerate(lines)
                                           if "navmesh" in line.lower() or "100" in line
                                           for near in range(max(0, index - 2),
                                                             min(len(lines), index + 3))})
                        print("NAVMESH_SOURCE", module_name,
                              json.dumps({"path": module.__file__,
                                  "lines": [[index + 1, lines[index]] for index in selected[:100]]}),
                              flush=True)
                    except Exception as error:
                        print("NAVMESH_SOURCE_ERROR", module_name, repr(error), flush=True)
                floor = prepared_stage.GetPrimAtPath("/Root/Floor/SM_Template_Map_Floor")
                if floor.IsValid():
                    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                           [UsdGeom.Tokens.default_]).ComputeWorldBound(floor).ComputeAlignedBox()
                    print("ROOM_FLOOR_BOUNDS", json.dumps({"min": list(box.GetMin()),
                        "max": list(box.GetMax()), "stage_meters_per_unit":
                        UsdGeom.GetStageMetersPerUnit(prepared_stage)}), flush=True)
            if not ira.load_config_file(str(config_path)):
                raise RuntimeError(f"IRA rejected config {config_path}")
            if room_selected:
                # IRA 1.7.9 gives this stage only 100 app updates for NavMesh
                # baking. Navigation settings use centimetre-scaled values
                # even though this USD stage expresses positions in metres.
                import importlib
                from omni.anim.navigation.core import NavMeshSettings

                environment_loader = importlib.import_module(
                    "isaacsim.replicator.agent.core.scene_assembly.environment_loader")

                async def wait_for_room_navmesh() -> bool:
                    nav_settings = carb.settings.get_settings()
                    nav_settings.set(NavMeshSettings.USE_GPU_NAVMESH_SETTING_PATH, False)
                    nav_settings.set(NavMeshSettings.CACHE_ENABLED_SETTING_PATH, False)
                    for setting, value in (
                        (NavMeshSettings.AGENT_MIN_HEIGHT_SETTING_PATH, 180.0),
                        (NavMeshSettings.AGENT_MIN_RADIUS_SETTING_PATH, 20.0),
                        (NavMeshSettings.AGENT_MAX_RADIUS_SETTING_PATH, 45.0),
                        (NavMeshSettings.AGENT_MAX_STEP_HEIGHT_SETTING_PATH, 20.0),
                        (NavMeshSettings.AGENT_MAX_FLOOR_SLOPE_SETTING_PATH, 45.0),
                        (NavMeshSettings.AGENT_MIN_ISLAND_RADIUS_SETTING_PATH, 50.0),
                        (NavMeshSettings.AGENT_SAMPLING_DISTANCE_SETTING_PATH, 25.0),
                    ):
                        nav_settings.set(setting, value)
                    print("ROOM_NAVMESH_AGENT",
                          nav_settings.get(NavMeshSettings.AGENT_MIN_HEIGHT_SETTING_PATH),
                          nav_settings.get(NavMeshSettings.AGENT_MIN_RADIUS_SETTING_PATH),
                          nav_settings.get(NavMeshSettings.AGENT_MAX_RADIUS_SETTING_PATH),
                          "gpu", nav_settings.get(NavMeshSettings.USE_GPU_NAVMESH_SETTING_PATH),
                          flush=True)
                    inav = navigation.acquire_interface()
                    if inav.get_navmesh() is not None:
                        return True
                    started = inav.start_navmesh_baking()
                    print("ROOM_NAVMESH_BAKE_START", started, flush=True)
                    if not started:
                        return False
                    for update in range(1, 10001):
                        await omni.kit.app.get_app().next_update_async()
                        mesh = inav.get_navmesh()
                        if mesh is not None:
                            print("ROOM_NAVMESH_BAKE_READY", update,
                                  "baking", inav.is_navmesh_baking(), flush=True)
                            return True
                        if update in (100, 1000, 2000, 4000, 6000, 8000, 10000):
                            print("ROOM_NAVMESH_BAKE_WAIT", update,
                                  "baking", inav.is_navmesh_baking(), flush=True)
                    return False

                environment_loader.ensure_navmesh_ready = wait_for_room_navmesh
            try:
                await ira.setup_simulation()
            except Exception:
                if room_selected:
                    failed_stage = omni.usd.get_context().get_stage()
                    print("ROOM_AFTER_LOAD_FAILURE", json.dumps({
                        "root_layer": failed_stage.GetRootLayer().identifier,
                        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(failed_stage),
                        "floor_valid": failed_stage.GetPrimAtPath(
                            "/Root/Floor/SM_Template_Map_Floor").IsValid(),
                        "navmesh_volumes": [str(prim.GetPath()) for prim in failed_stage.Traverse()
                            if prim.GetTypeName() == "NavMeshVolume"]}), flush=True)
                    from omni.anim.navigation.core import NavMeshSettings

                    settings = carb.settings.get_settings()
                    print("ROOM_NAVMESH_SETTINGS", json.dumps({name: {
                        "path": getattr(NavMeshSettings, name),
                        "value": settings.get(getattr(NavMeshSettings, name))}
                        for name in dir(NavMeshSettings) if name.endswith("SETTING_PATH")}),
                        flush=True)
                    kit_log_dir = Path("/tmp/robot-lab-isaac-env/lib/python3.12/"
                                       "site-packages/isaacsim/kit/logs/Kit/Isaac-Sim Python/6.1")
                    kit_logs = sorted(kit_log_dir.glob("kit_*.log"),
                                      key=lambda path: path.stat().st_mtime)
                    if kit_logs:
                        relevant = [line.strip() for line in kit_logs[-1].read_text(
                            errors="replace").splitlines() if
                            "navmesh" in line.lower() or "navigation" in line.lower()
                            or "[Error]" in line]
                        print("ROOM_KIT_NAV_LOG", json.dumps(relevant[-80:]), flush=True)
                raise
            app = omni.kit.app.get_app()
            for _ in range(10):
                await app.next_update_async()
            if path_tracing_probe:
                render_settings = carb.settings.get_settings()
                render_settings.set("/rtx/rendermode", "PathTracing")
                print("RENDER_MODE_PROBE", json.dumps({
                    "mode": render_settings.get("/rtx/rendermode"),
                    "samples_per_pixel_per_frame": render_settings.get(
                        "/rtx/pathtracing/spp"),
                    "optix_denoiser": render_settings.get(
                        "/rtx/pathtracing/optixDenoiser/enabled"),
                    "denoiser_blend": render_settings.get(
                        "/rtx/pathtracing/optixDenoiser/blendFactor")}), flush=True)

            stage = omni.usd.get_context().get_stage()
            route_cube_initial_xy = ([.47, -2.15] if near_social_view else
                                     [.92, -2.60] if far_right_cube_probe else
                                     [-.52, -2.60] if far_cube_probe else
                                     [-.22, -2.60] if near_cube_probe else
                                     [-.22, -2.35] if route_mid_room_start else
                                     [-.22, -2.65])
            if args.route_obstacle:
                cube = UsdGeom.Cube.Define(stage, "/World/ApeRouteBlock")
                cube.CreateSizeAttr(.32)
                cube.CreateDisplayColorAttr([Gf.Vec3f(.8, .28, .08)])
                UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(
                    Gf.Vec3d(*route_cube_initial_xy, floor_z + .16))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                print("APE_ROUTE_BLOCK", json.dumps({"center_xyz_m":
                    [*route_cube_initial_xy, floor_z + .16], "size_m": .32}), flush=True)
            route_cube_center_xy = route_cube_initial_xy.copy() if args.route_obstacle else None
            continuous_cube_start_s = None

            def update_continuous_cube(now_s: float) -> None:
                if dynamic_low_cube_mode not in ("continuous", "continuous_fast") or continuous_cube_start_s is None:
                    return
                speed_m_s = .24 if dynamic_low_cube_mode == "continuous_fast" else .15
                shift = min(.50 if (far_cube_probe or far_right_cube_probe) else .20,
                            max(0., speed_m_s * (now_s - continuous_cube_start_s)))
                desired_x = route_cube_initial_xy[0] + (
                    -shift if far_right_cube_probe else shift)
                if abs(desired_x - route_cube_center_xy[0]) > 1e-6:
                    route_cube_center_xy[0] = desired_x
                    UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(
                        Gf.Vec3d(desired_x, route_cube_center_xy[1], floor_z + .16))
            root = stage.GetPrimAtPath("/World/Characters")
            cache = UsdGeom.XformCache()
            characters = []
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() == "SkelRoot":
                    position = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                    layers = [spec.layer.identifier for spec in prim.GetPrimStack()]
                    characters.append({"prim_path": str(prim.GetPath()),
                                       "xyz_m": list(position), "source_layers": layers})
            print("RIGGED_CHARACTERS", json.dumps(characters), flush=True)
            if len(characters) < 2:
                raise RuntimeError(f"Expected two rigged characters, found {len(characters)}")
            crossing = args.scenario == "crossing"
            social_show = args.scenario == "social_show"
            active_look = args.scenario in ("active_look", "social_show")
            requested_target = ((-0.2, characters[1]["xyz_m"][1], floor_z)
                                if crossing else ((-1.4, -1.7, floor_z)
                                if social_show and args.distractor_motion == "cross" else
                                ((1.4 if room_selected else 1.6),
                                                  (-1.35 if room_selected else 0.35),
                                                  floor_z)))
            camera_distance_m = (.75 if near_social_view else
                                 1.2 if route_mid_room_start else
                                 1.5 if room_selected else
                                 2.0 if crossing else 1.5)
            camera_target_height_m = (.95 if near_social_view else
                                      .70 if room_selected else 1.0)
            camera_follow_target_z_m = (floor_z + .45 + 1.5 *
                (camera_target_height_m - .45) / camera_distance_m
                if near_social_view else floor_z + .70)
            simulation_steps_per_frame = 15 if crossing else 6
            inav = navigation.acquire_interface()
            volume_paths = [str(prim.GetPath()) for prim in stage.Traverse()
                            if prim.GetTypeName() == "NavMeshVolume"]
            if not volume_paths:
                ok, created = omni.kit.commands.execute(
                    "CreateNavMeshVolumeCommand",
                    parent_prim_path=Sdf.Path("/World"), volume_type=0,
                    position=Gf.Vec3d(0.6, 0.2, 1.0))
                if not ok:
                    raise RuntimeError(f"Cannot create NavMesh include volume: {created}")
                volume_paths = [str(prim.GetPath()) for prim in stage.Traverse()
                                if prim.GetTypeName() == "NavMeshVolume"]
                if not volume_paths:
                    raise RuntimeError("NavMesh include command returned no volume")
                UsdGeom.XformCommonAPI(stage.GetPrimAtPath(volume_paths[-1])).SetScale(
                    Gf.Vec3f(.08, .08, .03))
            print("NAVMESH_VOLUMES", json.dumps(volume_paths), flush=True)
            inav.start_navmesh_baking_and_wait()
            navmesh = inav.get_navmesh()
            if navmesh is None:
                raise RuntimeError("NavMesh bake returned no mesh")
            if room_selected:
                print("ROOM_NAVMESH_SURFACE", json.dumps({
                    "areas": [[navmesh.get_area_name(index),
                               len(navmesh.get_draw_triangles(index))]
                              for index in range(navmesh.get_area_count())],
                    "agent_min_height": navmesh.get_agent_min_height(),
                    "agent_min_radius": navmesh.get_agent_min_radius(),
                    "agent_max_radius": navmesh.get_agent_max_radius()}), flush=True)
                kit_log_dir = Path("/tmp/robot-lab-isaac-env/lib/python3.12/"
                                   "site-packages/isaacsim/kit/logs/Kit/Isaac-Sim Python/6.1")
                kit_logs = sorted(kit_log_dir.glob("kit_*.log"),
                                  key=lambda path: path.stat().st_mtime)
                if kit_logs:
                    lines = [line.strip() for line in kit_logs[-1].read_text(
                        errors="replace").splitlines() if
                        "omni.anim.navigation.core.plugin" in line]
                    print("ROOM_NAVMESH_BUILD_LOG", json.dumps(lines[-100:]), flush=True)
                grid = []
                for x in (-3., -2., -1., 0., 1., 2., 3.):
                    for y in (-3., -2., -1., 0., 1., 2., 3.):
                        point, island = navmesh.query_closest_point(
                            target=carb.Float3(x, y, floor_z))
                        if point is not None:
                            grid.append({"requested_xy": [x, y],
                                "closest_xyz": [point.x, point.y, point.z],
                                "island": island})
                print("ROOM_NAVMESH_GRID", json.dumps(grid), flush=True)
                inav.set_random_seed("ape-room-diagnostic", 761901)
                random_points = []
                for _ in range(8):
                    point = navmesh.query_random_point("ape-room-diagnostic")
                    if point is not None:
                        random_points.append([point.x, point.y, point.z])
                print("ROOM_NAVMESH_RANDOM", json.dumps(random_points), flush=True)
            start_xyz = characters[1]["xyz_m"]
            start_point, start_island = navmesh.query_closest_point(
                target=carb.Float3(*start_xyz))
            end_point, end_island = navmesh.query_closest_point(
                target=carb.Float3(*requested_target))
            if start_point is None or end_point is None or start_island != end_island:
                raise RuntimeError(f"NavMesh cannot connect character start and target: {start_island}, {end_island}")
            nav_path = navmesh.query_shortest_path(start_pos=start_point, end_pos=end_point)
            if nav_path is None or nav_path.get_point_count() < 2:
                raise RuntimeError("NavMesh returned no walkable path")
            print("NAVMESH_PATH", json.dumps({"start_island": start_island,
                "end_island": end_island,
                "start_xyz_m": [start_point.x, start_point.y, start_point.z],
                "target_xyz_m": [end_point.x, end_point.y, end_point.z],
                "length_m": nav_path.length(),
                "point_count": nav_path.get_point_count()}), flush=True)
            if social_show and not early_asset_converter:
                # Keep can conversion clear of the NavMesh bake, but finish
                # enabling the extension before creating RGB-D render products.
                enable_extension("omni.kit.asset_converter")
                for _ in range(3):
                    await app.next_update_async()
            midpoint = [(characters[0]["xyz_m"][i] + characters[1]["xyz_m"][i]) / 2
                        for i in range(2)]
            if args.route_owner:
                # Keep the eye to the right of the direct owner line. The
                # earlier back-wall placement is too close to a colliding
                # room rim for a larger route margin; the optional mid-room
                # scene moves both the eye and cube forward together.
                midpoint[0] += .20
            if near_social_view:
                # Change item pixel scale while preserving the cube's initial
                # bearing and range relative to the eye.
                midpoint[0] -= .45
            eye = Gf.Vec3d(midpoint[0], midpoint[1] - camera_distance_m, floor_z + 0.45)
            target = Gf.Vec3d(midpoint[0], midpoint[1],
                              floor_z + camera_target_height_m)
            Path(args.scene_info).write_text(json.dumps({"characters": characters,
                "scenario": args.scenario, "requested_target_xyz_m": list(requested_target),
                "distractor_motion": args.distractor_motion,
                "reference_floor_z_m": floor_z,
                "simulation_steps_per_frame": simulation_steps_per_frame,
                "camera_eye_xyz_m": list(eye), "camera_target_xyz_m": list(target),
                "range_heights_above_floor_m": list(range_heights_m),
                "route_obstacle_evaluator_only": {
                    "center_xy_m": route_cube_initial_xy.copy(),
                    "size_xy_m": [.32, .32]}
                    if args.route_obstacle else None,
                "horizontal_fovs_deg": [70, 90]}, indent=2))
            view = Gf.Matrix4d(1.0)
            view.SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
            world_from_camera = view.GetInverse()
            aperture_mm = 20.955
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            rep.orchestrator.set_capture_on_play(False)
            cameras = []
            for horizontal_fov in (70, 90):
                path = f"/World/Cameras/ape_social_{horizontal_fov}deg"
                camera = UsdGeom.Camera.Define(stage, path)
                camera.CreateHorizontalApertureAttr(aperture_mm)
                camera.CreateVerticalApertureAttr(aperture_mm * 3 / 4)
                camera.CreateFocalLengthAttr(
                    aperture_mm / (2 * math.tan(math.radians(horizontal_fov / 2)))
                )
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
                transform_op = UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
                transform_op.Set(world_from_camera)
                print(f"APE_CAMERA {path} eye={tuple(eye)} target={tuple(target)}", flush=True)
                product = rep.create.render_product(path, (640, 480), name=f"ape_{horizontal_fov}")
                rgb = rep.annotators.get("rgb")
                depth = rep.annotators.get("distance_to_camera")
                rgb.attach(product)
                depth.attach(product)
                camera_dir = output / f"ape_social_{horizontal_fov}deg"
                (camera_dir / "rgb").mkdir(parents=True)
                (camera_dir / "distance_to_camera").mkdir()
                (camera_dir / "camera_params").mkdir()
                (camera_dir / "camera_params/camera_params_00000.json").write_text(
                    json.dumps({"eye_xyz_m": list(eye), "target_xyz_m": list(target),
                        "horizontal_fov_deg": horizontal_fov,
                        "resolution_wh": [640, 480]}, indent=2))
                cameras.append((camera_dir, product, rgb, depth, transform_op))
            lidar_sensor = None
            lidar_mount_pose = None
            if args.rtx_lidar:
                from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

                lidar_mount = UsdGeom.Xform.Define(stage, "/World/ApeHeadLidarMount")
                lidar_mount_pose = UsdGeom.XformCommonAPI(lidar_mount.GetPrim())
                if not lidar_mount_pose.SetTranslate(eye):
                    raise RuntimeError("RTX lidar mount pose failed")
                lidar = Lidar.create(path="/World/ApeHeadLidarMount/Lidar",
                    config="OS1", variant="OS1_REV6_32ch20hz512res",
                    aux_output_level="FULL")
                lidar_sensor = LidarSensor(lidar, annotators=["generic-model-output"])
                print("APE_RTX_LIDAR_READY", "mount_xyz_m", list(eye), flush=True)
            for _ in range(10):
                await app.next_update_async()
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(12):
                await app.next_update_async()
            behavior_interface = behavior.acquire_interface()
            actors = [behavior_interface.get_agent(row["prim_path"])
                      for row in characters]
            if any(agent is None for agent in actors):
                raise RuntimeError("Runtime behavior agent missing after timeline play")
            print("MOVE_AGENT_ENABLED", actors[1].is_enabled(), flush=True)
            move_id = actors[1].move_to(target=(end_point.x, end_point.y, end_point.z))
            if move_id == behavior.BEHAVIOR_TASK_ID_INVALID:
                raise RuntimeError("Runtime distractor move task rejected")
            print("MOVE_TASK", move_id,
                  "immediate_status", str(actors[1].get_task_status(move_id)), flush=True)
            high_level_actor = None
            show_actor = None
            if active_look:
                try:
                    import hashlib
                    from huggingface_hub import snapshot_download
                    from online_owner_actor import OnlineOwnerActor
                    import torch
                    if args.person_detector == "ssdlite":
                        torch.set_num_threads(4)

                    models = [("features", "facebook/dinov2-small",
                               "ed25f3a31f01632728cabb09d1542f84ab7b0056")]
                    if (args.person_detector == "grounding_dino" or social_show or
                            person_fallback_enabled):
                        models.insert(0, ("detector", "IDEA-Research/grounding-dino-tiny",
                                          "25b0b43915cfc9e5dcb573031e99514eb7838d57"))
                    local_models = {}
                    model_info = {}
                    for name, repo_id, revision in models:
                        print("ACTIVE_MODEL_FETCH_START", name, repo_id, revision, flush=True)
                        local_path = Path(snapshot_download(
                            repo_id=repo_id, revision=revision,
                            allow_patterns=["*.json", "*.txt", "*.safetensors"],
                            cache_dir="/tmp/robot-lab-hf"))
                        weights = local_path / "model.safetensors"
                        model_info[name] = {"repo": repo_id, "revision": revision,
                            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
                        local_models[name] = local_path
                        print("ACTIVE_MODEL_FETCH_READY", name, flush=True)
                    if (args.person_detector == "grounding_dino" or social_show or
                            person_fallback_enabled):
                        # Upstream config uses a text type unsupported by the
                        # worker's Transformers version; retain the proven config.
                        compatible = Path(args.output).parent.parent / "compatible-detector"
                        compatible.mkdir()
                        for source in local_models["detector"].iterdir():
                            if source.is_file() and source.name != "config.json":
                                (compatible / source.name).symlink_to(source.resolve())
                        config_data = (Path(__file__).parent /
                                       "grounding-dino-tiny-working-config.json").read_bytes()
                        (compatible / "config.json").write_bytes(config_data)
                        model_info["detector"]["working_config_sha256"] = (
                            hashlib.sha256(config_data).hexdigest())
                        local_models["detector"] = compatible
                    high_level_actor = OnlineOwnerActor(
                        local_models.get("detector", Path(".")), local_models["features"],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        person_detector=args.person_detector,
                        fallback_detector=(local_models["detector"]
                            if person_fallback_enabled else None))
                    if args.person_detector == "ssdlite":
                        weights = (Path(torch.hub.get_dir()) / "checkpoints" /
                                   "ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
                        model_info["person_detector"] = {"torchvision": "0.26.0+cu128",
                            "weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
                            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                            "torch_threads": torch.get_num_threads()}
                    (output.parent / "online-model-info.json").write_text(
                        json.dumps(model_info, indent=2))
                    print("ACTIVE_ACTOR_READY", high_level_actor.device,
                          args.person_detector, flush=True)
                except Exception:
                    import traceback
                    print("ACTIVE_ACTOR_INIT_ERROR", traceback.format_exc(), flush=True)
                    raise
            shown_item_xform = None
            shown_item_xyz = None
            if social_show:
                import omni.kit.asset_converter as converter
                from pxr import UsdShade
                from social_show_actor import SocialShowActor

                source = Path(__file__).parent
                item_usd = output.parent / "TomatoSauce-converted.usd"
                # The HOPE OBJ records centimetre-sized coordinates. Bake metres
                # into the vertices before conversion: a referenced USD may reset
                # its root transform and ignore a parent-scale operation.
                item_obj_m = source / "TomatoSauce-meters.obj"
                with (source / "TomatoSauce.obj").open() as original, item_obj_m.open("w") as metric:
                    for line in original:
                        if line.startswith("v "):
                            parts = line.split()
                            line = "v " + " ".join(f"{float(value) * .01:.9g}" for value in parts[1:4])
                            if len(parts) > 4:
                                line += " " + " ".join(parts[4:])
                            line += "\n"
                        metric.write(line)
                context = converter.AssetConverterContext()
                context.ignore_material = False
                context.use_meter_as_world_unit = True
                convert_task = converter.get_instance().create_converter_task(
                    str(item_obj_m), str(item_usd),
                    lambda *_: None, context)
                if not await convert_task.wait_until_finished():
                    raise RuntimeError(f"HOPE TomatoSauce conversion failed: {convert_task.get_status()}")
                item = UsdGeom.Xform.Define(stage, "/World/NewShowCan")
                # The converter's default prim authors its own xformOp:translate.
                # Give the reference a separate parent so our show/hide pose is
                # an independent, stronger transform rather than a failed edit
                # of that referenced operation.
                item_asset = UsdGeom.Xform.Define(stage, "/World/NewShowCan/Asset")
                item_asset.GetPrim().GetReferences().AddReference(str(item_usd))
                shown_item_xform = UsdGeom.XformCommonAPI(item.GetPrim())
                # The HOPE OBJ's long axis is +Y; stand the can on +Z.
                if not shown_item_xform.SetRotate(Gf.Vec3f(90, 0, 0)):
                    raise RuntimeError("Failed to orient the shown item")
                if not shown_item_xform.SetTranslate(Gf.Vec3d(0, 0, -100)):
                    raise RuntimeError("Failed to hide the show item at setup")
                item_meshes = [prim for prim in stage.Traverse()
                               if str(prim.GetPath()).startswith("/World/NewShowCan/Asset/")
                               and prim.IsA(UsdGeom.Mesh)]
                if len(item_meshes) != 1:
                    raise RuntimeError(f"Expected one converted can mesh, got {len(item_meshes)}")
                # The OBJ converter retains face-varying UVs but drops this
                # HOPE asset's MTL/JPG binding. Author the standard USD preview
                # graph explicitly so image retrieval sees the taught texture.
                material = UsdShade.Material.Define(stage, "/World/Looks/NewShowCan")
                surface = UsdShade.Shader.Define(stage, "/World/Looks/NewShowCan/Surface")
                surface.CreateIdAttr("UsdPreviewSurface")
                surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.65)
                uv = UsdShade.Shader.Define(stage, "/World/Looks/NewShowCan/UV")
                uv.CreateIdAttr("UsdPrimvarReader_float2")
                uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                texture = UsdShade.Shader.Define(stage, "/World/Looks/NewShowCan/Texture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    str(source / "TomatoSauce.jpg"))
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    uv.ConnectableAPI(), "result")
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb")
                material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(item_meshes[0]).Bind(material)
                if not UsdShade.MaterialBindingAPI(item_meshes[0]).ComputeBoundMaterial()[0]:
                    raise RuntimeError("HOPE can texture material did not bind")
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                           [UsdGeom.Tokens.default_]).ComputeWorldBound(
                                               item.GetPrim()).ComputeAlignedBox()
                size = bounds.GetMax() - bounds.GetMin()
                print("SHOW_ITEM_READY", json.dumps({"usd": str(item_usd),
                    "scaled_size_xyz_m": list(size)}), flush=True)
                if not .03 <= max(size) <= .20:
                    raise RuntimeError(f"Converted item scale is implausible: {list(size)}")
                shown_item_xyz = [float(characters[0]["xyz_m"][0] + .35),
                                  float(characters[0]["xyz_m"][1] - .32),
                                  float(floor_z + .74)]
                show_actor = SocialShowActor(high_level_actor,
                    Image.open(source / "taught-rgb-crop.png").convert("RGB"),
                    object_detector_path=local_models.get("detector"))
            motion_trace = []
            action_trace = []
            previous_command = None
            range_trace = []
            range_query = omni.physx.get_physx_scene_query_interface() if room_selected else None
            if active_look and range_query is None:
                raise RuntimeError("active_look requires the furnished room range stream")
            range_angles_deg = list(range(-90, 91, 5))
            low_blind_mask = (np.ones(len(range_angles_deg), dtype=bool)
                if low_blind_mode == "all_from_frame9" else
                (np.asarray(range_angles_deg) >= -70) &
                (np.asarray(range_angles_deg) <= 0))

            def low_blind_active(frame_index: int) -> bool:
                return (low_blind_mode == "left_70deg" or
                        (low_blind_mode == "all_from_frame9" and frame_index >= 9))
            heading_deg = 0.0
            body_heading_deg = 0.0
            applied_yaw_delta_deg = 0.0
            social_search_looks = 0
            social_failed_item_views = 0
            social_clear_view_requested = False
            social_verify_anchor_frame = None
            proxy_eye = [float(eye[0]), float(eye[1]), float(eye[2])]
            visual_odometry = None
            if args.visual_odometry:
                from rgbd_visual_odometry import RgbdVisualOdometry
                visual_odometry = RgbdVisualOdometry(
                    np.asarray(proxy_eye[:2]), args.control_fov,
                    odometry_scale=args.odometry_scale)
            view_route = None
            low_motion_guard = None
            path_timed_guard = None
            last_guard_visual_time_s = None
            if args.route_owner:
                from fan_view_route import (FanViewRoute, LowRangeClosingGuard,
                                            PathTimedOccupancyGuard)
                view_route = FanViewRoute(
                    proxy_radius_m=route_proxy_radius_m,
                    planning_margin_m=route_planning_margin_m,
                    required_free_planes=((0, 1) if args.route_range_mode == "both"
                        else (0,) if args.route_range_mode == "low" else (1,))
                        if require_height_coverage else (),
                    required_free_age_s=(.4 if require_height_coverage else 4.))
                if low_motion_guard_enabled:
                    low_motion_guard = LowRangeClosingGuard(
                        warning_clearance_m=view_route.plan_clearance_m + .04,
                        minimum_closure_m=(.005 if route_proxy_radius_m >= .20
                                           else .012),
                        minimum_nearest_m=(view_route.plan_clearance_m
                            if route_planning_margin_m else None))
                if path_timed_guard_enabled:
                    path_timed_guard = PathTimedOccupancyGuard(
                        clearance_m=view_route.plan_clearance_m)
            frame_count = args.frame_count or (20 if social_show else 12)
            control_camera = f"ape_social_{args.control_fov}deg"
            social_scene_trace = []
            lidar_trace = []
            lidar_snapshots = {}
            tof_trace = []
            next_tof_scan_s = -math.inf
            tof_response_trace = []
            tof_supervisor = None
            if low_tof_probe_enabled:
                from near_body_tof import SAMPLE_PERIOD_S, MOUNTS, sample as sample_low_tof
            if tof_escape_enabled:
                from tof_intrusion_response import StationaryTofIntrusionResponse
                tof_supervisor = StationaryTofIntrusionResponse(tof_escape_distance_m)

            def sample_tof_if_due(now_s: float) -> None:
                nonlocal next_tof_scan_s
                if not low_tof_probe_enabled or now_s + 1e-6 < next_tof_scan_s:
                    return
                packet = sample_low_tof(range_query, carb,
                    tuple(proxy_eye[:2]), body_heading_deg, floor_z, now_s,
                    tuple(route_cube_center_xy) if route_cube_center_xy is not None else None)
                tof_trace.append(packet)
                next_tof_scan_s = now_s + SAMPLE_PERIOD_S
                if tof_supervisor is None:
                    return
                # The controller receives only actor-visible range/height
                # values; prim/cube labels and exact positions stay in audit.
                flank_sensors = {}
                for mount in ("front_left", "front_right"):
                    source = packet["sensors"][mount]
                    flank_sensors[mount] = {
                        "range_m_by_zone": source["range_m_by_zone"],
                        "hit_height_above_floor_m_by_zone": source[
                            "hit_height_above_floor_m_by_zone"]}
                cue = tof_supervisor.observe({"sim_time_s": now_s,
                    "sensors": flank_sensors})
                escape_planes = None
                escape_angles = None
                if (tof_supervisor.alarm is not None and
                        now_s >= tof_supervisor.alarm["time_s"] + .2 and
                        tof_supervisor.escape_m < tof_supervisor.escape_distance_m - 1e-8):
                    # A separate, ideal all-around 15/45 cm PhysX source
                    # supplies only the provisional swept escape check. The
                    # existing route fan spans ±90° and cannot certify rear.
                    escape_angles = list(range(-180, 181, 5))
                    escape_planes = []
                    for height_m in (.15, .45):
                        origin = carb.Float3(proxy_eye[0], proxy_eye[1],
                                             floor_z + height_m)
                        distances = []
                        for relative_deg in escape_angles:
                            bearing = math.radians(body_heading_deg + relative_deg)
                            hit = range_query.raycast_closest(origin,
                                carb.Float3(math.sin(bearing), math.cos(bearing), 0.), 8.)
                            distances.append(float(hit["distance"]) if hit["hit"]
                                             else math.inf)
                        escape_planes.append(distances)
                response = tof_supervisor.step(now_s, escape_planes, escape_angles)
                increment = response["translation_increment_m"]
                if increment > 0:
                    heading = math.radians(body_heading_deg +
                                           response["heading_rel_deg"])
                    proxy_eye[0] += math.sin(heading) * increment
                    proxy_eye[1] += math.cos(heading) * increment
                    camera_heading = math.radians(heading_deg)
                    next_eye = Gf.Vec3d(*proxy_eye)
                    next_target = Gf.Vec3d(
                        proxy_eye[0] + 1.5 * math.sin(camera_heading),
                        proxy_eye[1] + 1.5 * math.cos(camera_heading),
                        camera_follow_target_z_m)
                    next_view = Gf.Matrix4d(1.0)
                    next_view.SetLookAt(next_eye, next_target, Gf.Vec3d(0, 0, 1))
                    for _, _, _, _, transform_op in cameras:
                        transform_op.Set(next_view.GetInverse())
                    if lidar_mount_pose is not None:
                        lidar_mount_pose.SetTranslate(next_eye)
                tof_response_trace.append({"time_s": now_s,
                    "cue": cue, "response": response,
                    "eye_xy_m_evaluator_only": proxy_eye[:2].copy()})
            pickup_task_id = None
            pickup_actor_index = None
            show_walk_id = None
            show_walk_plan = None
            show_walk_near_frames = 0
            show_pickup_trigger = None
            clear_view_reach_id = None
            clear_view_response_plan = None
            inference_pool = ThreadPoolExecutor(max_workers=1) if args.continuous_vision else None
            vision_monitor_trace = []

            async def decide_while_world_moves(rgb_data, depth_data, ranges,
                                               angles, observation_time, baseline_row,
                                               social_data=None, capture_started_wall=None):
                """Advance Isaac and range while one sensor packet is inferred."""
                prior_bearing = high_level_actor.last_bearing_deg
                sector_center = float(prior_bearing if prior_bearing is not None else 0.0)
                monitor_planes = ((0,) if args.route_range_mode == "low" else
                                  (1,) if args.route_range_mode == "mid" else (0, 1))
                baseline = {plane: np.asarray([
                    beam["distance_m"] if beam["distance_m"] is not None else np.nan
                    for beam in baseline_row["planes"][plane]["beams"]], dtype=np.float32)
                    for plane in monitor_planes}
                if low_blind_active(frame) and 0 in baseline:
                    baseline[0][low_blind_mask] = np.nan
                sector = np.abs(angles - sector_center) <= 35.0
                monitor_rows = []
                alarm = False
                held_alarm = False
                initial_held_gate = None
                if held_raw_monitor_enabled:
                    from raw_swept_gate import assess_swept_low_hits
                    initial_low = [beam["distance_m"] if beam["distance_m"] is not None
                                   else math.inf for beam in baseline_row["planes"][0]["beams"]]
                    if low_blind_active(frame):
                        initial_low = [None if low_blind_mask[index] else value
                                       for index, value in enumerate(initial_low)]
                    initial_held_gate = assess_swept_low_hits(initial_low,
                        range_angles_deg, step_m=0., step_heading_rel_deg=0.,
                        clearance_m=.24)
                    held_alarm = (initial_held_gate["veto"] or
                                  initial_held_gate["status"] == "unknown")
                streak = 0
                started = time.perf_counter()

                def infer():
                    worker_started = time.perf_counter()
                    use_fast = (args.fast_owner_flow and high_level_actor.fast_flow_ready and
                                high_level_actor.fast_frames_since_detection < 3)
                    if use_fast:
                        result = high_level_actor.observe_fast(
                            rgb_data, depth_data, ranges, angles,
                            time_s=observation_time, fov_deg=args.control_fov,
                            ego_yaw_delta_deg=applied_yaw_delta_deg)
                    else:
                        result = high_level_actor.observe(
                            rgb_data, depth_data, ranges, angles,
                            time_s=observation_time, fov_deg=args.control_fov,
                            ego_yaw_delta_deg=applied_yaw_delta_deg)
                        proposal_source = ((high_level_actor.last_proposal_timing or {}).get(
                            "detector_source", args.person_detector)
                            if "proposals" in result else args.person_detector)
                        result["observation_method"] = proposal_source + "_dinov2"
                        # The first model call is a cold start; its frame can be
                        # >1 s old before the next capture, beyond the flow gap.
                        if args.fast_owner_flow and frame > 0:
                            result["fast_flow_seed"] = high_level_actor.seed_fast_flow(
                                rgb_data, depth_data, result,
                                time_s=observation_time, fov_deg=args.control_fov)
                    owner_wall_s = time.perf_counter() - worker_started
                    social_result = None
                    if social_data is not None:
                        show_rgb, show_depth = social_data
                        social_started = time.perf_counter()
                        social_result = show_actor.observe(
                            show_rgb, show_depth, time_s=observation_time,
                            frame_age_s=time.perf_counter() - capture_started_wall,
                            fov_deg=args.social_fov)
                        social_result["social_inference_wall_s"] = (
                            time.perf_counter() - social_started)
                    return (result, social_result, owner_wall_s,
                            time.perf_counter() - worker_started)

                future = inference_pool.submit(infer)
                next_scan = observation_time + .1
                updates = 0
                decision_and_latency = None
                advanced_before_done = None
                while True:
                    if future.done() and decision_and_latency is None:
                        decision_and_latency = future.result()
                        advanced_before_done = timeline.get_current_time() - observation_time
                    if (decision_and_latency is not None and
                            timeline.get_current_time() - observation_time >=
                            decision_and_latency[3]):
                        break
                    if time.perf_counter() - started > 900.0 or updates >= 900:
                        raise RuntimeError(f"Vision/world schedule exceeded budget: "
                            f"updates={updates}, elapsed_wall_s={time.perf_counter()-started:.2f}, "
                            f"advanced_sim_s={timeline.get_current_time()-observation_time:.3f}")
                    await rep.orchestrator.step_async(rt_subframes=1)
                    updates += 1
                    update_continuous_cube(timeline.get_current_time())
                    cube_delta = (.20 if dynamic_low_cube_mode == "instant" and
                                  frame == 9 and updates == 4 else
                                  .03 if dynamic_low_cube_mode == "creep_fast" and
                                  9 <= frame <= 18 and updates == 1 else
                                  .02 if dynamic_low_cube_mode == "creep" and
                                  9 <= frame <= 18 and updates == 1 else 0.)
                    if cube_delta:
                        route_cube_center_xy[0] += cube_delta
                        UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(
                            Gf.Vec3d(route_cube_center_xy[0],
                                      route_cube_center_xy[1], floor_z + .16))
                        print("DYNAMIC_ROUTE_CUBE", json.dumps({"frame": frame,
                            "time_s": timeline.get_current_time(),
                            "delta_x_m": cube_delta,
                            "center_xy_m_evaluator_only": route_cube_center_xy}), flush=True)
                    now_s = timeline.get_current_time()
                    sample_tof_if_due(now_s)
                    if now_s + 1e-6 < next_scan:
                        continue
                    current = {}
                    held_gate = None
                    changed_counts = {}
                    nearest = {}
                    for plane in monitor_planes:
                        origin = carb.Float3(proxy_eye[0], proxy_eye[1],
                                             floor_z + range_heights_m[plane])
                        distances = []
                        for angle_deg in range_angles_deg:
                            angle = math.radians(angle_deg + (
                                body_heading_deg if view_route is not None else heading_deg))
                            hit = range_query.raycast_closest(
                                origin, carb.Float3(math.sin(angle), math.cos(angle), 0.0), 8.0)
                            distances.append(float(hit["distance"]) if hit["hit"]
                                             else math.inf)
                        distances = np.asarray(distances, dtype=np.float32)
                        if low_blind_active(frame) and plane == 0:
                            distances[low_blind_mask] = np.nan
                        if held_raw_monitor_enabled and plane == 0:
                            held_gate = assess_swept_low_hits(
                                [None if np.isnan(value) else float(value)
                                 for value in distances], range_angles_deg,
                                step_m=0., step_heading_rel_deg=0., clearance_m=.24)
                            held_alarm |= (held_gate["veto"] or
                                           held_gate["status"] == "unknown")
                        current[plane] = distances
                        changed = sector & (((np.isfinite(distances) &
                            np.isfinite(baseline[plane]) &
                            (distances + .10 < baseline[plane]))) |
                            (np.isfinite(distances) & ~np.isfinite(baseline[plane]) &
                             (distances < 2.5)))
                        changed_counts[plane] = int(changed.sum())
                        finite_sector = distances[sector & np.isfinite(distances)]
                        nearest[plane] = (float(finite_sector.min())
                                          if finite_sector.size else None)
                    changed_count = max(changed_counts.values())
                    streak = streak + 1 if changed_count >= 3 else 0
                    alarm |= streak >= 2
                    monitor_rows.append({"time_s": float(now_s),
                        "wall_elapsed_s": time.perf_counter() - started,
                        "changed_sector_beams": changed_count,
                        "changed_sector_beams_by_plane": changed_counts,
                        "change_streak": streak,
                        "held_occupied_gate": held_gate,
                        "nearest_sector_m_by_plane": nearest,
                        "valid_bins_by_plane": {plane: int(np.isfinite(values).sum())
                            for plane, values in current.items()},
                        "range_m_by_plane": {plane: [float(value) if np.isfinite(value)
                            else None for value in values]
                            for plane, values in current.items()},
                        "route_cube_center_xy_m_evaluator_only": (
                            route_cube_center_xy.copy() if route_cube_center_xy is not None
                            else None),
                        "character_xyz_m_evaluator_only": [
                            [float(getattr(agent.get_world_translation(False), axis))
                             for axis in ("x", "y", "z")]
                            for agent in actors]})
                    next_scan += .1
                decision, social_result, owner_wall_s, inference_wall_s = decision_and_latency
                final_held_gate = None
                decision_scan_time_s = None
                if held_raw_monitor_enabled:
                    # Bridge the inference-to-next-capture gap. Otherwise a
                    # nominal 100 ms inference scan can leave ~180 ms with no
                    # low observation across the command boundary.
                    decision_scan_time_s = timeline.get_current_time()
                    origin = carb.Float3(proxy_eye[0], proxy_eye[1],
                                         floor_z + range_heights_m[0])
                    final_low = []
                    for angle_deg in range_angles_deg:
                        bearing = math.radians(body_heading_deg + angle_deg)
                        hit = range_query.raycast_closest(origin,
                            carb.Float3(math.sin(bearing), math.cos(bearing), 0.), 8.)
                        final_low.append(float(hit["distance"]) if hit["hit"]
                                         else math.inf)
                    if low_blind_active(frame):
                        final_low = [None if low_blind_mask[index] else value
                                     for index, value in enumerate(final_low)]
                    final_held_gate = assess_swept_low_hits(final_low,
                        range_angles_deg, step_m=0., step_heading_rel_deg=0.,
                        clearance_m=.24)
                    held_alarm |= (final_held_gate["veto"] or
                                   final_held_gate["status"] == "unknown")
                return decision, social_result, owner_wall_s, inference_wall_s, (
                    alarm or held_alarm), {
                    "source_time_s": observation_time,
                    "decision_time_s": timeline.get_current_time(),
                    "source_age_s": timeline.get_current_time() - observation_time,
                    "inference_wall_s": inference_wall_s,
                    "advanced_before_inference_finished_s": advanced_before_done,
                    "prior_owner_bearing_deg": sector_center,
                    "monitor_heights_above_floor_m": [range_heights_m[plane]
                                                       for plane in monitor_planes],
                    "monitor_alarm": alarm or held_alarm,
                    "person_sector_alarm": alarm,
                    "held_raw_alarm": held_alarm,
                    "initial_held_gate": initial_held_gate,
                    "decision_held_gate": final_held_gate,
                    "decision_scan_time_s": decision_scan_time_s,
                    "samples": monitor_rows}

            for frame in range(frame_count):
                lidar_range_bins = None
                if args.staged_walk_show and frame == 9:
                    presenter = actors[0].get_world_translation(False)
                    start_point, start_island = navmesh.query_closest_point(
                        target=carb.Float3(presenter.x, presenter.y, floor_z))
                    bearing = math.radians(heading_deg - 20.)
                    desired_xy = (proxy_eye[0] + math.sin(bearing) * 1.8,
                                  proxy_eye[1] + math.cos(bearing) * 1.8)
                    target_point, target_island = navmesh.query_closest_point(
                        target=carb.Float3(*desired_xy, floor_z))
                    show_walk_plan = {"requested_xy_m": list(desired_xy),
                                      "owner_action_task_before": actors[0].get_action_task_id()}
                    if start_point is not None and target_point is not None:
                        show_walk_plan["nav_target_xyz_m"] = [target_point.x,
                            target_point.y, target_point.z]
                        show_walk_plan["nav_snap_error_m"] = math.hypot(
                            target_point.x - desired_xy[0],
                            target_point.y - desired_xy[1])
                    if (start_point is not None and target_point is not None and
                            start_island == target_island and
                            show_walk_plan["nav_snap_error_m"] <= .25):
                        path = navmesh.query_shortest_path(
                            start_pos=start_point, end_pos=target_point)
                        if path is not None:
                            show_walk_plan["path_length_m"] = path.length()
                            show_walk_plan["path_points"] = path.get_point_count()
                            if path.get_point_count() >= 2 and path.length() <= 2.0:
                                show_walk_id = actors[0].move_to(target=(
                                    target_point.x, target_point.y, target_point.z))
                                show_walk_plan["task_id"] = show_walk_id
                                show_walk_plan["immediate_status"] = (
                                    str(actors[0].get_task_status(show_walk_id)) if
                                    show_walk_id != behavior.BEHAVIOR_TASK_ID_INVALID
                                    else "INVALID")
                    print("STAGED_SHOW_WALK", json.dumps(show_walk_plan), flush=True)
                pickup_succeeded = (pickup_task_id is not None and
                    str(actors[0].get_task_status(pickup_task_id)) ==
                    "BehaviorTaskStatus.SUCCEEDED")
                attempt_reach = (args.clear_view_response and
                    clear_view_response_plan is None and
                    social_clear_view_requested and pickup_task_id is not None and
                    ((not args.staged_walk_show and frame == 14) or
                     (args.staged_walk_show and frame >= 17 and pickup_succeeded)))
                if attempt_reach:
                    owner_position = actors[0].get_world_translation(False)
                    held = actors[0].get_hand_usage_objects()
                    clear_view_response_plan = {"requested_after_clear_view": True,
                        "owner_action_task_before": actors[0].get_action_task_id(),
                        "held_right": str(held.right_hand or ""),
                        "held_left": str(held.left_hand or ""),
                        "ik_target_xyz_m": None}
                    # Attempt a native hand gesture only after pickup has
                    # completed; record its task status separately from the
                    # successful hand-held-object state.
                    if str(held.right_hand or "") == "/World/NewShowCan":
                        center_bearing = math.radians(heading_deg)
                        centered_xy = np.asarray([
                            proxy_eye[0] + math.sin(center_bearing) * 1.7,
                            proxy_eye[1] + math.cos(center_bearing) * 1.7])
                        owner_xy = np.asarray([owner_position.x, owner_position.y])
                        to_center = centered_xy - owner_xy
                        reach_xy = owner_xy + .65 * to_center / np.linalg.norm(to_center)
                        target_xyz = (float(reach_xy[0]), float(reach_xy[1]),
                                      float(floor_z + 1.20))
                        reach_method = (getattr(actors[0], "ik_reach_hand", None) or
                                        getattr(actors[0], "reach_hand", None))
                        clear_view_response_plan["reach_method"] = (
                            getattr(reach_method, "__name__", None))
                        clear_view_response_plan["reach_signature"] = (
                            (getattr(reach_method, "__doc__", "") or "")[:700])
                        if reach_method is not None:
                            try:
                                clear_view_reach_id = reach_method(
                                    hand_usage=behavior.BehaviorHandUsage.RIGHT_HAND,
                                    target=target_xyz, duration=2.0,
                                    motion_scale=1.0)
                            except TypeError as error:
                                clear_view_response_plan["api_error"] = str(error)[:700]
                        clear_view_response_plan["ik_target_xyz_m"] = list(target_xyz)
                        clear_view_response_plan["task_id"] = clear_view_reach_id
                        clear_view_response_plan["immediate_status"] = (
                            str(actors[0].get_task_status(clear_view_reach_id)) if
                            clear_view_reach_id not in (
                                None, behavior.BEHAVIOR_TASK_ID_INVALID)
                            else "INVALID")
                    print("CLEAR_VIEW_RESPONSE_PLAN",
                          json.dumps(clear_view_response_plan), flush=True)
                show_walk_arrival = None
                if (args.staged_walk_show and show_walk_plan is not None and
                        show_walk_plan.get("nav_target_xyz_m") is not None):
                    presenter = actors[0].get_world_translation(False)
                    target_xy = show_walk_plan["nav_target_xyz_m"][:2]
                    arrival_error = math.hypot(presenter.x - target_xy[0],
                                               presenter.y - target_xy[1])
                    show_walk_near_frames = (show_walk_near_frames + 1
                        if arrival_error <= .12 else 0)
                    show_walk_arrival = {"distance_to_target_m": arrival_error,
                        "consecutive_near_frames": show_walk_near_frames,
                        "arrived_by_observation": show_walk_near_frames >= 2}
                walk_reported_succeeded = (show_walk_id not in
                    (None, behavior.BEHAVIOR_TASK_ID_INVALID) and
                    str(actors[0].get_task_status(show_walk_id)) ==
                    "BehaviorTaskStatus.SUCCEEDED")
                stage_pickup = (args.staged_walk_show and frame >= 12 and
                    pickup_task_id is None and
                    (walk_reported_succeeded or show_walk_near_frames >= 2))
                if (social_show and args.response in ("owner_show", "bystander_show") and
                        ((frame == 10 and not args.staged_walk_show) or stage_pickup)):
                    if stage_pickup:
                        show_pickup_trigger = ("task_succeeded" if
                            walk_reported_succeeded else "observed_arrival")
                    pickup_actor_index = 0 if args.response == "owner_show" else 1
                    presenter = actors[pickup_actor_index].get_world_translation(False)
                    shown_item_xyz = [float(presenter.x + .35), float(presenter.y - .32),
                                      float(floor_z + .74)]
                    if not shown_item_xform.SetTranslate(Gf.Vec3d(*shown_item_xyz)):
                        raise RuntimeError("Failed to place the shown item")
                    pickup_task_id = actors[pickup_actor_index].pickup_object(
                        target="/World/NewShowCan", snap_to_hand=True)
                    print("SHOW_PICKUP_TASK", pickup_actor_index, pickup_task_id, flush=True)
                if dynamic_low_cube_mode in ("continuous", "continuous_fast") and frame == 9:
                    continuous_cube_start_s = timeline.get_current_time()
                    print("CONTINUOUS_ROUTE_CUBE_START", json.dumps({
                        "frame": frame, "time_s": continuous_cube_start_s,
                        "speed_x_m_s": (-1. if far_right_cube_probe else 1.) *
                            (.24 if dynamic_low_cube_mode ==
                            "continuous_fast" else .15),
                        "travel_cap_m": .50 if (far_cube_probe or
                                                far_right_cube_probe) else .20}), flush=True)
                for substep in range(simulation_steps_per_frame):
                    await rep.orchestrator.step_async(rt_subframes=(8 if
                        clean_social_capture and
                        substep == simulation_steps_per_frame - 1 else 1))
                    update_continuous_cube(timeline.get_current_time())
                    sample_tof_if_due(timeline.get_current_time())
                frame_ready_wall = time.perf_counter()
                observation_time_s = timeline.get_current_time()
                lidar_started_wall = time.perf_counter()
                if lidar_sensor is not None:
                    from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data

                    raw_lidar, _ = lidar_sensor.get_data("generic-model-output")
                    if raw_lidar is None:
                        lidar_trace.append({"frame": frame, "available": False})
                    else:
                        gmo = parse_generic_model_output_data(raw_lidar)
                        count = int(gmo.numElements)
                        coords = str(gmo.elementsCoordsType)
                        if count:
                            xyz = np.column_stack((np.asarray(gmo.x),
                                                   np.asarray(gmo.y),
                                                   np.asarray(gmo.z)))
                            ranges = xyz[:, 2] if "SPHERICAL" in coords else np.linalg.norm(
                                xyz, axis=1)
                            distances = ranges[np.isfinite(ranges) & (ranges > 0)]
                        else:
                            distances = np.empty(0)
                        lidar_trace.append({"frame": frame, "available": True,
                            "num_elements": count, "coords_type": coords,
                            "frame_of_reference": str(gmo.frameOfReference),
                            "timestamp_ns": int(gmo.timestampNs),
                            "valid_ranges": int(len(distances)),
                            "range_min_m": float(np.min(distances)) if len(distances) else None,
                            "range_median_m": float(np.median(distances)) if len(distances) else None,
                            "range_under_2m": int(np.sum(distances < 2.0))})
                        if frame in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
                            lidar_snapshots[f"frame_{frame:02d}_azimuth_deg"] = np.asarray(gmo.x).copy()
                            lidar_snapshots[f"frame_{frame:02d}_elevation_deg"] = np.asarray(gmo.y).copy()
                            lidar_snapshots[f"frame_{frame:02d}_range_m"] = np.asarray(gmo.z).copy()
                        if "SPHERICAL" in coords and "SENSOR" in str(gmo.frameOfReference):
                            azimuth = np.deg2rad(np.asarray(gmo.x))
                            elevation = np.deg2rad(np.asarray(gmo.y))
                            radius = np.asarray(gmo.z)
                            local = np.column_stack((radius * np.cos(elevation) * np.cos(azimuth),
                                                     radius * np.cos(elevation) * np.sin(azimuth),
                                                     radius * np.sin(elevation)))
                            lidar_prim = stage.GetPrimAtPath("/World/ApeHeadLidarMount/Lidar")
                            transform = UsdGeom.Xformable(lidar_prim).ComputeLocalToWorldTransform(
                                Usd.TimeCode.Default())
                            lidar_origin = np.asarray(list(transform.Transform(Gf.Vec3d(0, 0, 0))))
                            basis = np.column_stack([
                                np.asarray(list(transform.TransformDir(axis)))
                                for axis in (Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0),
                                             Gf.Vec3d(0, 0, 1))])
                            world = local @ basis.T + lidar_origin
                            delta = world[:, :2] - lidar_origin[:2]
                            horizontal = np.linalg.norm(delta, axis=1)
                            low = np.isfinite(horizontal) & (horizontal > .05) & (
                                world[:, 2] >= floor_z + .03) & (
                                world[:, 2] <= floor_z + .32)
                            if route_cube_center_xy is not None:
                                on_cube = low & (np.abs(world[:, 0] - route_cube_center_xy[0]) <= .18) & (
                                    np.abs(world[:, 1] - route_cube_center_xy[1]) <= .18)
                                lidar_trace[-1]["evaluator_cube_low_returns"] = int(on_cube.sum())
                                lidar_trace[-1]["evaluator_cube_nearest_horizontal_m"] = (
                                    float(horizontal[on_cube].min()) if on_cube.any() else None)
                            lidar_trace[-1]["low_band_returns"] = int(low.sum())
                            lidar_trace[-1]["low_band_nearest_horizontal_m"] = (
                                float(horizontal[low].min()) if low.any() else None)
                        elif args.range_source == "rtx":
                            raise RuntimeError("RTX range adapter needs sensor-frame spherical points")
                        if args.range_source == "rtx":
                            bearing = (np.rad2deg(np.arctan2(delta[:, 0], delta[:, 1]))
                                       - heading_deg + 180) % 360 - 180
                            valid = np.isfinite(horizontal) & (horizontal > .3) & (horizontal < 8)
                            lidar_range_bins = np.full((len(range_heights_m),
                                                        len(range_angles_deg)), np.nan,
                                                       dtype=np.float32)
                            for plane_idx, height_m in enumerate(range_heights_m):
                                at_height = valid & (np.abs(world[:, 2] -
                                    (floor_z + height_m)) <= .10)
                                for angle_idx, angle_deg in enumerate(range_angles_deg):
                                    samples = horizontal[at_height &
                                        (np.abs(bearing - angle_deg) <= 2.5)]
                                    if samples.size:
                                        lidar_range_bins[plane_idx, angle_idx] = samples.min()
                            lidar_trace[-1]["range_adapter_valid_bins"] = int(
                                np.isfinite(lidar_range_bins).sum())
                            front = lidar_range_bins[:, np.abs(range_angles_deg) <= 20]
                            lidar_trace[-1]["range_adapter_nearest_front_m"] = (
                                float(np.nanmin(front)) if np.isfinite(front).any() else None)
                lidar_processing_wall_s = (time.perf_counter() - lidar_started_wall
                    if lidar_sensor is not None else 0.0)
                if social_show:
                    held = actors[pickup_actor_index or 0].get_hand_usage_objects()
                    pickup_state = {"frame": frame,
                        "scripted_response_evaluator_only": args.response,
                        "item_spawn_xyz_m_evaluator_only": shown_item_xyz if
                            pickup_actor_index is not None and frame >= 10 else None,
                        "pickup_actor_index_evaluator_only": pickup_actor_index,
                        "show_walk_status_evaluator_only": str(
                            actors[0].get_task_status(show_walk_id)) if
                            show_walk_id not in (None, behavior.BEHAVIOR_TASK_ID_INVALID)
                            else None,
                        "show_walk_plan_evaluator_only": show_walk_plan,
                        "show_walk_arrival_evaluator_only": show_walk_arrival,
                        "show_pickup_trigger_evaluator_only": show_pickup_trigger,
                        "pickup_task_status_evaluator_only": str(
                            actors[pickup_actor_index].get_task_status(pickup_task_id)) if
                            pickup_task_id is not None else None,
                        "clear_view_reach_status_evaluator_only": str(
                            actors[0].get_task_status(clear_view_reach_id)) if
                            clear_view_reach_id not in (None, behavior.BEHAVIOR_TASK_ID_INVALID)
                            else None,
                        "clear_view_response_plan_evaluator_only": (
                            clear_view_response_plan if frame >= 14 else None),
                        # The authored USD transform does not follow the
                        # runtime hand attachment; score rendered pixels, not
                        # this static spawn transform, for item visibility.
                        "item_authored_usd_xyz_m_evaluator_only": list(
                            UsdGeom.XformCache().GetLocalToWorldTransform(
                                stage.GetPrimAtPath("/World/NewShowCan"))
                            .ExtractTranslation()) if frame >= 10 else None,
                        "left_hand_object_evaluator_only": str(held.left_hand or ""),
                        "right_hand_object_evaluator_only": str(held.right_hand or "")}
                    social_scene_trace.append(pickup_state)
                    if pickup_task_id is not None:
                        print("SHOW_PICKUP_STATE", json.dumps(pickup_state), flush=True)
                fresh_cache = UsdGeom.XformCache()
                motion_trace.append({"frame": frame,
                    "timeline_time_s": timeline.get_current_time(),
                    "move_task_status": str(actors[1].get_task_status(move_id)),
                    "active_action_task_id": actors[1].get_action_task_id(),
                    "characters": [{"prim_path": row["prim_path"],
                        "xyz_m": [float(getattr(actors[index].get_world_translation(False), axis))
                                  for axis in ("x", "y", "z")],
                        "usd_xyz_m": list(fresh_cache.GetLocalToWorldTransform(
                            stage.GetPrimAtPath(row["prim_path"])).ExtractTranslation())}
                        for index, row in enumerate(characters)]})
                range_started_wall = time.perf_counter()
                if range_query is not None:
                    planes = []
                    for height_m in range_heights_m:
                        origin = carb.Float3(proxy_eye[0], proxy_eye[1],
                                             floor_z + height_m)
                        beams = []
                        for angle_deg in range_angles_deg:
                            angle = math.radians(angle_deg + (
                                body_heading_deg if view_route is not None else heading_deg))
                            direction = carb.Float3(math.sin(angle),
                                                    math.cos(angle), 0.0)
                            hit = range_query.raycast_closest(origin, direction, 8.0)
                            beams.append({"angle_deg": angle_deg,
                                "distance_m": float(hit["distance"]) if hit["hit"] else None,
                                "hit_prim_evaluator_only": str(hit.get("rigidBody") or
                                                               hit.get("collision") or "")
                                if hit["hit"] else None})
                        planes.append({"height_above_floor_m": height_m,
                            "origin_xyz_m": [origin.x, origin.y, origin.z],
                            "beams": beams})
                    range_trace.append({"frame": frame,
                        "timeline_time_s": timeline.get_current_time(),
                        "body_heading_deg_commanded": body_heading_deg if view_route is not None
                            else heading_deg,
                        "camera_heading_deg_commanded": heading_deg,
                        "planes": planes})
                range_processing_wall_s = (time.perf_counter() - range_started_wall
                    if range_query is not None else 0.0)
                frame_samples = {}
                capture_wall = frame_ready_wall
                control_read_wall = None
                camera_timing = {}
                for camera_dir, _, rgb, depth, _ in sorted(
                        cameras, key=lambda entry: entry[0].name != control_camera):
                    if camera_dir.name == control_camera:
                        control_read_wall = time.perf_counter()
                    read_started = time.perf_counter()
                    rgb_data = np.asarray(rgb.get_data())
                    rgb_read_wall = time.perf_counter() - read_started
                    depth_started = time.perf_counter()
                    depth_data = np.asarray(depth.get_data())
                    depth_read_wall = time.perf_counter() - depth_started
                    if rgb_data.shape[:2] != (480, 640) or depth_data.shape[:2] != (480, 640):
                        raise RuntimeError(f"Invalid frame {frame}: {rgb_data.shape}, {depth_data.shape}")
                    camera_timing[camera_dir.name] = {"rgb_read_s": rgb_read_wall,
                        "depth_read_s": depth_read_wall,
                        "save_s": None}
                    frame_samples[camera_dir.name] = (rgb_data[:, :, :3].astype(np.uint8),
                                                       depth_data.astype(np.float32))
                if high_level_actor is not None:
                    control_rgb, control_depth = frame_samples[control_camera]
                    rgbd_low_hits = None
                    rgbd_low_evidence = None
                    if rgbd_low_hit_fusion_enabled and low_blind_active(frame):
                        from rgbd_low_surface import low_surface_fan
                        rgbd_low_evidence = low_surface_fan(
                            control_depth[::3, ::3],
                            np.asarray(range_angles_deg, dtype=np.float32),
                            horizontal_fov_deg=args.control_fov,
                            camera_yaw_deg=heading_deg,
                            body_yaw_deg=body_heading_deg)
                        rgbd_low_hits = rgbd_low_evidence["hit_ranges_m"]
                    localization = (visual_odometry.update(
                        control_rgb, control_depth[::3, ::3], previous_command)
                        if visual_odometry is not None else None)
                    if control_read_wall is None:
                        raise RuntimeError("Control camera was not captured")
                    pre_actor_wall = time.perf_counter()
                    sensor_pipeline_wall = {"frame_ready_to_actor_s":
                        pre_actor_wall - frame_ready_wall,
                        "lidar_processing_s": lidar_processing_wall_s,
                        "range_fan_s": range_processing_wall_s,
                        "frame_ready_to_control_read_s": control_read_wall - frame_ready_wall,
                        "control_read_start_to_actor_s": pre_actor_wall - control_read_wall,
                        "camera_phases_s": camera_timing}
                    social_decision = None
                    social_started_wall = time.perf_counter()
                    if show_actor is not None:
                        show_rgb, show_depth = frame_samples[
                            f"ape_social_{args.social_fov}deg"]
                        if frame == 8:
                            social_decision = show_actor.request(
                                show_rgb, time_s=timeline.get_current_time())
                        elif frame == 9:
                            social_decision = {"intent": "wait_for_show",
                                "reason": "allow_response_window",
                                "time_s": timeline.get_current_time()}
                        elif frame >= 10 and inference_pool is None:
                            social_decision = show_actor.observe(
                                show_rgb, show_depth, time_s=timeline.get_current_time(),
                                frame_age_s=time.perf_counter() - capture_wall,
                                fov_deg=args.social_fov)
                    social_decision_wall_s = time.perf_counter() - social_started_wall
                    # A PhysX ray that ran to its calibrated limit is measured
                    # clear. An RTX bin with no valid return is unknown. Keep
                    # those states distinct for the route's free-space map.
                    distances = np.array([[beam["distance_m"] if beam["distance_m"] is not None
                                           else np.inf for beam in plane["beams"]]
                                          for plane in range_trace[-1]["planes"]], dtype=np.float32)
                    if args.range_source == "rtx":
                        if lidar_range_bins is None:
                            raise RuntimeError("RTX range source has no parsed scan")
                        distances = lidar_range_bins
                    actor_distances = distances.copy()
                    route_distances = distances.copy()
                    if low_blind_active(frame):
                        actor_distances[0, low_blind_mask] = np.nan
                        route_distances[0, low_blind_mask] = np.nan
                    if view_route is not None and args.route_range_mode != "both":
                        hidden_plane = 1 if args.route_range_mode == "low" else 0
                        actor_distances[hidden_plane] = np.nan
                        route_distances[hidden_plane] = np.nan
                    actor_range_angles_deg = np.asarray(
                        range_angles_deg, dtype=np.float32) + (
                        body_heading_deg - heading_deg if view_route is not None else 0.0)
                    owner_started_wall = time.perf_counter()
                    if force_one_person_miss and frame == 29:
                        # Inject one empty *person proposal* into the actor
                        # while keeping the rendered owner and depth intact.
                        # Full detection is forced so the fault cannot be
                        # bypassed by the short RGB-D flow path.
                        high_level_actor.fast_flow_ready = False
                        high_level_actor.force_empty_next_proposal = True
                    if force_primary_person_miss and frame == 5:
                        high_level_actor.fast_flow_ready = False
                        high_level_actor.force_primary_empty_next_proposal = True
                    if inference_pool is None:
                        decision = high_level_actor.observe(
                            control_rgb, control_depth[::3, ::3], actor_distances,
                            actor_range_angles_deg,
                            time_s=observation_time_s, fov_deg=args.control_fov,
                            ego_yaw_delta_deg=applied_yaw_delta_deg)
                        owner_decision_wall_s = time.perf_counter() - owner_started_wall
                        monitor_alarm = False
                        monitor_samples = 0
                    else:
                        deferred_social = ((show_rgb, show_depth)
                            if show_actor is not None and frame >= 10 else None)
                        (decision, worker_social, owner_decision_wall_s,
                         bundle_wall_s, monitor_alarm, monitor_trace) = (
                            await decide_while_world_moves(
                                control_rgb, control_depth[::3, ::3], actor_distances,
                                actor_range_angles_deg,
                                observation_time_s, range_trace[-1],
                                deferred_social, capture_wall))
                        if worker_social is not None:
                            social_decision = worker_social
                            social_decision_wall_s = worker_social["social_inference_wall_s"]
                        sensor_pipeline_wall["combined_vision_inference_s"] = bundle_wall_s
                        monitor_trace["frame"] = frame
                        vision_monitor_trace.append(monitor_trace)
                        monitor_samples = len(monitor_trace["samples"])
                    command = {"yaw_delta_deg": 0.0, "forward_step_m": 0.0}
                    heading_before_gaze_command_deg = heading_deg
                    if high_level_actor.last_proposal_timing is not None:
                        decision["proposal_timing_wall_s"] = (
                            high_level_actor.last_proposal_timing.copy())
                    source_age_s = timeline.get_current_time() - observation_time_s
                    safety_move_crossed_source = (tof_supervisor is not None and
                        tof_supervisor.last_step_time_s is not None and
                        tof_supervisor.last_step_time_s > observation_time_s + 1e-6)
                    if (safety_move_crossed_source and show_actor is not None and
                            social_decision is not None and
                            social_decision["intent"] == "check_fresh_show"):
                        # A slow proposal can finish after the near-body
                        # response moved the camera. Reacquire it from the
                        # post-response view before starting verification.
                        show_actor.template = None
                        show_actor.last_box = None
                        show_actor.streak = 0
                        show_actor.previous_point = None
                        social_verify_anchor_frame = None
                        social_decision = {**social_decision,
                            "intent": "wait_for_show",
                            "reason": "tof_escape_invalidated_item_candidate"}
                    if (social_decision is not None and
                            social_decision["intent"] == "new_item_seen_unattributed" and
                            (source_age_s > .20 or monitor_alarm or
                             safety_move_crossed_source)):
                        # The social worker may finish before the world has
                        # advanced through its measured latency. A final item
                        # claim must pass the same completed-source check.
                        show_actor.outcome = "new_show_unconfirmed_unknown"
                        show_actor.streak = 0
                        show_actor.previous_point = None
                        social_decision = {**social_decision,
                            "intent": "check_fresh_show",
                            "reason": ("tof_escape_invalidated_show_image" if
                                safety_move_crossed_source else
                                "stale_or_motion_during_object_inference"),
                            "outcome": show_actor.outcome,
                            "fresh": False,
                            "completed_source_age_s": source_age_s}
                    if (social_decision is not None and
                            social_decision.get("reason") ==
                            "candidate_needs_two_fresh_views"):
                        social_verify_anchor_frame = frame
                    if (social_decision is not None and
                            social_decision["intent"] == "new_item_seen_unattributed"):
                        social_verify_anchor_frame = None
                    if (social_show and frame >= 10 and social_decision is not None and
                            social_decision["intent"] == "wait_for_show"):
                        social_failed_item_views += 1
                    # A bounded gaze update only changes what the next image
                    # sees. Translation needs the stricter fresh range/image
                    # pair; neither action may ignore a monitor alarm.
                    freshness_limit_s = (.30 if decision["intent"] == "look_to_owner"
                                         else .20)
                    stale_or_alarm = (held_pose_probe or (inference_pool is not None and
                        (source_age_s > freshness_limit_s or monitor_alarm)))
                    if safety_move_crossed_source:
                        # The camera moved while this image was being
                        # interpreted. Preserve the source result as evidence
                        # but make its command intent unusable and force a
                        # fresh full person check at the new pose.
                        decision["pre_interrupt_intent"] = decision["intent"]
                        decision["intent"] = "hold_reobserve"
                        decision["reason"] = "tof_escape_invalidated_image"
                        decision["safety_response"] = tof_supervisor.summary()
                        high_level_actor.fast_flow_ready = False
                    localization_unconfirmed = (localization is not None and
                        previous_command is not None and
                        abs(previous_command["forward_step_m"]) > 1e-8 and
                        localization["status"] != "visual_registered")
                    if stale_or_alarm:
                        decision["command_interlock"] = (
                            "tof_escape_reobserve" if tof_supervisor is not None and
                            tof_supervisor.alarm is not None else
                            "held_low_proximity" if monitor_alarm and
                            inference_pool is not None and
                            monitor_trace.get("held_raw_alarm") else
                            "person_sector_motion" if monitor_alarm else
                            "fixed_pose_monitor_probe" if held_pose_probe else
                            "stale_vision")
                    decision["command_source_age_limit_s"] = freshness_limit_s
                    if (social_show and high_level_actor.identity_lost and
                            social_decision is not None and
                            social_decision["intent"] == "wait_for_show" and
                            (social_search_looks == 2 or
                             social_failed_item_views >= 4) and
                            not social_clear_view_requested):
                        social_decision = {**social_decision,
                            "intent": "request_clear_view",
                            "reason": "item_not_visible_after_bounded_observation",
                            "speech_intent": "I can't see it yet. Could you hold it out in front of me?",
                            "identity_state": "new_teacher_unattributed",
                            "failed_item_views": social_failed_item_views,
                            "completed_gaze_looks": social_search_looks}
                        social_clear_view_requested = True
                    route_plan = None
                    low_motion_evidence = None
                    path_motion_evidence = None
                    raw_swept_evidence = None
                    owner_world_heading_deg = None
                    if view_route is not None:
                        bearing = decision.get("owner_bearing_deg")
                        person_depth = decision.get("local_depth_m")
                        if (bearing is not None and person_depth is not None and
                                localization is not None and
                                decision["intent"] in ("look_to_owner",
                                    "approach_bounded", "hold_near_person")):
                            pose_xy = np.asarray(localization["pose_xy_m"])
                            direction = math.radians(
                                localization["heading_deg"] + bearing)
                            horizontal_depth = person_depth * math.cos(
                                math.atan(.25 / 1.5))
                            target_xy = pose_xy + horizontal_depth * np.array([
                                math.sin(direction), math.cos(direction)])
                            owner_world_heading_deg = math.degrees(math.atan2(
                                target_xy[0] - pose_xy[0],
                                target_xy[1] - pose_xy[1]))
                            route_plan = view_route.plan(route_distances,
                                np.asarray(range_angles_deg), pose_xy,
                                body_heading_deg, target_xy,
                                observation_time_s,
                                hit_only_ranges_m=rgbd_low_hits)
                        else:
                            route_plan = {"intent": "hold_reobserve",
                                "reason": "person_or_pose_not_confirmed"}
                        if low_motion_guard is not None and localization is not None:
                            low_motion_evidence = low_motion_guard.observe(
                                actor_distances[0], np.asarray(range_angles_deg),
                                np.asarray(localization["pose_xy_m"]),
                                body_heading_deg, observation_time_s)
                        if path_timed_guard is not None and localization is not None:
                            pose_xy = np.asarray(localization["pose_xy_m"])
                            visually_registered = (
                                localization["status"] == "visual_registered")
                            stationary_pose_fallback = (
                                localization["status"] == "odometry_fallback" and
                                last_guard_visual_time_s is not None and
                                observation_time_s - last_guard_visual_time_s <= .45 and
                                previous_command is not None and
                                abs(previous_command["forward_step_m"]) < 1e-8)
                            if visually_registered or stationary_pose_fallback:
                                # A brief VO dropout after a zero-translation
                                # command still gives a bounded simulation
                                # pose for observing low-surface motion. The
                                # fallback cannot authorize translation; a
                                # physical version needs IMU/encoder/contact
                                # corroboration of this stationary assumption.
                                path_timed_guard.observe(actor_distances[0],
                                    np.asarray(range_angles_deg), pose_xy,
                                    body_heading_deg, observation_time_s)
                            if visually_registered:
                                last_guard_visual_time_s = observation_time_s
                                if source_age_s > .30:
                                    path_motion_evidence = {"action": "hold_reobserve",
                                        "reason": "dynamic_projection_source_stale",
                                        "source_age_s": source_age_s,
                                        "path_threat": False, "hold_threat": False}
                                else:
                                    step = (route_plan.get("forward_step_m", 0.) if
                                            route_plan["intent"] == "step_to_waypoint" else 0.)
                                    path_motion_evidence = path_timed_guard.assess(
                                        pose_xy, body_heading_deg, step, source_age_s)
                            else:
                                path_motion_evidence = {"action": "hold_reobserve",
                                    "reason": "dynamic_projection_pose_unconfirmed",
                                    "localization_status": localization["status"],
                                    "path_threat": False, "hold_threat": False}
                        if (raw_swept_gate_enabled and route_plan["intent"] ==
                                "step_to_waypoint"):
                            from raw_swept_gate import assess_swept_low_hits

                            # This final PhysX fan is taken at decision time,
                            # independent of the earlier image/routing packet.
                            # It vetoes an occupied next-step corridor only;
                            # the map still must positively authorize free space.
                            gate_origin = carb.Float3(proxy_eye[0], proxy_eye[1],
                                                      floor_z + range_heights_m[0])
                            gate_low = []
                            for angle_deg in range_angles_deg:
                                direction = math.radians(body_heading_deg + angle_deg)
                                hit = range_query.raycast_closest(
                                    gate_origin, carb.Float3(math.sin(direction),
                                                             math.cos(direction), 0.), 8.)
                                gate_low.append(float(hit["distance"]) if hit["hit"]
                                                else math.inf)
                            if low_blind_active(frame):
                                gate_low = [None if low_blind_mask[index] else value
                                            for index, value in enumerate(gate_low)]
                            raw_swept_evidence = assess_swept_low_hits(
                                gate_low, range_angles_deg,
                                step_m=route_plan["forward_step_m"],
                                step_heading_rel_deg=0., clearance_m=.24)
                            raw_swept_evidence.update({
                                "scan_time_s": timeline.get_current_time(),
                                "range_source": "fresh_physx_low_15cm",
                                "low_ranges_m": [value if value is not None and
                                                 math.isfinite(value) else None
                                                 for value in gate_low],
                                "measured_clear_beam_indices": [index for index, value
                                    in enumerate(gate_low) if value == math.inf],
                                "unknown_beam_indices": [index for index, value
                                    in enumerate(gate_low) if value is None],
                                "proposed_route_intent": route_plan["intent"]})
                            if raw_swept_evidence["veto"] or (
                                    raw_swept_evidence["status"] == "unknown"):
                                route_plan = {**route_plan,
                                    "intent": "hold_reobserve",
                                    "reason": ("raw_swept_low_hit" if
                                               raw_swept_evidence["veto"] else
                                               "raw_swept_low_unknown")}
                    social_search_wall_age_s = time.perf_counter() - capture_wall
                    search_bearing = high_level_actor.last_bearing_deg
                    safe_social_search = (social_show and frame >= 10 and
                        high_level_actor.identity_lost and social_decision is not None and
                        social_decision["intent"] == "wait_for_show" and
                        search_bearing is not None and social_search_looks < 2 and
                        not monitor_alarm and source_age_s <= .30 and
                        social_search_wall_age_s <= .30)
                    if view_route is not None:
                        if (path_motion_evidence is not None and
                                path_motion_evidence["action"] != "allow" and
                                (route_plan["intent"] == "step_to_waypoint" or
                                 path_motion_evidence["hold_threat"])):
                            pose_xy = np.asarray(localization["pose_xy_m"])
                            heading = math.radians(body_heading_deg)
                            reverse_xy = pose_xy - .08 * np.array([
                                math.sin(heading), math.cos(heading)])
                            reverse_prediction = (path_timed_guard.assess(
                                pose_xy, body_heading_deg, -.08, source_age_s)
                                if path_motion_evidence["action"] == "retreat" else None)
                            reverse_predicted_clear = (reverse_prediction is not None and
                                (reverse_prediction["predicted_path_min_m"] is None or
                                 reverse_prediction["predicted_path_min_m"] >=
                                 view_route.plan_clearance_m))
                            reverse_known_free = (reverse_predicted_clear and
                                view_route._clear_segment(
                                    pose_xy, reverse_xy, observation_time_s,
                                    view_route._occupied(observation_time_s)))
                            route_plan = {**route_plan,
                                "intent": ("retreat_from_path_timed_obstacle"
                                           if reverse_known_free else "hold_reobserve"),
                                "reason": ("predicted_held_pose_low_clearance"
                                           if path_motion_evidence["hold_threat"] else
                                           "predicted_next_step_low_clearance"),
                                "reverse_known_free": reverse_known_free,
                                "reverse_predicted_clear": reverse_predicted_clear}
                            if (reverse_known_free and not stale_or_alarm and
                                    not localization_unconfirmed):
                                command["forward_step_m"] = -.08
                                command["translation_heading_deg"] = body_heading_deg
                                proxy_eye[0] += math.sin(heading) * command["forward_step_m"]
                                proxy_eye[1] += math.cos(heading) * command["forward_step_m"]
                        elif (low_motion_evidence is not None and
                                low_motion_evidence["threatened"]):
                            pose_xy = np.asarray(localization["pose_xy_m"])
                            heading = math.radians(body_heading_deg)
                            reverse_xy = pose_xy - .08 * np.array([
                                math.sin(heading), math.cos(heading)])
                            reverse_known_free = view_route._clear_segment(
                                pose_xy, reverse_xy, observation_time_s,
                                view_route._occupied(observation_time_s))
                            route_plan = {**route_plan,
                                "intent": ("retreat_from_closing_low_obstacle"
                                           if reverse_known_free else "hold_reobserve"),
                                "reason": ("predicted_low_clearance"
                                           if reverse_known_free else
                                           "predicted_low_clearance_no_known_reverse"),
                                "reverse_known_free": reverse_known_free}
                            if (reverse_known_free and not stale_or_alarm and
                                    not localization_unconfirmed):
                                command["forward_step_m"] = -.08
                                command["translation_heading_deg"] = body_heading_deg
                                proxy_eye[0] += math.sin(heading) * command["forward_step_m"]
                                proxy_eye[1] += math.cos(heading) * command["forward_step_m"]
                        elif (route_plan["intent"] == "turn_to_waypoint" and
                                not stale_or_alarm and not localization_unconfirmed):
                            command["body_yaw_delta_deg"] = route_plan["yaw_delta_deg"]
                            body_heading_deg += command["body_yaw_delta_deg"]
                        elif (route_plan["intent"] == "step_to_waypoint" and
                                not stale_or_alarm and not localization_unconfirmed):
                            command["forward_step_m"] = route_plan["forward_step_m"]
                            command["translation_heading_deg"] = body_heading_deg
                            heading = math.radians(body_heading_deg)
                            proxy_eye[0] += math.sin(heading) * command["forward_step_m"]
                            proxy_eye[1] += math.cos(heading) * command["forward_step_m"]
                        elif (decision.get("reason") == "owner_appearance_unconfirmed" and
                              decision.get("requested_gaze_delta_deg") is not None and
                              not held_pose_probe and
                              not monitor_alarm and source_age_s <= 2.5 and
                              high_level_actor.last_time_s is not None and
                              timeline.get_current_time() -
                              high_level_actor.last_time_s <= 3.0):
                            # The latest person crop was not accepted. A
                            # bounded head-only look toward the last confirmed
                            # bearing seeks a new image; it cannot authorize
                            # body motion or refresh owner identity itself.
                            command["yaw_delta_deg"] = decision[
                                "requested_gaze_delta_deg"]
                            heading_deg += command["yaw_delta_deg"]
                            decision["memory_gaze_only"] = True
                            decision["memory_gaze_source_age_s"] = source_age_s
                        elif (held_head_look_probe and
                              decision["intent"] == "look_to_owner" and
                              source_age_s <= 2.5 and not monitor_alarm and
                              high_level_actor.last_time_s is not None and
                              timeline.get_current_time() -
                              high_level_actor.last_time_s <= 3.0 and
                              tof_supervisor.state in ("watching", "escape_complete")):
                            # The base stays fixed for the near-body probe.
                            # A bounded look may use the last confirmed
                            # bearing even after slow inference. It cannot
                            # authorize translation, and safety movement owns
                            # attention while a response is pending.
                            command["yaw_delta_deg"] = float(np.clip(
                                decision["yaw_delta_deg"], -10., 10.))
                            heading_deg += command["yaw_delta_deg"]
                            decision["held_head_look_only"] = True
                        # The social camera can pan on a simulated head while
                        # the route/range frame turns with the body. Keep the
                        # owner in view without feeding truth geometry to gaze.
                        if (owner_world_heading_deg is not None and
                                route_plan["intent"] != "hold_reobserve" and
                                not stale_or_alarm):
                            relative = (owner_world_heading_deg - body_heading_deg +
                                        180) % 360 - 180
                            desired_camera_heading = body_heading_deg + float(
                                np.clip(relative, -35, 35))
                            camera_error = (desired_camera_heading - heading_deg +
                                            180) % 360 - 180
                            command["yaw_delta_deg"] = float(np.clip(
                                camera_error, -10, 10))
                            heading_deg += command["yaw_delta_deg"]
                    elif safe_social_search:
                        # Search only the last observed side, with no identity
                        # rebinding or translation. A new shown item remains an
                        # unattributed session even if the scan finds it.
                        command["yaw_delta_deg"] = math.copysign(10.0, search_bearing)
                        heading_deg += command["yaw_delta_deg"]
                        social_search_looks += 1
                        social_decision["bounded_gaze_search"] = {
                            "look_number": social_search_looks,
                            "prior_owner_bearing_deg": search_bearing,
                            "frame_ready_to_command_wall_s": social_search_wall_age_s}
                    elif (decision["intent"] == "look_to_owner" and
                            not (social_show and frame >= 8) and not stale_or_alarm):
                        command["yaw_delta_deg"] = decision["yaw_delta_deg"]
                        heading_deg += command["yaw_delta_deg"]
                    elif (decision["intent"] == "approach_bounded" and
                            not (social_show and frame >= 8) and not stale_or_alarm and
                            not localization_unconfirmed):
                        command["forward_step_m"] = decision["step_m"]
                        heading = math.radians(heading_deg)
                        proxy_eye[0] += math.sin(heading) * command["forward_step_m"]
                        proxy_eye[1] += math.cos(heading) * command["forward_step_m"]
                    verifying_item = (social_verify_anchor_frame is not None and
                        frame - social_verify_anchor_frame <= 3 and
                        social_decision is not None and
                        social_decision["intent"] == "check_fresh_show")
                    if (verify_gaze_conflict_probe and verifying_item and
                            tof_supervisor is not None and
                            tof_supervisor.state == "escape_complete" and
                            command.get("body_yaw_delta_deg", 0.) == 0. and
                            command["forward_step_m"] == 0. and
                            not monitor_alarm):
                        # Diagnostic only: request a head turn at the exact
                        # image-driven item-verification boundary. It never
                        # adds a body or safety command, nor item evidence.
                        if heading_deg == heading_before_gaze_command_deg:
                            command["yaw_delta_deg"] = 10.0
                            heading_deg += 10.0
                        decision["diagnostic_gaze_request_deg"] = (
                            heading_deg - heading_before_gaze_command_deg)
                    requested_gaze_delta_deg = (
                        heading_deg - heading_before_gaze_command_deg)
                    if verifying_item and requested_gaze_delta_deg != 0.:
                        if not verify_gaze_conflict_bypass:
                            # One gaze owner: keep the candidate framed for
                            # two fresh views; the independent ToF response
                            # above still owns any necessary safety motion.
                            decision["gaze_held_for_item_verification"] = {
                                "requested_yaw_deg": requested_gaze_delta_deg,
                                "anchor_frame": social_verify_anchor_frame}
                            heading_deg = heading_before_gaze_command_deg
                            command["yaw_delta_deg"] = 0.0
                        else:
                            decision["diagnostic_gaze_bypass_executed"] = True
                    applied_yaw_delta_deg = command["yaw_delta_deg"]
                    if localization_unconfirmed and decision["intent"] == "approach_bounded":
                        decision["command_interlock"] = "pose_motion_unconfirmed"
                    previous_command = command.copy()
                    heading = math.radians(heading_deg)
                    next_eye = Gf.Vec3d(*proxy_eye)
                    next_target = Gf.Vec3d(proxy_eye[0] + 1.5 * math.sin(heading),
                                           proxy_eye[1] + 1.5 * math.cos(heading),
                                           camera_follow_target_z_m)
                    next_view = Gf.Matrix4d(1.0)
                    next_view.SetLookAt(next_eye, next_target, Gf.Vec3d(0, 0, 1))
                    for _, _, _, _, transform_op in cameras:
                        transform_op.Set(next_view.GetInverse())
                    if lidar_mount_pose is not None:
                        lidar_mount_pose.SetTranslate(next_eye)
                    sensor_pipeline_wall.update({
                        "social_decision_s": social_decision_wall_s,
                        "owner_decision_s": owner_decision_wall_s,
                        "decision_source_age_sim_s": source_age_s,
                        "monitor_alarm": monitor_alarm,
                        "tof_escape_alarm": (tof_supervisor.alarm is not None
                            if tof_supervisor is not None else False),
                        "monitor_samples": monitor_samples,
                        "frame_ready_to_command_s": time.perf_counter() - frame_ready_wall,
                        "control_read_start_to_command_s": time.perf_counter() - control_read_wall})
                    action_trace.append({"frame": frame,
                        "control_fov_deg": args.control_fov,
                        "social_fov_deg": args.social_fov if social_show else None,
                        "range_source": args.range_source,
                        "observation_time_s": observation_time_s,
                        "decision_time_s": timeline.get_current_time(),
                        "actor": decision, "social_actor": social_decision,
                        "forced_person_proposal_miss_evaluator_only": (
                            force_one_person_miss and frame == 29),
                        "forced_primary_detector_miss_evaluator_only": (
                            force_primary_person_miss and frame == 5),
                        "localization": localization,
                        "route_plan": route_plan,
                        "rgbd_low_hit_fusion": ({
                            "candidate_pixels": rgbd_low_evidence["candidate_pixels"],
                            "hit_bins": rgbd_low_evidence["hit_bins"],
                            "min_horizontal_hit_m": rgbd_low_evidence[
                                "min_horizontal_hit_m"]}
                            if rgbd_low_evidence is not None else None),
                        "low_motion_evidence": low_motion_evidence,
                        "path_motion_evidence": path_motion_evidence,
                        "raw_swept_evidence": raw_swept_evidence,
                        "tof_response": (tof_supervisor.summary()
                            if tof_supervisor is not None else None),
                        "route_range_mode": args.route_range_mode if view_route is not None
                            else None,
                        "route_proxy_radius_m": (route_proxy_radius_m
                            if view_route is not None else None),
                        "route_planning_clearance_m": (view_route.plan_clearance_m
                            if view_route is not None else None),
                        "route_required_free_planes": (view_route.required_free_planes
                            if view_route is not None else None),
                        "low_range_blind_sector_deg": (
                            [-90, 90] if low_blind_mode == "all_from_frame9" and frame >= 9
                            else [-70, 0] if low_blind_mode == "left_70deg" else None),
                        "route_body_heading_deg_commanded": body_heading_deg
                            if view_route is not None else None,
                        "route_head_yaw_offset_deg_commanded": heading_deg - body_heading_deg
                            if view_route is not None else None,
                        "sensor_pipeline_wall_s": sensor_pipeline_wall,
                        "command_for_next_frame": command,
                        "proxy_eye_xyz_m_evaluator_only": proxy_eye.copy(),
                        "route_cube_center_xy_m_evaluator_only": (
                            route_cube_center_xy.copy() if route_cube_center_xy is not None
                            else None),
                        "proxy_heading_deg_evaluator_only": heading_deg})
                for camera_dir, _, _, _, _ in cameras:
                    rgb_data, depth_data = frame_samples[camera_dir.name]
                    save_started = time.perf_counter()
                    Image.fromarray(rgb_data).save(
                        camera_dir / f"rgb/rgb_{frame:05d}.png")
                    np.save(camera_dir / f"distance_to_camera/distance_to_camera_{frame:05d}.npy",
                            depth_data)
                    camera_timing[camera_dir.name]["save_s"] = time.perf_counter() - save_started
                print("CAPTURE_FRAME", frame, flush=True)
            (output.parent / "character-motion-trace.json").write_text(
                json.dumps(motion_trace, indent=2))
            if range_trace:
                (output.parent / "range-fan-trace.json").write_text(
                    json.dumps({"kind": "PhysX collision ray fan; no optical material or sensor noise",
                        "max_range_m": 8.0, "frames": range_trace}, indent=2))
            if action_trace:
                (output.parent / "online-action-trace.json").write_text(
                    json.dumps({"scope": "sensor-driven look/approach with kinematic camera proxy; "
                        "optional short RGB-D flow between periodic full detections",
                        "actions": action_trace}, indent=2))
            if inference_pool is not None:
                inference_pool.shutdown(wait=True)
                (output.parent / "continuous-vision-monitor.json").write_text(
                    json.dumps({"scope": "ideal PhysX fan sampled while vision runs; "
                        "world advances through measured inference latency, with no physical stop model",
                        "monitor_heights_above_floor_m": [range_heights_m[plane]
                            for plane in ((0,) if args.route_range_mode == "low" else
                                          (1,) if args.route_range_mode == "mid" else (0, 1))],
                        "frames": vision_monitor_trace}, indent=2))
            if social_show:
                (output.parent / "social-scene-trace.json").write_text(
                    json.dumps({"scope": "scripted spawn and native hand occupancy for evaluation only; "
                        "the authored spawn pose is not the rendered pose after pickup",
                        "frames": social_scene_trace}, indent=2))
                (output.parent / "social-actor-outcome.json").write_text(
                    json.dumps({"outcome": show_actor.outcome,
                        "identity_state": "new_teacher_unattributed"}, indent=2))
            if lidar_sensor is not None:
                np.savez_compressed(output.parent / "rtx-lidar-three-scans.npz",
                                    **lidar_snapshots)
                (output.parent / "rtx-lidar-trace.json").write_text(
                    json.dumps({"model": "OS1_REV6_32ch20hz512res",
                        "mount": "45 cm kinematic head proxy",
                        "actor_consumes_lidar": args.range_source == "rtx",
                        "frames": lidar_trace}, indent=2))
                if not any(row.get("valid_ranges", 0) > 0 for row in lidar_trace):
                    raise RuntimeError("RTX lidar produced no valid returns")
            if low_tof_probe_enabled:
                (output.parent / "low-tof-trace.json").write_text(
                    json.dumps({"model": "VL53L5CX-like 8x8 central PhysX ray screen",
                        "reference": "https://www.st.com/resource/en/datasheet/vl53l5cx.pdf",
                        "horizontal_vertical_fov_deg": [45, 45],
                        "range_m": [.02, 4.0],
                        "nominal_rate_hz": 15,
                        "mounts_body_relative": MOUNTS,
                        "actor_consumes_tof": tof_escape_enabled,
                        "limitations": "Single center ray per zone; no optical response, reflectance, crosstalk, noise, dropout, packaging or body occlusion",
                        "samples": tof_trace}, indent=2))
            if tof_supervisor is not None:
                (output.parent / "tof-response-trace.json").write_text(
                    json.dumps({"scope": "actor-visible front-left low ToF cue, "
                        "separate ideal 360-degree dual-height swept corridor, "
                        "kinematic 25 cm/s eye escape after 0.2 s response delay; "
                        "no articulated body, IMU, joint/contact or optical faults",
                        "summary": tof_supervisor.summary(),
                        "samples": tof_response_trace}, indent=2))
            timeline.stop()
            for _, product, rgb, depth, _ in cameras:
                rgb.detach()
                depth.detach()
                product.destroy()

        failure = []

        async def report_failure():
            try:
                await run()
            except Exception as error:
                import traceback
                print("ISAAC_RUN_FAILURE", traceback.format_exc(), flush=True)
                failure.append(error)

        result = simulation_app.run_coroutine(report_failure())
        if hasattr(result, "result"):
            result.result()
    finally:
        simulation_app.close()
    if failure:
        raise failure[0]


if __name__ == "__main__":
    main()
