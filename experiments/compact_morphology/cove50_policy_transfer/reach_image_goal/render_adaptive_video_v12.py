"""Render saved adaptive RGB reach with selected policy and contact evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import (
    VirtualRgbGoal, orange_bbox,
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fitted_label(draw: ImageDraw.ImageDraw, label: str, face: ImageFont.ImageFont,
                 max_width: int) -> str:
    if draw.textlength(label, font=face) <= max_width:
        return label
    low, high = 0, len(label)
    while low < high:
        middle = (low + high + 1) // 2
        if draw.textlength(label[:middle] + "…", font=face) <= max_width:
            low = middle
        else:
            high = middle - 1
    return label[:low].rstrip() + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=135)
    variants = tuple(f"v{i}" for i in range(12, 91))
    parser.add_argument("--variant", choices=variants,
                        help="Run version shown on the frame; defaults to the report path")
    args = parser.parse_args()
    version = args.variant or next((part for part in args.report.parts
                                    if part in variants), None)
    if version is None:
        raise ValueError("Adaptive video needs a versioned report path or --variant")
    report = json.loads(args.report.read_text())
    selection_mode = report.get("selection_mode")
    if not report["one_sim_without_skill_reset"] or (selection_mode not in (
            "v9_locked", "v9_live", "v10_locked", "v13_locked", "v13_live",
            "v14_locked", "v14_live", "v15_locked", "v15_live",
            "v15_v10_height_locked", "v16_locked", "v16_live",
            "v16_v10_locked", "v16_v10_height_locked", "v16_bias_y_locked",
            "v16_specialist_locked", "v21_scoped_y_locked") and not
            (version in ("v17", "v23", "v34") and report["reach_controller"] == "feedback" and
             selection_mode is None) and not
            (version == "v41" and report.get("ended_early") and
             report.get("image_goal_missing_at_s") is not None and selection_mode is None)):
        raise ValueError("Expected a saved no-reset adaptive RGB-goal result")
    if (report["toy_goal_source"] not in (
            "calibrated_virtual_rgb_known_plane_exact_pose",
            "template_matched_virtual_rgb_exact_pose_known_cube",
            "selective_template_or_calibrated_virtual_rgb_exact_pose_known_cube",
            "observable_negative_camera_partial_correction_or_large_positive_sparse_abstention_or_edge_bias_or_calibrated_virtual_rgb_exact_pose_known_cube") or
            report["body_role"] != "rejected_historical_direct_frame_sandbox"):
        raise ValueError("Expected historical calibrated RGB replay")
    with np.load(args.trace) as trace:
        qpos = trace["qpos"].copy()
        dt = float(trace["step_dt_s"])
    if len(qpos) != len(report["timeseries"]):
        raise ValueError("Trace/report frame count differs")
    source = VirtualRgbGoal(selected_body_sha256=report["body_sha256"],
                            max_age_s=report["max_goal_age_s"])
    if qpos.shape[1] != source.model.nq:
        raise ValueError("Trace/render model qpos differs")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.10, 0.0, 0.17)
    camera.distance = .65
    camera.azimuth = args.azimuth
    camera.elevation = -20
    external = mujoco.Renderer(source.model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    args.video.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    times = np.arange(0, len(qpos) * dt, 1 / fps)
    f_title, f_body, f_meta = font(30), font(23), font(18)
    reach_start_s = report.get("reach_start_s")
    if reach_start_s is None:
        reach_start_s = next(e["time_s"] for e in report["image_goal_capture_events"]
                             if e.get("used_for_control"))
    entry = next(e for e in report["image_goal_capture_events"]
                 if e["time_s"] == reach_start_s)
    entry_error = entry.get("xy_error_truth_mm_scoring_only")
    correction = entry["calibration_correction_xy_mm"]
    contact_s = report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0)
    with imageio.get_writer(args.video, fps=fps, codec="libx264", quality=8,
                            macro_block_size=16) as writer:
        for t in times:
            index = min(len(qpos) - 1, round(t / dt))
            source.data.qpos[:] = qpos[index]
            source.data.qvel[:] = 0
            mujoco.mj_forward(source.model, source.data)
            external.update_scene(source.data, camera=camera, scene_option=option)
            world = external.render().copy()
            source.renderer.update_scene(source.data, camera="lower_rgb", scene_option=option)
            rgb = source.renderer.render().copy()
            bbox = orange_bbox(rgb)
            frame = Image.new("RGB", (1280, 608), "#111a26")
            frame.paste(Image.fromarray(world), (0, 64))
            frame.paste(Image.fromarray(rgb).resize((640, 480), Image.Resampling.NEAREST), (640, 64))
            draw = ImageDraw.Draw(frame)
            bias_mm = report.get("posthoc_diagnostic_goal_bias_body_xy_mm")
            if report.get("adaptive_variant") == "v18_opened_derived_teacher_probe":
                bias_mm = [1000 * value for value in report["goal_bias_body_xy_m"]]
            if version == "v21" and selection_mode == "v21_scoped_y_locked":
                bias_mm = [1000 * value for value in report["goal_bias_body_xy_m"]]
            if version == "v90":
                bias_mm = [1000 * value for value in report["goal_bias_body_xy_m"]]
            if version == "v90":
                title = "prescribed RGB goal shift · opened teacher"
            elif version == "v89":
                title = "rear-view arm route boundary · opened diagnostic"
            elif version == "v88":
                title = "specialist route boundary · opened diagnostic"
            elif version == "v87":
                if "native-fresh-neighbors" in args.report.parts:
                    title = "scoped reach timing · new-seed challenge"
                elif "native-prior75-retention" in args.report.parts:
                    title = "scoped reach timing · older retention"
                elif "native-routes63-retention" in args.report.parts:
                    title = "scoped reach timing · route retention"
                else:
                    title = "height-route timing only · opened diagnostic"
            elif version == "v86":
                title = "global slow approach · opened diagnostic"
            elif version == "v85":
                title = "height boundary · opened diagnostic"
            elif version == "v41":
                title = "visibility-aware RGB + retained arm"
            elif version == "v40":
                title = "RGB route stability + image goal"
            elif version == "v39":
                title = "observed route + corner image rule · opened screen"
            elif version == "v38":
                title = "observable route + trained RGB arm · opened gate"
            elif version == "v37":
                title = "forced route + trained RGB arm · opened screen"
            elif version == "v36":
                title = "retrained RGB arm + wider image rule · opened screen"
            elif version == "v35":
                title = "wider selective RGB · opened diagnostic"
            elif version == "v34":
                title = "RGB feedback teacher · opened gap"
            elif version == "v33":
                title = "forced saved route · opened failure screen"
            elif version == "v32":
                phase_label = "older failure screen" if "prior-failure-screen" in args.report.parts else "opened gate"
                title = f"selective RGB goal + height route · {phase_label}"
            elif version == "v31":
                phase_label = "fresh challenge" if "fresh-reserved" in args.report.parts else "opened gate"
                title = f"automatic template RGB + gaze · {phase_label}"
            elif version == "v30":
                title = "rendered-template RGB · opened diagnostic"
            elif version == "v21":
                phase_label = "fresh challenge" if "fresh-reserved" in args.report.parts else "opened gate"
                title = f"automatic RGB selector · {phase_label}"
            elif version == "v28":
                title = "simulator-truth goal shift · opened diagnostic"
            elif version == "v23":
                title = "saved feedback · opened diagnostic"
            elif version in ("v22", "v24", "v25"):
                title = "forced branch · opened diagnostic"
            elif version == "v20":
                title = "forced Y shift · opened development"
            elif version == "v19":
                title = "scoped actor · opened development"
            elif report.get("adaptive_variant") == "v17_posthoc_forced_branch_diagnostic":
                title = "forced branch · opened diagnostic"
            elif report.get("adaptive_variant") == "v18_opened_derived_teacher_probe":
                title = "forced teacher · opened development"
            elif version == "v18":
                title = "automatic baseline · opened development"
            elif bias_mm is not None:
                title = "goal shift · opened diagnostic"
            elif version in ("v17", "v23") and report["reach_controller"] == "feedback":
                title = "feedback teacher · opened diagnostic"
            else:
                title = "adaptive RGB reach"
            draw.text((20, 16), fitted_label(draw, f"Cove-50 · {version} {title}", f_title, 810),
                      font=f_title, fill="#f5f9fc")
            draw.text((860, 10),
                      f"seed {report['seed']} · {report['rail_v']:g} V · {report['delay_ms']:g} ms",
                      font=f_meta, fill="#c6dbe9")
            route_label = selection_mode or ("route unselected" if version == "v41" else "feedback_locked")
            draw.text((860, 35), fitted_label(draw, route_label, f_meta, 400),
                      font=f_meta, fill="#c6dbe9")
            draw.rectangle((0, 64, 639, 101), fill="#111a26")
            draw.rectangle((640, 64, 1279, 101), fill="#111a26")
            draw.text((16, 73), "Native robot and toy", font=f_body, fill="#e6f4fa")
            draw.text((656, 73), "Virtual RGB · historical body", font=f_body, fill="#e6f4fa")
            if bbox:
                x, y, w, h = bbox["xywh_px"]
                draw.rectangle((640 + 2*x, 64 + 2*y, 640 + 2*(x+w), 64 + 2*(y+h)),
                               outline="#00f5a4", width=3)
            row = report["timeseries"][index]
            phase = row["phase"]
            supported = all(row["feet_grounded"])
            status = "FINGERTIP CONTACT" if row["finger_contact"] else "no fingertip contact"
            draw.text((16, 557), f"t={t:4.1f}s   {phase}   feet: {'both' if supported else 'lost'}",
                      font=f_body, fill="#e6f4fa")
            draw.text((16, 583), status, font=f_body,
                      fill="#62e2ac" if row["finger_contact"] else "#e1bcb4")
            if version == "v41" and report.get("image_goal_missing_at_s") is not None and t >= 7.0:
                camera_status = "No RGB toy goal at entry · run stopped"
            elif t < reach_start_s:
                camera_status = "Awaiting reach-entry RGB goal"
            elif selection_mode in ("v9_live", "v13_live", "v14_live",
                                               "v15_live", "v16_live"):
                camera_status = f"Live corrected RGB goal · age {row['image_goal_age_s']:.1f}s"
            elif version in ("v30", "v31"):
                camera_status = f"Frozen template RGB goal · age {t-reach_start_s:.1f}s"
            else:
                camera_status = f"Frozen corrected RGB goal · age {t-reach_start_s:.1f}s"
            draw.text((656, 557), fitted_label(draw, camera_status, f_body, 605),
                      font=f_body, fill="#e6f4fa")
            if version == "v41" and entry_error is None:
                goal_line = (f"RGB {entry['orange_pixels']}px · goal unavailable · "
                             f"contact {contact_s:.2f}s · FAIL")
            elif version == "v41":
                goal_line = (f"RGB {entry['orange_pixels']}px · template "
                             f"{'on' if entry['template_used_for_control'] else 'off'} · "
                             f"error {entry_error:.1f}mm · contact {contact_s:.2f}s")
            elif version in ("v30", "v31"):
                shift = entry["template_correction_xy_mm"]
                goal_line = (f"Template RGB err {entry_error:.2f} mm · image shift "
                             f"{shift[0]:+g},{shift[1]:+g} mm · contact {contact_s:.2f}s")
            elif bias_mm is not None:
                goal_line = (f"RGB err {entry_error:.2f} mm · body goal shift "
                             f"{bias_mm[0]:+g},{bias_mm[1]:+g} mm · contact {contact_s:.2f}s")
            elif version in ("v17", "v23") and report["reach_controller"] == "feedback":
                goal_line = f"RGB err {entry_error:.2f} mm · saved feedback · contact {contact_s:.2f}s"
            else:
                goal_line = (f"Goal err {entry_error:.2f} mm · Δ "
                             f"{correction[0]:+.1f},{correction[1]:+.1f} mm · contact {contact_s:.2f}s")
            draw.text((656, 583), fitted_label(draw, goal_line, f_meta, 605),
                      font=f_meta, fill="#b5d1e2")
            writer.append_data(np.asarray(frame))
    external.close()
    source.close()
    print(f"rendered {args.video} from {len(qpos)} native qpos at {dt:g}s")


if __name__ == "__main__":
    main()
