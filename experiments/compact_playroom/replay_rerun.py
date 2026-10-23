"""Inspect one native Arena evaluation: rendered robot plus measured motion/reward."""

import argparse
import json
from pathlib import Path

import rerun as rr
import rerun.blueprint as rb


parser = argparse.ArgumentParser()
parser.add_argument("evaluation", type=Path)
parser.add_argument("video", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

episode = json.loads(args.evaluation.read_text())
rows = episode["trace_env0"]
is_combo = "selections" in episode
is_groot = "GR00T" in episode.get("controller", "")
origin = episode.get("trace_origin_xyz", episode.get("env0_origin_xyz", [4.5, -4.5, 0.0]))
goal_local = episode.get("goal_local_xyz", [1.15, 0.0, 0.0])
goal = [a + b for a, b in zip(goal_local, origin)]
is_pad = "red floor button" in episode["task"] and not is_combo
is_hand_button = "yellow toy button" in episode["task"] and not is_combo
trace_index = episode.get("trace_env_index", 0)

title = "Berkeley Lite · GR00T visual hand-button policy" if is_groot else (
    "Berkeley Lite · Cosmos-selected two-skill episode" if is_combo else (
    "Berkeley Lite · learned hand-button skill" if is_hand_button else
    "Berkeley Lite · learned pressure-pad skill" if is_pad else "Berkeley Lite · learned ball skill"
))
rec = rr.RecordingStream(title)
rec.save(args.output)
if is_combo:
    combo_hand = episode.get("button_checkpoint") is not None or "GR00T" in episode.get("hand_controller", "")
    hand_controller = episode.get("hand_controller", "RSL-RL PPO")
    scope = (
        "One uninterrupted Isaac Lab Arena episode. Cosmos Reason2 selected each skill "
        "from the simulator image, instruction, and measured subgoal state. "
        f"An RSL-RL policy nudged the ball; {hand_controller} controlled the second skill "
        "through Berkeley's frozen walking policy. Ball completion "
        "requires active-skill foot contact and outside-to-inside goal entry; the ball "
        "may later roll away. The second skill requires measured hand-button or "
        "foot-pad contact above 3 N while that skill is active. "
        f"Final success: {episode['success']}. Only environment 0 is evaluated here."
    )
elif is_groot:
    scope = (
        "One fresh Isaac Lab Arena episode. NVIDIA GR00T N1.7 receives the simulator camera, "
        "robot proprioception, and a text instruction; it predicts walk and left-arm action chunks. "
        "Berkeley's released low-level walker executes the walking command. The button outcome "
        "requires measured palm-to-button force above 3 N. Model calls, predicted chunks and "
        "latency are in the evaluation JSON. "
        f"First-episode result: {episode['first_outcomes'][0]}."
    )
else:
    scope = (
        "Isaac Lab Arena evaluation of a frozen released Berkeley walking policy plus an "
        "RSL-RL learned velocity policy. The camera is the simulator's actual robot render. "
        "The 3D markers show recorded root/object positions, not a CAD replay. "
        + ("Hand-button success requires measured palm-to-button force above 3 N. " if is_hand_button else
           "Pad success requires measured foot-to-pad force above 3 N. " if is_pad else
           "Ball success requires entering the floor goal. ")
        + f"First-episode result: {episode['first_outcomes'][trace_index]}. "
        + f"Only env {trace_index} is visualized; see evaluation JSON for all parallel outcomes."
    )
rec.log("about", rr.TextDocument(scope), static=True)
rec.log("space", rr.ViewCoordinates.FLU, static=True)
rec.log("space/target", rr.Points3D([goal], radii=[0.05 if is_hand_button else 0.13], colors=[[245, 185, 30] if is_hand_button else [235, 35, 55] if is_pad else [55, 220, 100]]), static=True)
if is_combo:
    pad = [a + b for a, b in zip(episode.get("pad_local_xyz", [0.70, -0.35, 0.025]), origin)]
    rec.log("space/pad", rr.Points3D([pad], radii=[0.13], colors=[[235, 35, 55]]), static=True)
    if combo_hand:
        button = [a + b for a, b in zip(episode["button_local_xyz"], origin)]
        rec.log("space/hand_button", rr.Points3D([button], radii=[0.05], colors=[[245, 185, 30]]), static=True)
rec.log("camera/video", rr.AssetVideo(path=args.video), static=True)
views = [rb.Spatial2DView(origin="camera"), rb.Spatial3DView(origin="space"), rb.TimeSeriesView(origin="signals")]
if is_combo or is_groot:
    views.append(rb.TextLogView(origin="controller"))
rec.send_blueprint(rb.Blueprint(rb.Vertical(
    rb.TextDocumentView(origin="about"), rb.Horizontal(*views), row_shares=[0.2, 0.8],
)))

if is_combo:
    for choice in episode["selections"]:
        rec.set_time("simulation", duration=choice["step"] * 0.04)
        rec.log("controller/decision", rr.TextLog(
            f"step {choice['step']}: Cosmos chose {choice['raw_prediction']} "
            f"from {choice['measured_state']}"
        ))
    for call in episode.get("groot_model_calls", []):
        rec.set_time("simulation", duration=call["step"] * 0.04)
        rec.log("controller/model_call", rr.TextLog(
            f"step {call['step']}: GR00T predicted {len(call['action_chunk'])} actions; "
            f"first={call['action_chunk'][0]}"
        ))
if is_groot:
    for call in episode["model_calls"]:
        rec.set_time("simulation", duration=call["step"] * 0.04)
        rec.log("controller/model_call", rr.TextLog(
            f"step {call['step']}: GR00T predicted {len(call['action_chunk'])} actions; "
            f"first={call['action_chunk'][0]}; latency={call['latency_s']:.2f}s"
        ))
        rec.log("signals/model_latency_s", rr.Scalars(call["latency_s"]))

for row in rows:
    rec.set_time("simulation", duration=row["step"] * 0.04)
    rec.log("camera/frame", rr.VideoFrameReference(
        seconds=row["step"] * 0.04, video_reference="camera/video"
    ))
    rec.log("space/robot_root", rr.Points3D([row["robot_xyz"]], radii=[0.06], colors=[[90, 120, 255]]))
    rec.log("space/ball", rr.Points3D([row["ball_xyz"]], radii=[0.055], colors=[[250, 150, 30]]))
    if row.get("hand_xyz") is not None:
        rec.log("space/left_hand", rr.Points3D([row["hand_xyz"]], radii=[0.035], colors=[[130, 220, 255]]))
    rec.log("signals/reward", rr.Scalars(row["reward"]))
    if is_combo:
        rec.log("signals/ball_in_goal", rr.Scalars(int(row["ball_in_goal"])))
        rec.log("signals/pad_pressed", rr.Scalars(int(row["pad_pressed"])))
        rec.log("signals/ball_done_latched", rr.Scalars(int(row["ball_done_latched"])))
        rec.log("signals/pad_done_latched", rr.Scalars(int(row["pad_done_latched"])))
        if combo_hand:
            rec.log("signals/hand_button_done_latched", rr.Scalars(int(row["hand_button_done_latched"])))
        rec.log("signals/skill_id", rr.Scalars(1 if row["skill"] == "ball_nudge" else 2))
    else:
        rec.log("signals/success", rr.Scalars(int(row["success"])))
        rec.log("signals/fall", rr.Scalars(int(row["fall"])))
    ball_contact = row.get("ball_foot_force_n", row.get("foot_ball_force_n"))
    if ball_contact is not None:
        rec.log("signals/left_foot_ball_force_n", rr.Scalars(ball_contact[0]))
        rec.log("signals/right_foot_ball_force_n", rr.Scalars(ball_contact[1]))
    if "pad_force_n" in row:
        rec.log("signals/pad_force_n", rr.Scalars(row["pad_force_n"]))
    if row.get("hand_button_force_n") is not None:
        rec.log("signals/hand_button_force_n", rr.Scalars(row["hand_button_force_n"]))
    for i, name in enumerate(("forward_m_s", "lateral_m_s", "yaw_rad_s")):
        rec.log("signals/command/" + name, rr.Scalars(row["action"][i]))

rec.flush()
rec.disconnect()
print(args.output)
