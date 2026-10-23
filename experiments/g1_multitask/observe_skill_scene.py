"""Turn one recorded Arena G1 frame into the existing Cosmos skill-selector input."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("demo")
    p.add_argument("frame", type=int)
    p.add_argument("goal")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source) as file:
        episode = file[f"data/{args.demo}"]
        tray = episode["states/articulation/surgical_tray/root_pose"][args.frame, :3]
        cart = episode["states/articulation/cart/root_pose"][args.frame, :3]
        image = episode["obs/robot_head_cam"][args.frame]

    tray_gap = float(np.linalg.norm(tray[:2] - cart[:2]))
    station_gap = float(cart[1] - (-3.3))
    lateral_gap = float(abs(cart[0] - 0.35))
    station_xy_gap = float(np.linalg.norm(cart[:2] - [0.35, -3.3]))
    # These are the upstream push_cart.py native termination tolerances.
    station_reached = lateral_gap < 0.50 and abs(station_gap) < 0.30 and abs(float(cart[2]) + 0.7875) < 0.10
    tray_on_cart = float(tray[2]) > 0 and tray_gap < 0.35
    goal_complete = station_reached and tray_on_cart
    choices = "tray_place, cart_push, stop" if goal_complete else "tray_place, cart_push"
    image_path = args.output / "head.png"
    Image.fromarray(image).save(image_path)
    state = (
        f"tray-to-cart XY separation {tray_gap:.3f} m; "
        f"cart-to-station forward gap {station_gap:.3f} m; "
        f"cart-to-station lateral offset {lateral_gap:.3f} m; "
        f"cart-to-station XY separation {station_xy_gap:.3f} m; "
        f"native station success {str(station_reached).lower()}"
    )
    prompt = (
        "Choose the next G1 robot skill for the user's goal using the head-camera image and measured scene state. "
        "Available IDs: tray_place (place tray on cart), cart_push (push cart to station)"
        + (", stop (goal done). " if goal_complete else ". Stopping is infeasible until the loaded cart reaches the native station condition. ")
        + f"Goal: {args.goal} State: {state}. Reply with exactly one available ID: {choices}."
    )
    case = {
        "name": args.output.name,
        "image": str(image_path),
        "goal": args.goal,
        "state": state,
        "prompt_text": prompt,
        "source_hdf5": str(args.source),
        "source_demo": args.demo,
        "source_frame": args.frame,
        "tray_xyz_m": tray.tolist(),
        "cart_xyz_m": cart.tolist(),
        "tray_cart_xy_gap_m": tray_gap,
        "station_reached": station_reached,
        "tray_on_cart": tray_on_cart,
        "goal_complete": goal_complete,
        "available_skill_ids": choices.split(", "),
    }
    (args.output / "case.json").write_text(json.dumps([case], indent=2) + "\n")
    print(json.dumps({"case": str(args.output / "case.json"), "state": state}))


if __name__ == "__main__":
    main()
