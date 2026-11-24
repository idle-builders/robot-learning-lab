# APE0 Neck & Head 4-Stage 3D-Printable FDM Connector Redesign (`r1-20261104`)

- **Idle Task:** `T-zesmm6`
- **Dashboard:** `http://127.0.0.1:8805/design/` (`robot_sections: ["Neck", "Head"]`)
- **Published Studies:**
  - `cove50-neck-head-fdm-truss-box-r1` (Family 1: Blank-State 3D Space-Truss & Ribbed Box-Cradle)
  - `cove50-neck-head-fdm-topopt-r1` (Family 2: Blank-State 3D SIMP Topology-Optimized Monocoque)
  - `cove50-neck-head-fdm-combined-r1` (Combined Optimal 4-Stage Neck & Head Architecture)

## Replaced Legacy Parts (14 Parts → 4 Monocoque Stages)
- Stage 1 (`stage1_neck_root_cradle` on `torso` — 6 pressed-in 304 SS sleeves): replaces `fixed_neck_ribbon_anchor_v21`, `fixed_neck_ribbon_cap_v21`, `fixed_neck_ribbon_pad_rear_v21`, `fixed_neck_ribbon_pad_front_v21`, `fixed_ribbon_M2x8_0/1_v21`, `fixed_ribbon_M2_washer_0/1_v21`.
- Stage 2 (`stage2_neck_yaw_to_pitch_link` on `neck_yaw` — 8 pressed-in 304 SS sleeves): replaces `upper__one_piece_bent_neck_L`.
- Stage 3 (`stage3_neck_pitch_to_head_cradle` on `head` — 6 pressed-in 304 SS sleeves): replaces `compact_neck_U_strap_v23`.
- Stage 4 (`stage4_head_sensor_compute_carrier` on `head` — 14 pressed-in 304 SS sleeves): consolidates `single_shared_carrier_v23` and `microphone_compact_tab_v23` (and eliminates `mic_foot_M2x3_0/1`).
