# Perception

The robot's perception system: the reusable modules behind the current best results in
[perception and sensing](../../docs/research/perception-and-sensing.md). Restored on 2026-11-12
from `experiments/ape_high_level/` (commit `6fbb278e4`), keeping only the modules other code builds on.

| Folder | What it does |
|---|---|
| `identity/` | Teach and find an object: `active_scan.Vision` (Grounding DINO proposals, DINOv2 features, SAM 2 click crops), the owner's yes/no review bound to the shown crop, pinned model revisions (`model-revisions.json`). `scene.py` is the head-camera MuJoCo scene `active_scan` imports. |
| `people/` | Owner attention from pixels, depth and range with a taught exemplar (`online_owner_actor`), and short-horizon motion of the bound owner surface (`fast_owner_flow`). |
| `range/` | The independent near-body range channel: a separately clocked 10 Hz two-height fan (`capture_range_monitor`), and the camera-only comparison (`rgbd_range_projection`). |
| `localization/` | Translation-only two-plane scan matching (`range_pose_corrector`), visual odometry functions, scan-to-map registration (`localization.ScanLocalizer`). |
| `sensor_models/` | Joint/contact/IMU odometry error model, camera mounting geometry and the observed-free fan, the low-plane scanner model. |
| `worlds/` | Scene inputs: the RGB-D and range renderer for captured rooms with people and props (`real_capture_dynamic_scene`), the Gaussian-world RGB adapter (`observation`, `browser_observer.cjs`), and the phone-capture-to-world builder (`README` of that route in `capture-guide.md`; `spectacular_prepare`, `train`, `replay_sensor_capture`, `audit_sensor_capture`, `sensor-runtime.Dockerfile`). |

**Inputs not in Git:** model weights in `.cache/ape-high-level/models/` (rebuilt from `identity/model-revisions.json`), captured scenes in `assets/real_indoor_scenes/`, and earlier run outputs under `artifacts/` (restored from the bucket by path).

**Whole-task numbers.** The best end-to-end results (range route 28/30, combined fault run 27/30, frozen option head 5/5) were produced by a chain of about 30 runner scripts that patch one another's classes. They are not in the tree; `git checkout history/perception-runners` gives that chain (`experiments/ape_high_level/real_scan_benchmark/`, entry `task_policy_learning/live_option_episode.py`). Rerunning one in the tree means turning the patched layers into settings of these modules.
