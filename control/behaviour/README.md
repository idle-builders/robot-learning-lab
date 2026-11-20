# Behaviour

High-level decisions on top of perception, from [high-level behaviour](../../docs/research/high-level-behaviour.md).

| Folder | What it does |
|---|---|
| `options/` | The frozen learned option head (V4/V6) and its trainers: `option_controller`, `owner_route_policy`, `train_event_policy`, `train_owner_route_options`, `policy_store` (models in the bucket under `artifacts/runs/ape-high-level-task-policy-20261016`). |
| `route/` | Viewpoint routing from range sectors (`fan_view_route`), including the low-range closing and path-timed occupancy guards. |

The whole-task harness that ran these (`live_option_episode`) is in history: `git checkout history/perception-runners`.
