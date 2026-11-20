# Forward-policy retry: compute

The new account's Terraform-owned endpoint has completed native R3 GPU smoke,
whole replay and a two-round actor-only PPO pilot. That pilot was plumbing
evidence: it restarted the critic, and its adapted robot fell earlier. Its
native evidence remains at `artifacts/cove50-walk-resume/runpod-trial-r3-20261026/`
and the [pilot dashboard](http://127.0.0.1:8805/?result=cove50-forward-gpu-pilot-20261026).

The corrected continuation uses both saved networks and their normalizers,
the original critic's privileged feature meanings, and the checkpoint's
effective Adam learning rate. `recipe.json` names the capacity measurements,
fresh paired replay seeds and bounded training allocation. `bundle.py` builds
the exact closure; `launch_fit.py submit` sends one request, `status` makes one
explicit provider read, and `fetch` restores worker-pushed stage packs by hash.
Status/progress inspections share a five-minute minimum interval. See
`docs/operations/runpod-infrastructure.md` for provisioning and teardown.

Retain full native episode traces and selected checkpoints. `assess_fit.py`
aggregates native-unit distributions and can render complete replay states.
`bundle_snapshot.py` preserves small exact source/configuration inputs and
verifies byte-identical transport regeneration before disposable input archives
are deleted. `restore_trial_bundle.py` restores the older frozen pilot input.

The R2 continuation completed 200 PPO rounds and 4,000 optimizer steps. Both
fresh eight-second replays show alternating forward steps (1.28–1.33 m), with
substantial lateral drift and imperfect speed/yaw tracking. Full videos and
before/after evidence are in the
[saved actor/value result](http://127.0.0.1:8805/?result=cove50-forward-actor-value-fit-20261026).
The temporary endpoint was removed through Terraform; the volume and template
remain. Closed-run transport archives are disposable duplicates of retained
native files. `transport_cleanup.py` verifies complete member regeneration
before deleting local/remote archives, retaining hashes and restore commands
in the run's `transport-cleanup.json`. Do not fetch removed transport keys.

## Historical old-account check

The 26 October check authenticated with Runpod, but pod and network-volume
inventory each returned HTTP 429, `rate limit exceeded for the day window`.
Account spend was $1.101/hour, shared and unallocated. No resource was created,
no workload was submitted, and no reset time was inferred. Other active project
work was left untouched. This receipt does not establish GPU readiness.

The original compact observations and selected model hashes are in
`data/forward-urdf-retry-20261026/compute/health.json`. The selected R3 URDF,
native MJCF, actuator, sensor and timing hashes matched local inputs.

After the provider limit is resolved, perform one fresh read-only check:

```sh
.venv/bin/python -B control/common/compute/check_health.py --out data/forward-urdf-retry-20261026/compute/health-NEW.json
```

There are no automatic retries or resource mutations. A successful inventory
must be followed by a real selected-task CUDA reset/step, delayed sensor/action
comparison and whole-episode throughput measurement before choosing the PPO
world count or training budget. Keep the old forward reward/configuration and
the selected R3 physics identities separate; source physics options are not
the selected body's options.

The local zero-update comparison is published at
http://127.0.0.1:8805/?result=cove50-forward-urdf-retry-20261026 . Its four whole
CPU episodes are motion evidence, with zero optimizer updates; they do not
substitute for a GPU task check. See `control/skills/walker/` for
the separately prepared policy/task/runtime integration.

`--observed` packages retained observations offline, without API traffic.
`data/forward-urdf-retry-20261026/compute/regeneration.json` records verified
semantic regeneration and removal of its temporary output. No remote output
or backup was made for these compact receipts.
