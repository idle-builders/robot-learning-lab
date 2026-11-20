# Sequential basic skills on selected R3

Resolve `robot/model/current-model.json`; native physics is its exact
rigid model with rated motor/sensor Runtime at0.25ms and20ms policy commands.
Standing is the first skill. Do not run the next physical skill until its lead
assigns it after standing review/publication.

The retained R3 walker200 at velocity command(0,0,0) provides the useful learned
standing component. This retry changed no weights: two reused diagnostic starts
and four fresh reserved upright starts each retain all8s. Fresh maximum root
drift is7.2–12.5mm; both-foot contact samples99.5–100%; no sampled nonfoot support.
After2s the joint-speed RMS is.0027–.0034rad/s, with no motor saturation. Initial
variation is not a recovery perturbation, and contact sampling is50Hz.

Original Direct Frame posture BC is separately retained. Its named absolute
pose transfer falls from CAD-neutral entry; simply starting at mapped source
home also falls, including the fixed-home reference. A consistent affine
home retarget changes both observed joint origin and commanded target origin.
It holds one nominal diagnostic but falls in the other varied start at1.70s.
This profile is a posture-retargeting clue for squat, not a learned rescue.

All24 full episodes, including resting tails, remain in
`artifacts/cove50-basic-skills-retry/stand-20261027/`; method-specific source
preimages and hash resolution are in
`data/basic-skills-retry-20261027/stand/source-proof.json`. Original reports
remain unchanged. `summary.json` and `publication.json` correct generic walker
metadata: actual81/24 tensors, slot71 yaw-velocity command rather than heading
estimate, one saved R3 actor with zero new weights. Two distinct saved actors
were used across the whole retry. The earlier200 walking rounds are not counted
again.

Commands below reproduce native observations, so use fresh output paths and
freeze a new method rather than overwrite originals. They are not cheap-derived
regeneration recipes:

```sh
.venv-control/bin/python -B control/skills/basic-skills/run.py --freeze-only
.venv-control/bin/python -B control/skills/basic-skills/run.py
.venv-control/bin/python -B control/skills/basic-skills/centered.py --phase probe
.venv-control/bin/python -B control/skills/basic-skills/walker_zero.py --phase fresh
```

`stand-entry.json` defines the separately frozen entry-only comparison;
`run.py --config ... --data ... --out ...` selects that profile. Reuse the
actual source policy normalization, authority, clipping and action history.
Posture BC uses a72+2 task-onehot physical-target decoder, masked history and
0.2 EMA. R3 walker uses81 observations and24 raw-history outputs with its own
target mapping. Neither can substitute for the other.

Cheap derivations from retained originals:

```sh
.venv-control/bin/python -B control/skills/basic-skills/assess.py \
  --run artifacts/cove50-basic-skills-retry/stand-20261027/walker-zero-fresh \
  --out data/basic-skills-retry-20261027/stand/walker-zero-fresh/assessment.json
.venv-control/bin/python -B control/skills/basic-skills/render.py \
  --run artifacts/cove50-basic-skills-retry/stand-20261027/walker-zero-fresh \
  --episodes walker-zero-137304
.venv/bin/python -B control/skills/basic-skills/seal_inputs.py
.venv-control/bin/python -B control/skills/basic-skills/publish.py
```

Rendering uses retained state only, showing complete selected motor cases with
appearance shells hidden and a grey background. Full videos are decoded and
their robot content inspected. `publish.py` prepares JSON only; the campaign
lead publishes/uploads through the existing results hub. No new HTML page,
Runpod request, PPO update or background service is part of these scripts.

## Shallow crouch retry

The original saved squat actor (`8d17...`) is BC in an RSL checkpoint container,
with an untouched critic. Its source-specific active mask, normalization,
clipping, command slew and filter are retained in `squat_profile.py`. Both the
full and one reduced-amplitude current-body retry tip backward. The native
traces show leaning before hip target clipping; the reduced reference also
falls with zero target clipping. This does not establish a mechanical limit.

`com_crouch.py` instead uses the separately frozen current-body COM-preserving
geometry and conditional static-torque audit. Its desired crouch lowers the
root only about 6.3 mm. Position targets alone fall; adding the labelled static
`tau/Kp` command offset produces a supported shallow crouch and return. This is
a new model-based reference, not an inherited learned actor or recovery expert.

`collect_crouch.py` retains four complete reference visits and the actual
pre-action delayed packet, blend command and executed raw-action history.
`crouch_policy.py` expands the saved R3 walker from 81 to 82 inputs with a zero
blend column and explicitly enables all 22 physical targets. The initial
physical outputs match the source exactly. `fit_crouch.py` performs 600 CPU BC
updates from three whole families; the fourth is held out. This candidate falls.
`fit_crouch_entry.py` changes only sampling (half from the first 20 valid rows)
and remains upright but largely ignores the crouch command. Both candidates,
all failed tails and their copied, unused walking critic provenance remain.

`command_student.py` is a distinct fresh 1→32→32→22 Tanh student: scalar live
requested blend in [0,1], named output-joint targets in radians, 3,000 fresh Adam
steps at 0.0003. Its actor reads no state, time, table or critic. The external
scheduler supplies blend; the unchanged Runtime supplies PD and the motor loop.
`evaluate_command_student.py` keeps reference annotations outside the actor.
The original three donor families and held-out fourth are unchanged; four new
native start families are reserved before fitting. The held-out command curve
is the same deterministic trajectory, so offline validation is not evidence of
novel command or recovery generalization. Native support, depth and return are
reported separately.

Inputs and full original observations are under
`data/basic-skills-retry-20261027/squat/` and
`artifacts/cove50-basic-skills-retry/squat-20261027/`. `source-proof.json` resolves
each executed method to exact retained preimages. Source 15-reward provenance
is retained; these small fits optimize supervised action error only, with zero
PPO or squat-value updates. The original deep squat remains open.

Native collection and fitting commands refuse existing evidence; do not rerun
them as cleanup or regeneration. Cheap analysis/publication preparation:

```sh
.venv-control/bin/python -B control/skills/basic-skills/assess_crouch.py
.venv-control/bin/python -B control/skills/basic-skills/crouch_command_diagnosis.py
.venv/bin/python -B control/skills/basic-skills/seal_squat_inputs.py
.venv/bin/python -B control/skills/basic-skills/publish_crouch.py
```

`publish_crouch.py` prepares the existing results-hub manifest and immutable
artifact aliases, without database writes or uploads. The lead performs those
mutations. Clip rendering uses the shared `render.py` and retained native states.
