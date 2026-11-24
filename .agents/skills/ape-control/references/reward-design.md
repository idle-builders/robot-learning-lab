# Reward design is a design loop

A reward is a design artefact like a mechanism. It needs evidence, iteration and review. It is not a constant
copied from an upstream task. Most APE0 whole-body-controller (WBC) setbacks so far were reward failures, not
body, simulation or training-length failures. Treat the reward as the first suspect whenever a policy survives but
avoids what its reference asks for.

## What went wrong so far (APE0 WBC, 2026-11)

| Symptom seen in native traces | Reward cause | Fix that worked (record) |
|---|---|---|
| Legs barely moved through squats; the torso stayed tall | Only body-position terms with a 12 cm width; nothing paid for joint angles | Joint-angle term, tighter torso termination, feasible tape (step 14, `cove50-wbc-signal-mixed-20261106`) |
| More training traded arm error against leg error and added no agility | Every term near its maximum; the extra budget only re-balanced terms | Not a budget problem (step 16, `cove50-wbc-long2-mixed-20261106`) |
| Knees held 0.08–0.25 rad straighter than the reference | The joint term averages 22 joints at a 0.5 rad width: two knees 0.15 rad off still earn 0.992 | Knee term with a 0.1 rad width: knees within 0.03 rad, squat to full depth, march feet 40 mm (step 19, `cove50-wbc-knee-20261106`) |
| A faster 24 V knee was used only partly: march knee 6.5–10.8 against 13.8–16.9 rad/s; a brief hop flight was trained away | No joint-velocity term; the body velocity terms dilute the knee over 12 bodies; terminating on a missed flight costs more than skipping it | Reward set v2 (shrink-only adaptive widths plus worst-joint, joint-velocity and foot terms): legs tracked twice as closely and knees kept their bend without a knee term (step 20, `cove50-wbc-reward-v2-20261106`). Joint speed is still diluted by the 22-joint mean, and planted feet began to lift; a worst-joint or leg velocity term and a contact match are next. The 24 V knee added no visible value under v2 |
| Planted feet lifted on arm moves; the hop crouch stayed shallow | v2 had no contact phase, and one kernel over all joints | Reward set v3 (per-group joint terms, speed-weighted leg velocity, contact match, swing height): planted-foot lift halved, hop knee offset −0.09 → −0.03 rad, hop flight 1 → 3.5 frames (step 21, `cove50-wbc-loop1-v3-mixed-20261106`). Peak knee speed did not change |
| Every policy lagged the reference by about 20 ms; with half the plant delay the lag returned to 20 ms after training | mjlab rewards compare the robot at t + dt with the reference at t (check 8) | `reference_lead=1` (step 21; the training comparison is `cove50-wbc-loop2-mixed-lead-20261106`) |
| Training on bigger hops reduced the hops (2 cm: 16 → 10.5 frames with both feet up) | Skipping flight is the safer return | Torso height and vertical-speed terms (v4/v5) restored full flight on the ape body (step 22, `cove50-wbc-loop3-ape-v5-20261106`), at the cost of more forward-walk drift |
| The knuckle walk stepped in place (1.4 of 7.5 cm) | Its 8 mm foot lift sat at the swing term's 8 mm threshold; travel was paid only by a fixed 11 cm global root term | Foot lift 12 mm, swing threshold 5 mm, adaptive global root term (v5, v6); result in step 23 |

## Checks before any paid run

1. **Saturation.** Read the logged `Episode_Reward/<term>` of the starting teacher, as a fraction of each term's
   weight. A term above about 0.9 carries almost no gradient. The policy is then free to give up whatever that
   term measures. The v1 terms sat at 0.93–0.99.
2. **Contrast.** Push the reference and degraded copies of it through every term:
   - knees 0.15 rad straighter;
   - feet 10 mm low;
   - joints at half speed;
   - the hop's flight removed.
   Report the fraction each term keeps. A term that keeps more than about 0.95 under a degradation the user
   cares about does not protect that quality. Compute it from the kernel and the error, before spending.
3. **Width against the error.** Keep σ² near the current mean squared error, so the kernel sits where its slope
   is steep (PBHC, arXiv 2506.12851, Eq. 7–8). A fixed width far above the remaining error makes that error
   free. A width far below it gives no gradient to a worse policy. Prefer an adaptive, shrink-only width, as in
   `tracking.AdaptiveTrackingExp`.
4. **Dilution.** A mean over many joints or bodies hides one bad joint. A mean over time hides a short fast phase.
   Add one of:
   - a worst-joint term (`joint_max`);
   - a subset term (knees, feet, end effectors);
   - DeepMimic-style sums instead of means.

   Do not average separate per-joint kernels. mean_j exp(-e_j/s_j^2) caps one joint's cost at 1/n, and a joint far
   off flattens to zero gradient. Two knees 0.15 rad straight kept 83% of such a leg term, against 44% for v2's
   single kernel. Instead, normalise each joint's error by its own width and apply one exponential:
   exp(-mean_j w_j e_j/s_j^2). That form scores 10–35% for the same fault (`tracking.AdaptiveJointExp`). Weighting
   joint-speed errors by the reference's speed, w_j = 1 + |q̇*_j|/2, makes the fast phases count. Legs at 75% speed
   then keep 63–71% of the term, against 83–86% for v2's 22-joint mean.
5. **Coverage.** Every quality the reference carries needs a term that sees it:
   - joint angles;
   - joint velocities (tracked by ASAP, PBHC, HOVER, GMT and DeepMimic; BeyondMimic does not);
   - foot position or height;
   - contact or flight phases (PBHC's contact mask);
   - stance;
   - travel. Body terms relative to the robot's own torso do not pay for it; only a global root term does, and a
     fixed wide width lets the policy step in place.

   Check each event threshold against the smallest event in each reference. A swing term that turns on at 8 mm
   never saw a knuckle walk whose feet lift 8 mm, and the trained walk dragged its feet.
6. **Regulariser cost.** Compute what each penalty costs at the reference's own speed against what the tracking
   terms pay for reaching it. Action rate is the usual suspect. Our residual action (target = reference +
   0.5·a) removes most of its speed dependence, but shrinking widths raise the penalties' relative weight.
   Prefer a penalty curriculum, ramping up from 0.1× (ASAP, PBHC).
7. **Termination risk.** With γ 0.99 at 50 Hz, a termination forfeits about 100 steps of reward, while skipping
   a short flight costs a few. A strict termination therefore teaches the policy to stay on the ground.
   - Our 5 cm torso tolerance is twice as strict as BeyondMimic's 0.25 m scaled to APE0.
   - Loosen terminations around dynamic moves, or tighten them on a curriculum.
   - Do not penalise both feet in the air (HOVER's −200 does exactly that).
   - Sample starts by error, not only by failures (GMT, EGM, `ErrorAwareMotionCommand`): a skipped hop never
     terminates.
8. **Timing.** Check which reference frame each term compares the robot state with. mjlab computes rewards after
   the physics step and before the command advances, so it compares the robot at t + dt with the reference at t,
   and pays for a one-frame lag (20 ms at 50 Hz). On APE0 every trained policy settled at that lag, also on a
   plant with half the delay; at 13 rad/s one frame is 0.26 rad. Use `reference_lead=1`, which makes the command
   and every term use the reference for the robot state's time, and log evaluation traces at the true time
   (`ErrorAwareMotionCommand.reference_now`).

## The iteration loop

The score is agility across all motions (owner, 2026-11-08). The tape's motions are made-up stand-ins, and the final
set is not known. Any single motion, or single family, is irrelevant on its own. Never shape a reward around one.
A reward change is worth a run only when it targets a weakness that spans many motions, and it is judged by
distributions over all motions, including motions not trained on.

1. Name the behaviour the policy avoids across many motions, in native units, from retained full episodes.
2. Run checks 1–8 on the current reward. Write down which term should have paid for the missing behaviour, and
   why it did not.
3. Change one thing per run: a width, a term, a regulariser or a termination. Keep the starting teacher, tape,
   segments, worlds and training wall the same as the comparison run. Use several seeds when the budget allows;
   otherwise say that a run is one seed.
4. Log, per run and per 1 s motion bin where possible:
   - each term's value;
   - its error in native units;
   - the current width (the `adaptive_std` field of each training chunk in `train-native/report.json`);
   - termination causes.
   Compare distributions over worlds:
   - joint error;
   - knee offset;
   - peak joint speed against the reference (p99 includes servo jitter: read it with swing height and lift-offs);
   - swing peak height and lift-offs;
   - frames with both feet up;
   - per-joint current against rated.
5. Grade the result on a range and describe the motion in words (no pass/fail). Keep the reward change if motion
   across all motions moved toward the reference without a new failure mode; record it in the task. A gain on one
   motion or family counts for nothing on its own.
6. When a reward fix works, repeat checks 1–2 on the new reward. The next saturated term is the next suspect.
7. Only when hand iteration stalls, consider automated search. Eureka feeds back per-term values; ADD learns the
   error metric with a discriminator. Each candidate costs a full training run.

Before blaming the body, the actuators, the simulator or the budget, show that the reward pays for the missing
behaviour. A hardware change justifies a bench or a buy only once a reward that asks for the behaviour shows the
current hardware cannot deliver it, and the new hardware can.

## Terms available in `tracking.py`

| Option | What it adds |
|---|---|
| `joint_weight` | `joint_position_tracking`, fixed width `joint_std` |
| `knee_weight` | `knee_tracking`, fixed width `knee_std` (0.1 rad), on the two knees |
| `reward_set='v3'` | `REWARD_SET_V3`: per-group (legs, arms) joint-angle and speed-weighted joint-velocity terms with per-joint shrink-only widths in one normalised kernel; foot contact-phase match (PBHC); swing-foot height while the reference lifts a foot; body position, body velocity and feet as v2 |
| `reward_set='v3r'` | v3 with the global root position on a shrink-only adaptive width (11.4 cm down to 3 cm) |
| `reward_set='v5'` | v4 with that adaptive global root term: full hop flight, but the forward walk drifted twice as far on the ape body |
| `reward_set='v6'` | v5 with the global root weight doubled (0.5 → 1.0) |
| suffix `k` (`v3k`, `v3rk`, `v5k`, `v6k`) | adds knuckle contact match (`HandContactMatch`), for tapes with knuckle-support moves (long arms) |
| `reward_set='v4'` | v3 plus torso height (`anchor_z`, min 5 mm) and torso vertical speed (`anchor_vz`, min 0.05 m/s), weight 0.5 each; on the v2 teacher these contrast mostly on the hop |
| `reference_lead` | the command and every term lead the reference by this many control frames (check 8) |
| `preview_steps` | not a reward: the actor and critic also see the reference this many frames ahead (`reference_preview`) |
| `reward_set='v2'` | `REWARD_SET_V2` (adaptive, shrink-only widths starting at the v1 widths): body position (min 2 cm), body linear velocity (min 0.1 m/s), joint position (min 0.05 rad), worst joint (min 0.05 rad), joint velocity (min 0.5 rad/s, weight 0.5), feet position (min 1 cm, weight 1.0) |
| `stop_weight` | penalty on commanded targets past the joint stops |
| `anchor_tolerance` | torso-height termination in m |

Not built yet, and the obvious next candidates:
- a termination or penalty curriculum;
- adaptive widths on body orientation and angular velocity (they sat at 91–95% of their weight under v3);
- a lower floor on the body-position width (2 cm is wide against a 1 cm hop).

## Sources

Primary papers and code, checked 2026-11-06 at pinned commits:
- BeyondMimic: [arXiv 2508.08241](https://arxiv.org/abs/2508.08241); [mjlab tracking config](https://github.com/mujocolab/mjlab/blob/dcdf6ce519abfd2ea45e9503ae6fb650b75bb869/src/mjlab/tasks/tracking/tracking_env_cfg.py#L209-L278).
- PBHC (KungfuBot): [arXiv 2506.12851](https://arxiv.org/abs/2506.12851); [adaptive σ code](https://github.com/TeleHuman/PBHC/blob/ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0/humanoidverse/envs/motion_tracking/motion_tracking.py#L1030-L1049). In its Table 12 the adaptive width is never the worst of five settings on four motions.
- ASAP: [arXiv 2502.01143](https://arxiv.org/abs/2502.01143), with termination and penalty curricula.
- HOVER: [arXiv 2410.21229](https://arxiv.org/abs/2410.21229); OmniH2O: [arXiv 2406.08858](https://arxiv.org/abs/2406.08858). OmniH2O reports that foot-height and air-time regularisers made the robot stomp.
- GMT: [arXiv 2506.14770](https://arxiv.org/abs/2506.14770), error-weighted sampling ablation.
- SONIC: [arXiv 2511.07820](https://arxiv.org/abs/2511.07820), end-effector term at weight 2.0.
- DeepMimic: [arXiv 1804.02717](https://arxiv.org/abs/1804.02717). Without reference-state starts, the backflip became "a small backwards hop".
- AMP: [arXiv 2104.02180](https://arxiv.org/abs/2104.02180); ADD: [arXiv 2505.04961](https://arxiv.org/abs/2505.04961).
- Walk These Ways: [arXiv 2212.03238](https://arxiv.org/abs/2212.03238), multiplicative auxiliary rewards.
- Eureka: [arXiv 2310.12931](https://arxiv.org/abs/2310.12931), reward reflection.
- AdaMimic: [arXiv 2510.14454](https://arxiv.org/abs/2510.14454). Strict terminations impede exploration of jumps.
