# Learning methods and failure diagnosis

Use when deciding whether a failed ape behavior needs a better interface, data, controller, model or body. Method popularity is not evidence of improvement on the selected ape.

## Diagnose before changing the learner

| Observed failure | First discriminating check | Learning hypothesis only after that check |
|---|---|---|
| Reward rises without useful movement | Independent signed motion/contact outcome; correct frame, reset and reward input | Change task reward/curriculum if the objective is wrong. |
| Actor ignores a new goal | Goal feature present, correctly normalized, varied in training; target tolerances distinguish commands | Collect goal-conditioned data or retrain on discriminating goals. |
| Grasp tracks the hand path but loses the object | Actual reset, aperture/contact geometry, object motion and teacher feasibility | Add successful diverse/recovery demonstrations or object-aware feedback. |
| A composed pair fails after isolated success | Actual incoming states, offsets, history/phase/recurrent state and possession at switch | Corrective data on the learner's visited states, with a teacher that can recover. |
| Changed body falls | Same source actor on source body, named action map, reset, actuator law and timing; then one body factor | Bounded adaptation to the changed dynamics, retaining an unchanged-actor comparison. |
| Long-horizon behavior diverges | Observation age, unobserved state, distribution shift and accumulated action error | History/recurrent policy, re-observation, recovery data or shorter replanning horizon. |

The local control digest and `interfaces-and-gates.md` contain examples of each distinction. Fixing an observation adapter is not a learned improvement; relabeling a scripted teacher is not training. Also ask whether the requirement is the gap: an outcome definition, reference value or duty limit can be wrong as well as the learner. Record both answers and propose requirement changes to the user ([evaluation principles](../../../../docs/design/evaluation-principles.md)).

## Choose a method that answers the failure

**BC and interactive correction.** Behavior cloning is still a useful small-task baseline. Sequence actions change future observations, so a low supervised loss on expert trajectories does not establish recovery after the learner leaves them. DAgger collects expert labels on learner-induced states under its stated assumptions. Our time-indexed place teacher cannot necessarily provide those corrective labels after a displaced/slipped object; check its competence before adding more samples. [Original DAgger paper](https://proceedings.mlr.press/v15/ross11a.html).

**PPO and task design.** A clipped objective does not guarantee useful contact behavior or safe commands. Preserve command/reset distributions, actor-visible inputs and independent physical scoring. More steps cannot reveal an omitted goal or fit an impossible aperture. The existing MJLab PPO routes remain the control starting point where compatible; a different trainer needs a concrete interface or capability reason.

**Tracking and distillation.** Retargeted motion is a desired reference, not proof of force/contact feasibility. A residual tracker and an absolute-action student require different replay adapters. Preserve phase/history and deployable inputs. Distillation is not a universal performance ceiling: students have outperformed teachers in published supervised settings, but that result does not establish better ape control. Evaluate the student independently on held-out physical outcomes and on any information removed from its inputs. [Born Again Networks](https://proceedings.mlr.press/v80/furlanello18a.html).

**Chunks, diffusion, flow and VLAs.** These are different policy representations and training choices; a VLA is not necessarily a diffusion policy. A fixed MSE action head, autoregressive actions, ACT-style chunks and diffusion/flow experts have different data and execution interfaces. Multimodal demonstrations can make pointwise regression average incompatible behaviors, but a more expressive head can still fail through data mismatch, delay or missing sensing. Keep the working small-task baseline until a matched closed-loop comparison supports replacing it. Do not convert catalog model names into mandatory infrastructure.

**History and privileged training.** History can help infer hidden dynamics only when the relevant signals exist and the trained history/reset convention survives deployment. Simulator-only contact, object pose or dynamics may train a critic/teacher; using them in the deployed actor is a different problem. Label oracle runs as upper bounds. A source policy's observation tensor width alone does not certify compatibility.

## Control layers are implementation choices

Position targets through a PD/servo law are common in the retained ape workflows, but neither PD nor a particular frequency is universal to every learned robot. MuJoCo distinguishes transmission, activation dynamics and force generation; inspect the actual compiled actuator and wrapper before interpreting `ctrl` as a joint angle or torque. [MuJoCo 3.3.6 actuation model](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#actuation-model). Pin the installed simulator version in run evidence; this reference explains concepts, not a required upgrade.

A bounded skill interface is useful when it makes entry conditions, command envelopes, cancel/stop and measured outcome inspectable. Whether a generalist policy or explicit collection of skills performs better for this ape remains an experiment. An upper planner should consume observed completion/failure; language output and elapsed duration cannot establish physical possession or placement.

## Where unresolved claims go

`T-3cjqyd` owns changed-body adaptation, `T-vb9g3a` commanded stopping, `T-sn49d7` corrective-data hypotheses, `T-54ds37` discriminating placement and `T-g49nnf` full continuous composition. Delay/chunk execution belongs to `T-kxky8y`. Read current task state and exact body evidence before work; these IDs are destinations, not claims of successful implementation.
