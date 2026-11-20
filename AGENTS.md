# Robot Learning POC

## Owner intent and cost

This is a private hobby project, paid for out of the owner's pocket. Reading, thinking and analysing existing results locally are cheap. GPU hours, purchased parts and the owner's attention are expensive. Use the cheap things to save the expensive ones. **No simulation, training, evaluation rollouts or smoke tests on the laptop CPU** (owner, 2026-11-12). Physics rollouts, learning runs and smoke tests run on the GPU; when a smoke test is needed, make it a short GPU job. The laptop is for reading, writing code, and analysing or rendering results that already exist. CAD verification (static FEM of printed parts and kinematic collision sweeps) runs on the laptop CPU, each heavy job under `infra/tools/memguard.py`, several in parallel within the machine's budget (owner, 2026-11-20: use the resources, don't kill the computer); RL learning never does (owner, 2026-11-12). Every piece of work must end in something reasonable that changes a decision. Storage and dashboards hold source of truth only; everything derived is recreated by its script or training when needed, and anything that cannot be recreated is a defect to refactor (owner, 2026-11-17).

The goal is a 50 cm ape robot that moves in an agile, playful way, like a little monkey or a baby ape. Technical approaches serve that goal, and they may change. The goal stays.

**The score is agility across all motions. Any single motion is irrelevant on its own.** The controller is a whole-body controller, and the robot is an agile ape design. Both are judged by how well they move across the whole range of motions, never by one motion. The motions in the target and the training tapes are made-up stand-ins. Nobody knows the final set yet. So:

- never tune a reward, a controller, a servo or a body part for one motion or one motion family;
- never let one motion's result drive a run, a requirement or a cost;
- grade over all motions, and over motions the controller has not trained on, shown as distributions;
- use single-motion values only as evidence, to show where and how motion degrades.

- **Make the decisions yourself, in line with this intent.** Ask the owner only in these cases:
  - the intent does not settle the question;
  - a new or changed target;
  - spending beyond the set limits;
  - a step that is hard to undo.
- **Work toward an owner-approved target.** A target names concretely what the robot actually needs to do and how well. Never invent harder versions, or rules like "everything must be covered", and then chase them. A stress test shows where the limits are. It is never a requirement and never drives cost.
- **Think realistically.** Check every ambition against physics, budget and what a hobby robot can really do.
- **Before spending GPU time or money, or proposing hardware,** write down:
  - the question;
  - the approved target it serves;
  - the decision it could change;
  - its expected cost;
  - the cheaper check that came first (a reading of existing evidence, or a short GPU smoke job; never a local CPU simulation).

  If the target, the intent or the value is unclear, stop and question the owner with a structured questionnaire. Do not extrapolate on your own.
- **Check the intent again in every iteration.** Would the owner care about this result? If not, stop that line of work. When the owner rejects a direction, stop spending on it at once.
- **The owner sets GPU and money limits.** Never raise them. Never top up, pay or buy. Leave nothing running or billing after the work ends. Report the GPU hours and money spent on each line of work together with its results.
- **Spend in proportion to value.** Run the cheapest check that can change the decision first, and change one thing per comparison. Spend long runs only on what matters. After two uninformative cycles, rethink instead of spending more.

## Scores, not verdicts

Never produce a yes/no result in any domain: design, mechanics, looks, electronics, control or perception. Every result is a continuous measurement in native units, or a grade on a range. No gate, threshold, keep/exclude list, "covered" flag or review may accept or reject work by itself. Decide on the scores, in line with the owner's intent. When one or two items drive a cost or a requirement, show each item's cost and judge whether the intent justifies it. Ask the owner when it does not settle the question. Report counts of bad events next to the continuous measurements, never as the headline.

## Taste for how to act

- Propose only options that have had the same full evaluation as the options they compete with. A quick screen ranks options internally. It is not a proposal.
- Prefer cheap, simple and few kinds of parts. Add power or complexity only where it measurably adds to the goal.
- Body changes are welcome when they add agility and keep the ape look. Avoid changes that cascade into bigger, heavier or costlier parts. Show notable look trade-offs to the owner as renders for discussion.
- Keep control learning ahead of CAD and electronics.
- Start from the best existing work, not from scratch.
- When results disappoint, question your own setup first (objectives, references, measurements), before asking for more compute or stronger hardware.

## Operations

The working procedures for the current tools and layout are in `docs/operations/working-rules.md`: planning in Idle, the selected design pointer, storage and cleanup, publishing results, and the dashboards. Follow them. They change with the architecture; the intent above does not.
