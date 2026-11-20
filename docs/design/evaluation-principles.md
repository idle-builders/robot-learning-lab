# Evaluation principles: metrics, judgement, graded distance to viable

User rule, 2026-10-20. It applies to **every domain of the robot**: mechanical design, joints, shell and
aesthetics, structure, mass, electronics and power, firmware, control and training, perception, and
operations.

## 1. No pass/fail

Nothing in this project is decided by a binary verdict.

- **No PASS/FAIL.** No gate, threshold, checklist, review or script may "accept", "reject", "call" or
  "promote" work on its own.
- **Thresholds are reference points.** Requirements and tolerances describe where we currently believe
  "good enough" lies. Being past a reference point is a finding to explain and grade, not a kill.
- **Every number is evidence.** Collect the metric cleanly, in native units, with its source and
  conditions. Distributions beat single values. Keep the severe cases visible.
- **Judgement is qualitative and written down.** Say what the setup is, how it behaves, where and how it
  degrades, and what it would take to improve it.
- **Grades are on a range.** Each dimension gets a grade for how far it is from a viable solution (below),
  with the reason and the confidence of the evidence.
- **Humans decide trade-offs.** Above all, aesthetic adaptations (bulges, recesses, seams, openings, cover
  shapes) are rendered and shown to the user for discussion, never judged by a threshold.

Historical gate verdicts ("PASS", "FAIL", "called failed") in older reports stay as provenance. Re-read them
as evidence and grade them again; do not treat them as decisions.

## 2. Distance-to-viable scale

Grade each dimension separately. Do not average grades into one number. The overall assessment is a
written judgement that names the dimensions that dominate it.

| Grade | Meaning |
|---|---|
| **0 · unknown** | Not yet addressed, or no usable evidence. |
| **1 · far** | Fundamental conflict. No known path without changing the concept *or* the requirement. |
| **2 · major gaps** | A path is plausible but unproven. Several large open problems. |
| **3 · workable** | Known fixes exist and are sized. The remaining work is engineering, not invention. |
| **4 · close** | Minor gaps only, with credible evidence. Open items are detailing, or tests that are already defined. |
| **5 · viable** | Strong evidence for this dimension. The remaining uncertainty can only be closed physically (bench or build). |

Add **confidence** to every grade:
- *low*: estimate or hand reasoning;
- *medium*: model, CAD or simulation;
- *high*: independently reproduced, or measured.

Also add the **trend** since the last round: improving, flat or regressing.

Typical dimensions for a mechanical section (pick what applies; other domains define their own):

| Dimension | What the grade covers |
|---|---|
| Kinematics and range | DOF, reach and dexterity across the workspace |
| Fit | Envelope, collisions, clearances, protrusion, the mirrored side |
| Structure | Load paths, joints, bearings, stops, compliance (LC1–LC4) |
| Drive | Torque, speed, thermal, transmission life, backlash |
| Mass | Honest total against the budget |
| Buildability | Parts that exist, fasteners, assembly sequence, service |
| Harness and covers | Moving cables, cover strain and fatigue |
| Looks | Renders reviewed **with the user** |
| Completeness | What is modelled versus only described |

## 3. Challenge both the solution and the requirement

Whenever evidence shows a gap, ask **two** questions:

1. **Is the solution wrong?** Could a different design or a fix close the gap?
2. **Is the requirement wrong?**
   - Where does the number come from: user decision, physics, a reference robot, or a guess?
   - What does it buy: a behaviour, a margin, or just habit?
   - What does it cost in mass, complexity or looks?

Record both answers. When a requirement looks wrong, propose a revised value with the evidence and put it to
the user. Never quietly relax it, and never quietly keep killing designs against it.

Examples from the hip studio:
- the 360 g hip budget turned out to be a requirement problem (raised to 400 g by the user);
- the visible-recess and appearance limits were aesthetic judgements disguised as hard requirements.

## 4. Reporting template (any domain)

For each candidate and round:

```
<candidate> · <round>
Evidence (native units, source, conditions): …
Dimension grades: <dimension> <0–5> (<confidence>, <trend>): one-line reason · …
Requirement challenges: <requirement> — solution-side fix? requirement-side question? proposal …
Visible trade-offs for the user: <renders + what to decide>
Overall: <written judgement of distance to viable, the dominating gaps, the next most valuable step>
```

Reviews follow the same template. A reviewer grades and explains. They do not issue verdicts. Severity
words (critical, significant, minor) describe how far an issue moves a grade. They are not blockers.
