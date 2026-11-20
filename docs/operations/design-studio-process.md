# Joint design studio: process

Applies to every joint section: hip, knee, ankle/foot, shoulder, elbow and neck. Written 2026-10-20 from
the hip studio's lessons.

Read first: [evaluation principles](../design/evaluation-principles.md) (no pass/fail; the 0–5
distance-to-viable scale; challenge the solution *and* the requirement).

## What the studio produces

For each joint, **three genuinely different solutions** (different drive allocation or mechanism), each
with:
- a **graded evidence card**: per dimension a 0–5 grade, confidence and trend, with metrics in native
  units;
- a written overall judgement of the distance to viable and the next most valuable step;
- renders of every visible adaptation, put to the user for an aesthetic decision;
- requirement challenges that the evidence raised, with proposed values.

There is no "accepted" or "failed". A solution is *closer* or *further* from viable. Work on it stops when
the lead and the user judge that the next round is unlikely to move its grades enough to be worth it. That
is a recorded judgement with reasons, not a threshold.

## Roles

- **Lead** (orchestrating session):
  - plans rounds and keeps the Idle task for the joint (one task covers all variants);
  - curates the evidence;
  - grades together with reviewers;
  - surfaces trade-offs to the user.
- **Designers A / B / C:** three lenses that keep the solutions different.
  - A, Packager: local and compact.
  - B, Mass Centraliser: proximal motors and transmissions.
  - C, Mechanism Inventor: coupled, parallel or custom mechanisms.
- **Independent reviewer:** grades and explains, using the same template. Reviewers never issue verdicts.
- **User:** decides aesthetics, requirement changes and when to stop investing in a variant.

## Rhythm

1. **Diverge.** Each designer works alone. At least four concepts each, at block-model level, with sourced
   motors and arithmetic. Each gets a graded card with low confidence.
2. **Converge.** The lead groups the concepts, removes duplicate architectures, and writes the board:
   shared facts, measured room, tool notes, and which concept each designer develops.
   **Surface the aesthetic adaptations** each concept needs to the user now, with renders, *before*
   development. Do not let a looks metric silently steer the choice.
3. **Develop rounds.** Each round:
   - change something substantive;
   - run the shared design check `robot/body/checks/run.py` scoped to the changed parts, with its
     `render` check for the owner's case drawing (command and checks in `working-rules.md`);
   - grade the evidence per dimension (the sheet-metal studio's `DESIGN-CHECKLIST.md` is at tag
     `history/design-fitkit`);
   - write the round card.
   Keep rounds short and focused. Each round names the one or two dimensions it intends to move.
4. **Review.** An independent reviewer re-measures, grades and challenges. Only after the designer's own
   checklist is complete, so reviews grade quality rather than discover missing basics.
5. **User touchpoints.** After converge, and whenever a visible adaptation or a requirement challenge comes
   up, the lead shows renders and evidence and asks. Never decide looks by a number.

## Requirement challenges

For every recurring gap, the round card asks both questions of the
[evaluation principles](../design/evaluation-principles.md#3-challenge-both-the-solution-and-the-requirement):
what design change would close it, and where the number comes from and what it buys. When the requirement
side looks weak, the lead proposes a revised value to the user with the evidence.

## Round card (template)

```
<joint> · <designer> · <variant> · round N   (date)
Intent this round: <dimensions to move>
Evidence: run.py report (link report.json), LC table, mass ledger, sweep, renders
Grades (0–5, confidence, trend):
  kinematics/range · fit · structure · drive · mass · buildability · harness/covers · looks(user) · completeness
Requirement challenges: …
For the user: <renders + decision asked>
Overall: <distance to viable, dominating gaps, next most valuable step>
```

## Lessons carried over from the hip studio

- Design the **seam and covers first**. Show them to the user early.
- Draw the free body of *your* mechanism before CAD. Joints between members, stops, overhung bearings and
  tension anchors are where reviews kept finding gaps.
- Honest mass from the start: named materials, a full hardware ledger.
- Gate loopholes and checker bugs will appear. Fix the tool and re-measure everyone. Don't re-litigate
  designs.
- Long single-agent rounds get lost when sessions end. Write the round card to disk early and update it.
