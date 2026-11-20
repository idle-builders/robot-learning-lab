# Claude Code

@AGENTS.md

@docs/operations/working-rules.md

`AGENTS.md` holds the owner's intent, decision rules and taste for every agent. `docs/operations/working-rules.md` holds the current procedures. Notes for Claude:

- Make decisions yourself in line with the intent. When the intent does not settle a question, or a target, spending limit or hard-to-undo step is involved, ask with `AskUserQuestion`. Ask one decision per question and put real numbers (cost, GPU hours, tempos) in the options.
- Before spending GPU time or money, or proposing hardware, re-read "Owner intent and cost".
- Never run simulation, training, evaluation rollouts or smoke tests on the local CPU (owner, 2026-11-12). A smoke test is a short GPU job. CAD verification (static FEM, collision sweeps) runs locally under `infra/tools/memguard.py`; RL learning never does.
