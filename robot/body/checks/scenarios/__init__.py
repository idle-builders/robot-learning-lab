"""More evaluate.py scenarios, one module per area (fasteners.py, part_checks.py, fem.py, pairs.py, runner.py).

A module imports what it needs from evaluate (scenario, run_tool, basics, copy_manifest, part, translate_part, truth_pair,
results_of, _shift_in_place, selected_manifest, ROOT, REPORTS, THIGH, THIGH_NB, LIGHT) and registers its scenarios
with @scenario(id, group, purpose). Each scenario runs the real run.py on the selected body (or a copy with one injected
fault) and returns measurements; an in-place file rewrite only through _shift_in_place (refused in the main checkout)."""
