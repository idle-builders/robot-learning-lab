"""Mesh convergence of the FEM setups: python3 fem_convergence.py REPORT_COARSE REPORT_FINE [PART ...]
Both reports come from run.py --checks fem (the fine one with --fem-mesh-h H). Per part, load case, material and layer normal: p99 / peak von Mises and
max displacement at the two mesh sizes and the change (fine vs coarse, %); then per part the range of the changes over all rows and the nodes / CG numbers.
Writes CONVERGENCE.json next to the fine report."""
import json
import sys
from pathlib import Path

a, b = (json.load(open(p))["checks"]["fem"]["per_part"] for p in sys.argv[1:3])
parts = sys.argv[3:] or [n for n in a if n in b]
rel = lambda x, y: 100.0 * (y - x) / x if x else 0.0  # noqa: E731
out = {}
for n in parts:
    rows = []
    for ln, bm in a[n]["by_layer_normal"].items():
        for m, cases in bm.items():
            for c, r in cases.items():
                f = b[n]["by_layer_normal"][ln][m][c]
                rows.append(dict(layer_normal=ln, material=m, case=c, **{k: dict(coarse=r[k], fine=f[k], change_pct=round(rel(r[k], f[k]), 1))
                                                                  for k in ("p99_vm_MPa", "peak_vm_MPa", "max_displacement_mm")}))
    rng = {k: [min(r[k]["change_pct"] for r in rows), max(r[k]["change_pct"] for r in rows)] for k in ("p99_vm_MPa", "peak_vm_MPa", "max_displacement_mm")}
    cg = [s for s in b[n].get("solver", []) if s.get("cg_iterations")]
    out[n] = dict(h_coarse_mm=a[n].get("mesh_h_mm"), h_fine_mm=b[n].get("mesh_h_mm"), nodes=[a[n]["nodes"], b[n]["nodes"]], change_range_pct=rng, rows=rows,
                  fine_cg_iterations_max=max((max(s["cg_iterations"]) for s in cg), default=None),
                  fine_cg_true_residual_max=max((s["true_relative_residual_max"] for s in cg), default=None))
    print(f"{n}: h {out[n]['h_coarse_mm']} -> {out[n]['h_fine_mm']} mm, nodes {a[n]['nodes']} -> {b[n]['nodes']}; change fine vs coarse: "
          + ", ".join(f"{k.split('_')[0]} {v[0]:+.1f}..{v[1]:+.1f} %" for k, v in rng.items()) + f"; fine CG iterations max {out[n]['fine_cg_iterations_max']}, residual {out[n]['fine_cg_true_residual_max']}")
    for r in rows:
        print(f"   {r['layer_normal']} {r['material']:24s} {r['case']:34s} " + "  ".join(f"{k.split('_')[0]} {r[k]['coarse']:.3f}->{r[k]['fine']:.3f} ({r[k]['change_pct']:+.1f}%)" for k in ("p99_vm_MPa", "peak_vm_MPa", "max_displacement_mm")))
json.dump(out, open(Path(sys.argv[2]).parent / "CONVERGENCE.json", "w"), indent=1)
