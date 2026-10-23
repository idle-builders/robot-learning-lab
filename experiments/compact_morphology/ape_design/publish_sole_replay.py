"""Publish the matched five-seed sole-policy comparison as a local review page."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-sole-policy-replay-20261002"
ROWS = (
    ("baseline", "Original gym sole", "90 × 60 × 24 mm · 14 mm toe offset"),
    ("thick_visible_boot_locked", "Visible boot outline", "78 × 60 × 24 mm · 7 mm toe offset"),
    ("thin_full_footprint_locked", "Thin full outline", "90 × 60 × 2.5 mm · 14 mm toe offset"),
    ("rounded_thin_full_locked", "Rounded thin outsole", "90 × 60 × 2.5 mm · rounded corners and upper toe"),
    ("thin_visible_boot_locked", "Thin visible boot", "78 × 60 × 2.5 mm · 7 mm toe offset"),
)


def main() -> None:
    panels = []
    for key, title, shape in ROWS:
        reports = [json.loads((OUT / "runs" / f"{key}-{seed}.json").read_text())
                   for seed in range(142, 147)]
        worlds = [report["worlds"][0] for report in reports]
        crossed = sum(world["first_meter"] is not None for world in worlds)
        clean = sum(report["one_meter_with_alternating_steps"] for report in reports)
        complete = sum(world["ended_by_timeout"] for world in worlds)
        durations = ", ".join(f"{world['duration_s']:g}" for world in worlds)
        progresses = ", ".join(f"{world['forward_m']:.2f}" for world in worlds)
        seed142 = worlds[0]
        slip = seed142["contact_slip_m_left_right"]
        video = f"runs/{key}-142.mp4"
        panels.append(f"""
        <article>
          <div class="title"><h2>{escape(title)}</h2><span>{escape(shape)}</span></div>
          <video src="{video}" controls muted playsinline preload="metadata"></video>
          <div class="numbers"><strong>{complete}/5</strong><span>survive 8 s</span>
            <strong>{crossed}/5</strong><span>cross 1 m</span>
            <strong>{clean}/5</strong><span>clean 1 m at end</span></div>
          <p>Five-seed forward distance (m): {progresses}<br>
             Five-seed survival (s): {durations}<br>
             Seed 142 touchdowns: {seed142['touchdown_sequence'] or 'none'} ·
             cumulative stance slip L/R: {slip[0]:.3f}/{slip[1]:.3f} m ·
             peak hand-ground force: {seed142['hand_force_n']:.1f} N</p>
          <p><a href="runs/{key}-142.json">Seed 142 measurements</a> ·
             <a href="models/{key}.xml">Collision model</a></p>
        </article>""")
    baseline = json.loads((OUT / "runs/baseline-142.json").read_text())
    reference = baseline["checkpoint"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ape foot contact replay</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f6f3ed;color:#263431;font:16px/1.45 system-ui,sans-serif}}
header{{padding:28px max(28px,4vw);background:#263a35;color:#f7f3e9}}h1{{font-size:clamp(28px,3vw,42px);margin:0 0 8px}}header p{{max-width:900px;margin:6px 0;color:#e5e2d6}}
.wrap{{padding:24px max(24px,4vw) 54px}}.lead{{max-width:1050px;margin:0 0 22px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}
article{{background:white;border:1px solid #d5dcd5;border-radius:16px;overflow:hidden;box-shadow:0 3px 15px #172e1b12}}
.title{{padding:18px 20px 12px}}h2{{margin:0;font-size:22px}}.title span{{color:#576c63;font-size:14px}}
video{{display:block;width:100%;aspect-ratio:4/3;background:#1c2426}}.numbers{{display:grid;grid-template-columns:auto 1fr auto 1fr auto 1fr;align-items:baseline;gap:5px 7px;padding:16px 20px 0}}
.numbers strong{{font-size:23px;color:#205e49}}.numbers span{{font-size:13px;color:#465b50}}article p{{margin:12px 20px 18px;font-size:13px;color:#41584f}}a{{color:#18604a}}
aside{{margin:24px 0 0;padding:18px 22px;border-left:5px solid #cd7854;background:#fff9f0;max-width:1080px}}
code{{font-size:12px;overflow-wrap:anywhere}}@media(max-width:1050px){{.grid{{grid-template-columns:1fr}}}}
@media(max-width:650px){{.numbers{{grid-template-columns:auto 1fr;row-gap:2px}}}}
</style></head><body><header><h1>What does the learned ape gait stand on?</h1>
<p>One saved learned policy. The same 35 cm elbowed ape, command (+0.35 m/s), five fresh seeds (142–146), 2 ms physics, and locked foot mass/inertia. Only the sole collision profile and matching contact sites change.</p></header>
<div class="wrap"><p class="lead">Press play on the matched seed-142 videos. The original gym box completes its walk. A boot-sized box reaches one metre in every seed but usually falls later. A 2.5 mm sole fails early even if its outline remains 90 × 60 mm. The rounded CAD outsole tests whether sharp corners caused that failure. These are collision proxies; the full CAD skin, compliant rubber, and joint forces are not yet faithfully modeled.</p>
<div class="grid">{''.join(panels)}</div><aside><strong>Engineering reading.</strong> The original policy depends on the foot collision body, especially its thickness; shortening the toe also reduces sustained stability. A printed boot can be thick, so the thin-outsole failure alone does not rule out the visual design. Model the complete boot contact geometry and retrain/evaluate before choosing a final foot. The smaller rectangular boot is an approximation to the sketch shell at z = 10 mm, not an exact CAD collision mesh. The foot mass and inertia are held at the original compiled values; no physical friction or compliance measurements exist yet.</aside>
<p>Checkpoint: <code>{escape(reference)}</code> · SHA-256 <code>39e316e0346cb538692832aaf300b451f2dec0f101f2cca1c7542d2ffe9f7fda</code><br>
Original body SHA-256: <code>{baseline['body_sha256']}</code> · source and all run JSON/logs are in this artifact folder.</p></div></body></html>"""
    (OUT / "index.html").write_text(html)
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
