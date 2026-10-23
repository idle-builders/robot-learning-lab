"""Self-contained local review of retained balance measurements."""

import html
import json
from pathlib import Path


def write_review(report, output):
    output = Path(output)
    traces = {}
    rows = []
    fields = [
        "time_s",
        "truth_x_m",
        "reference_x_m",
        "pitch_rad",
        "estimated_pitch_rad",
        "truth_velocity_m_s",
    ]
    for case in report["cases"]:
        name = case["case"]
        path = output / name / "trajectory.jsonl"
        if path.exists():
            traces[name] = [
                [s[k] for k in fields]
                for s in map(json.loads, path.read_text().splitlines())
            ]
        if name in traces and not traces[name]:
            del traces[name]
        rows.append(
            "<tr><td>"
            + html.escape(name)
            + "</td><td>"
            + case["status"]
            + "</td><td>"
            + str(round(case.get("duration_s", 0), 2))
            + "</td><td>"
            + str(round(1000 * case.get("final_position_error_m", 0), 2))
            + "</td><td>"
            + html.escape(case.get("termination") or case.get("error") or "completed")
            + '</td><td><a href="'
            + html.escape(name)
            + '/result.json">metrics</a></td></tr>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Balance experiments</title>
<style>body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:24px;background:#f4f2eb;color:#172c32}h1{font-size:36px}td,th{text-align:left;padding:10px;border-bottom:1px solid #bccac5}table{width:100%;font-size:14px}select{font:inherit;padding:8px;border-radius:8px}svg{background:white;border-radius:12px;width:100%;margin:12px 0}a{color:#007966}.note{padding:18px;background:#e0e8df;border-radius:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;border-radius:12px}.legend{font-size:14px}</style>
<h1>ROBOT — balance experiments</h1><p class="note">Free-base simulation using IMU and encoder feedback. Actuator dynamics, contact, mass and sensor behavior are assumptions. These results do not qualify the physical robot, rough-terrain motion or manufacturing cost. The disabled-controller case is a deliberate negative control; other failures remain visible.</p>
<div class="columns"><div><label for="case">Inspect scenario </label><select id="case"></select><p class="legend">Green: measured simulation state. Orange: position reference / estimated pitch.</p><p id="end"></p><p><a href="balance-report.json">Complete report and source hashes</a></p></div><div id="movie"></div></div>
<div id="plots"></div><table><thead><tr><th>Case</th><th>Result</th><th>Time (s)</th><th>Final error (mm)</th><th>Termination</th><th>Evidence</th></tr></thead><tbody>ROWS</tbody></table>
<script>const traces=DATA;const cases=CASES;
const select=document.getElementById('case');for(const name of Object.keys(traces)){let o=document.createElement('option');o.textContent=name;o.value=name;select.append(o)}
function plot(rows,index,other,title){const w=1060,h=205,left=68,right=25,top=35,bottom=30;let ys=rows.flatMap(r=>other===null?[r[index]]:[r[index],r[other]]);let lo=Math.min(0,...ys),hi=Math.max(0,...ys);if(hi-lo<.002){hi+=.001;lo-=.001}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;const tmax=rows.at(-1)[0];const x=t=>left+t/tmax*(w-left-right),y=v=>h-bottom-(v-lo)/(hi-lo)*(h-top-bottom);let s=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><text x="16" y="22" font-size="16">${title}</text>`;for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;s+=`<path d="M${left} ${y(v)}H${w-right}" stroke="#dde5df"/><text x="4" y="${y(v)+4}" font-size="12">${v.toFixed(3)}</text>`}for(let i=0;i<6;i++){let t=tmax*i/5;s+=`<text x="${x(t)-8}" y="${h-8}" font-size="12">${t.toFixed(1)}s</text>`}for(const [k,color] of [[index,'#007966'],[other,'#c27b15']]){if(k===null)continue;s+=`<polyline fill="none" stroke="${color}" stroke-width="2" points="${rows.map(r=>x(r[0])+','+y(r[k])).join(' ')}"/>`}return s+'</svg>'}
function update(){let name=select.value,rows=traces[name],item=cases.find(c=>c.case===name);document.getElementById('plots').innerHTML=plot(rows,1,2,'Forward position (m)')+plot(rows,3,4,'Torso pitch (rad)')+plot(rows,5,null,'Forward velocity (m/s)');document.getElementById('end').textContent=`${item.status} · ${item.duration_s.toFixed(2)} s · ${item.termination||'completed'}`;document.getElementById('movie').innerHTML=name==='drive-return'?'<img src="drive-return/episode.gif" alt="Simulated articulated robot driving and returning"><p>10 frames/s, fixed camera; no playback speed-up.</p>':'';}select.value=traces['drive-return']?'drive-return':Object.keys(traces)[0];select.addEventListener('change',update);if(Object.keys(traces).length)update();else document.getElementById('end').textContent='No motion traces; inspect the failure metrics below.';</script></html>"""
    page = (
        page.replace("ROBOT", html.escape(report["variant"].title()))
        .replace("ROWS", "".join(rows))
        .replace("DATA", json.dumps(traces).replace("</", "<\\/"))
        .replace("CASES", json.dumps(report["cases"]).replace("</", "<\\/"))
    )
    (output / "review.html").write_text(page)
