"""Present the virtual serial evidence without calling it physical telemetry."""

import argparse
import html
import json
from pathlib import Path
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/hardware-link-demo-v4")
    report = verify_report(source)
    if report["status"] != "pass":
        raise ValueError("virtual transport run failed")
    data = []
    bindings = {str(source / "report.json"): sha(source / "report.json")}
    for robot in report["robots"]:
        name = robot["name"]
        captures = source / f"{name}-capture"
        capture = verify_report(captures)
        if (
            capture["source_kind"] != "synthetic_transport"
            or capture["command_writes"] != 0
        ):
            raise ValueError("unexpected capture origin/writes")
        paths = [
            source / f"{name}-events.json",
            source / f"{name}-writes.json",
            captures / "events.jsonl",
            captures / "chunks.jsonl",
            captures / "report.json",
        ]
        for path in paths:
            bindings[str(path)] = sha(path)
        data.append(
            {
                **robot,
                "event_count": robot["events"],
                "events": json.loads(paths[0].read_text()),
                "writes": json.loads(paths[1].read_text()),
                "capture": capture,
            }
        )
    intro = """<h1>Connect the future prototype without inventing feedback.</h1><p>The host link is now exercised for Pip and Scout through local virtual serial ports. Every sample here is <strong>synthetic</strong>. No physical device was opened or moved.</p><div class="note">USB write completed ≠ robot command accepted. The firmware has no per-command acknowledgement. Device acceptance, actuation time, joint torque and root pose remain unavailable.</div>"""
    body = ""
    for robot in data:
        checks = "".join(
            f"<li>{html.escape(c.replace('_', ' '))}: pass</li>"
            for c in robot["checks"]
        )
        body += f"""<article><h2>{robot["name"].title()}</h2><p>{robot["captured_frames"]} synthetic frames in the read-only capture · zero command writes from capture.</p><ul>{checks}</ul><p><a href="../hardware-link-demo-v4/{robot["name"]}-capture/report.json">Capture report</a> · <a href="../hardware-link-demo-v4/{robot["name"]}-capture/events.jsonl">Parsed events</a> · <a href="../hardware-link-demo-v4/{robot["name"]}-capture/chunks.jsonl">Original bytes and host arrival times</a> · <a href="../hardware-link-demo-v4/{robot["name"]}-writes.json">Separate STOP write receipt</a></p></article>"""
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prototype serial connection</title><style>body{font:16px/1.5 system-ui;background:#f5f3ed;color:#193d36;max-width:1150px;margin:32px auto;padding:24px}h1{font-size:38px}.note{background:#fff0c7;padding:20px;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{padding:20px;background:white;border-radius:14px}select{padding:8px;font:inherit;margin:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:20px;background:white;max-height:540px;overflow:auto}a{color:#006b58}</style>"""
        + intro
        + '<div class="grid">'
        + body
        + """</div><h2>Inspect a retained event</h2><p>Host arrival and device clocks remain separate. Zero sample time means unsampled; raw zero bytes do not become joint measurements. Malformed lines and rejected identity records remain visible.</p><label>Robot <select id="robot"></select></label><label>Event <select id="event"></select></label><pre id="detail"></pre><h2>What the physical bench must establish</h2><p>Image/map identity, actual USB timing and drops, calibration, joint feedback interpretation, command acknowledgements, stop behavior and measured actuator response. This is a supported-rig transport candidate, not onboard balance or a ready learned-policy adapter. Canonical Rerun recording conversion remains separate work.</p><p><a href="../../docs/design/hardware-link.md">Host contract, capture procedure and retained failures</a> · <a href="../../docs/design/actuator-hold-bench.md">Actuator bench</a> · <a href="report.json">Presentation source bindings</a></p><script id="data" type="application/json">"""
        + json.dumps(data).replace("<", "\\u003c")
        + """</script><script>const data=JSON.parse(document.getElementById('data').textContent),r=document.getElementById('robot'),e=document.getElementById('event'),d=document.getElementById('detail');function option(v,t){const x=document.createElement('option');x.value=v;x.textContent=t;return x;}data.forEach((x,i)=>r.append(option(i,x.name)));function show(){d.textContent=JSON.stringify(data[+r.value].events[+e.value],null,2);}function choose(){e.replaceChildren(...data[+r.value].events.map((x,i)=>option(i,`${i+1}: ${x.kind}`)));show();}r.onchange=choose;e.onchange=show;choose();</script></html>"""
    )
    (output / "index.html").write_text(page)
    sources = [
        Path(__file__).relative_to(Path.cwd()),
        Path("docs/design/hardware-link.md"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Source/display integrity only; synthetic transport evidence",
            physical_status="not_run",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in sources}),
            output_hashes={"index.html": sha(output / "index.html")},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
