"""Put the two observed responses and ignored request on one V67 clock."""

from __future__ import annotations

import json

import cv2
import imageio
import numpy as np

import owner_deferred_recovery as v67


def main():
    report = json.loads((v67.OUT / "report.json").read_text())
    base = json.loads((v67.v64.OUT / "report.json").read_text())
    starts = {x["path"]: x["run"]["video"]["frames"]-5
              for x in base["results"]}
    rows = []
    for item in report["results"]:
        reader = imageio.get_reader(item["run"]["video"]["path"])
        frames = [frame for frame in reader]
        reader.close()
        label = ("outward / owner cooperates" if item["path"] ==
                 "outbound_control" and item["cooperative"] else
                 "outward / request ignored" if not item["cooperative"]
                 else "far-west / owner cooperates")
        rows.append((label, frames, starts[item["path"]]))
    output = v67.OUT / "comparisons/second-request-recovery.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=5, codec="libx264",
                            macro_block_size=None) as writer:
        for i in range(43):
            panels = []
            for label, frames, first in rows:
                frame = frames[min(first+i, len(frames)-1)]
                panel = cv2.resize(frame, (520, 264),
                                   interpolation=cv2.INTER_AREA)
                cv2.rectangle(panel, (0, 0), (520, 28), (12, 18, 22), -1)
                cv2.putText(panel, label, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .50,
                            (245, 245, 245), 1)
                panels.append(panel)
            writer.append_data(np.concatenate(panels, axis=1))
    print(output)


if __name__ == "__main__":
    main()
