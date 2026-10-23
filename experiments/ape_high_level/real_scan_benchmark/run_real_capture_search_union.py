"""SAM 2 teaching plus a class-broad search proposal union.

The original detector proposals remain available. The second prompt is a
generic search fallback; neither prompt contains a target object's name.
Everything else in the captured-room owner and teach/find runner is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "teach_find"))
import active_scan  # noqa: E402
import run_real_capture_teach_find as baseline  # noqa: E402


SEARCH_PROMPT = "carton . package . food container ."


class SearchUnionVision(active_scan.Vision):
    def __init__(self, _selector: str, device: str | None = None):
        super().__init__("sam2-click", device=device)
        self._teaching_frame_seen = False

    def detect(self, image):
        original = super().detect(image)
        if not self._teaching_frame_seen:
            self._teaching_frame_seen = True
            return original
        old_prompt = active_scan.PROMPT
        try:
            active_scan.PROMPT = SEARCH_PROMPT
            extra = super().detect(image)
        finally:
            active_scan.PROMPT = old_prompt
        for proposal in extra:
            if not any(active_scan._iou(proposal["box"], prior["box"]) >= .85
                       for prior in original):
                original.append(proposal)
        return original


baseline.Vision = SearchUnionVision


if __name__ == "__main__":
    baseline.main()
