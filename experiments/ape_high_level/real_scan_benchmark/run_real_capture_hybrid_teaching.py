"""Use a generic owner-click crop when present, SAM 2 only if it is missing.

Search still keeps original generic proposals plus the package/carton source.
This is one shared policy for all room and object pairs, not a case list.
"""

from __future__ import annotations

import run_real_capture_search_union as union
import active_scan


class HybridTeachingVision(union.SearchUnionVision):
    def select_click(self, image, click, proposals):
        selected = active_scan._clicked_proposal(proposals, click)
        if selected is not None:
            return [round(v) for v in selected["box"]], {
                "source": "highest-scoring enclosing generic proposal",
                "model_score": selected["detector_score"]}
        return super().select_click(image, click, proposals)


union.baseline.Vision = HybridTeachingVision


if __name__ == "__main__":
    union.baseline.main()
