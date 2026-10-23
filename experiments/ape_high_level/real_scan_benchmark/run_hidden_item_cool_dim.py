"""Same 20-room hidden-change loop with a post-change cool/dim RGB shift."""

from __future__ import annotations

from run_common_policy_20_object_generalization import ROOT, ART
import run_hidden_item_rearrangement as trial


trial.OUT = ROOT / "assets/real_indoor_scenes/hidden_item_cool_dim_20261015"
trial.PRECOMMIT = ART / "hidden_item_cool_dim_precommit.json"
trial.REPORT = ART / "hidden_item_cool_dim_report.json"
trial.POST_CHANGE_PHOTOMETRY = "cool_dim"


if __name__ == "__main__":
    trial.main()
