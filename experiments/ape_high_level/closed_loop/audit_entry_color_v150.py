"""V150: chromatic identity evidence from saved V149 owner/candidate crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT,sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v150"
V149=ROOT / "artifacts/ape-closed-loop-cove50-v149"


def descriptor(rgb:Image.Image,box:list[float]) -> dict:
    crop=rgb.crop(tuple(box)).convert("HSV")
    hsv=np.asarray(crop,dtype=float)/255.
    hue,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    foreground=(sat>.23)&(val>.18)
    if not np.any(foreground):
        return {"saturated_pixels":0,"red_fraction":0.,"green_fraction":0.,
                "yellow_fraction":0.,"hue_histogram":[0.]*12}
    h=hue[foreground]
    hist,_=np.histogram(h,bins=np.linspace(0.,1.,13))
    hist=hist/max(1,hist.sum())
    return {"saturated_pixels":int(len(h)),
            "red_fraction":round(float(np.mean((h<.055)|(h>.96))),4),
            "green_fraction":round(float(np.mean((h>.22)&(h<.48))),4),
            "yellow_fraction":round(float(np.mean((h>.08)&(h<.22))),4),
            "hue_histogram":hist.tolist()}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((V149 / "continuous-teach-entry-audit.json").read_text())
    teach=Image.open(V149 / "owner-teach-upper-rgb.png")
    owner=descriptor(teach,audit["teaching"]["owner_confirmed_box"])
    owner_hist=np.asarray(owner["hue_histogram"])
    rows=[]
    for case in audit["rows"]:
        for row in case["rows"]:
            if row["top_box"] is None:continue
            path=V149 / f"seed{case['seed']}-{'absent' if case['absent'] else 'present'}" / (
                f"{row['camera']}-angle-{row['commanded_head_yaw_rad']:+.1f}.png")
            current=descriptor(Image.open(path),row["top_box"])
            overlap=float(np.minimum(owner_hist,np.asarray(current["hue_histogram"])).sum())
            rows.append({"seed":case["seed"],"absent":case["absent"],
                         "camera":row["camera"],
                         "angle_rad":row["commanded_head_yaw_rad"],
                         "dino_similarity":row["top_similarity"],
                         "dino_accepted":row.get("top_accepted_appearance_only",False),
                         "correct_evaluator_only":row["top_correct_evaluator_only"],
                         "hue_intersection_with_owner":round(overlap,4),
                         "red_fraction_relative_to_owner":round(
                             current["red_fraction"]/max(owner["red_fraction"],.001),4),
                         "color":current,
                         "rgb_sha256":sha(path)})
    result={"scope":"V150 read-only owner-taught HSV saturated-hue histogram and red/green/yellow fraction on saved V149 top proposal crops; evaluator box labels only score discrimination. No actor action changed.",
            "source_sha256":sha(SOURCE),
            "v149_audit_sha256":sha(V149 / "continuous-teach-entry-audit.json"),
            "owner_color":owner,"rows":rows}
    (OUT / "entry-color-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("owner",owner["red_fraction"],owner["green_fraction"],owner["yellow_fraction"],flush=True)
    for row in rows:
        if row["dino_accepted"]:
            print(row["seed"],row["absent"],row["angle_rad"],
                  "correct",row["correct_evaluator_only"],
                  "DINO",row["dino_similarity"],
                  "hue",row["hue_intersection_with_owner"],
                  "red-relative",row["red_fraction_relative_to_owner"],flush=True)


if __name__=="__main__":main()
