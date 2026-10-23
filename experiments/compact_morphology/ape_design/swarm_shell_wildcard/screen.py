"""One neutral exact-solid package screen for the wildcard concept.

The screen finds hard envelope/shell intersections. It is deliberately not a
moving-joint or minimum-clearance certificate.
"""
from __future__ import annotations

import json
from pathlib import Path
from build import model, packages

HERE=Path(__file__).resolve().parent


def overlap_box(a,b):
    aa=a.bounding_box(); bb=b.bounding_box()
    amin,amax,bmin,bmax=map(tuple,(aa.min,aa.max,bb.min,bb.max))
    return all(amin[i]<bmax[i] and bmin[i]<amax[i] for i in range(3))


def volume_both(a,b):
    if not overlap_box(a,b): return 0.0
    ab=a & b; ba=b & a
    first=0.0 if ab is None else ab.volume
    second=0.0 if ba is None else ba.volume
    return max(float(first),float(second))


def main():
    skin=model(); hardware=packages()
    source_cases=[p for p in hardware if any(s in p.name for s in
                  ("XL330","ST3250","ST3215","shared_Tattu","converter_service"))]
    contacts=[]
    for p in source_cases:
        for shell in skin:
            v=volume_both(p.shape,shell.shape)
            if v>0.01:
                contacts.append({"reserve":p.name,"shell":shell.name,"volume_mm3":round(v,3)})
    pair_contacts=[]
    for i,a in enumerate(source_cases):
        for b in source_cases[i+1:]:
            v=volume_both(a.shape,b.shape)
            if v>0.01:
                pair_contacts.append({"a":a.name,"b":b.name,"volume_mm3":round(v,3)})
    report={"scope":"neutral 500 mm exact BRep package wall/pair intersections; both Boolean operand orders retained conservatively",
        "shell_parts":len(skin),"source_package_cases":len(source_cases),
        "case_wall_contact_count":len(contacts),
        "cases_with_wall_contact":len({c['reserve'] for c in contacts}),
        "case_pair_contact_count":len(pair_contacts),
        "wall_contacts":contacts,"case_pair_contacts":pair_contacts,
        "moving_clearance":"not_run","support_fit":"not_qualified",
        "membrane_strain":"not_run","thermal_duty":"not_run"}
    (HERE/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"{len(contacts)} case-to-shell hits on {report['cases_with_wall_contact']}/{len(source_cases)} cases; {len(pair_contacts)} case-pair hits")


if __name__=="__main__":main()
