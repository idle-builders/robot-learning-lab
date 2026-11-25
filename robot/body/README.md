# Body

The robot body is the sum of the parts its definitions declare: one `<builder>.bom.yaml` next to each builder script
(schema [bom/schema.json](bom/schema.json)), with the builder's generators, the parts it owns and its assembly steps. The
body manifest is declared once, in `body:` of [complete_body.bom.yaml](complete_body.bom.yaml) (the output of the last
generator of the chain); `.venv/bin/python -B robot/body/bom/make.py body` builds it, `make.py status body` shows what is out
of date. Generated files go to `artifacts/robot/body/<folder>/`.

| Folder | Builders |
|---|---|
| `assembly/` | The first body manifest from the section packages, held parts and the neck collar |
| `motor-refit/` | Refit to the locked three-tier servos (ST3025, STS3250, ST3215-C047), hip-yaw and ankle-coupler screws |
| `ape-proportions/` | Ape proportions and the shell pass |
| `leg/`, `arm/`, `neck-head/`, `torso/` | Section connectors and shells; the section builders (`rebuild: false`) and their pinned packages |
| `hip-shell/`, `design_kit/` | Hip shell work and the shared shell / connector kit |
| `fastening/`, `standard-hardware/` | Screw measurement and generated standard hardware |
| `checks/` | The shared design check (`run.py`) |
| `bom/` | **The BOM tooling, start here for hardware** ([bom/README.md](bom/README.md)): definitions, make, BOM, publish |

CAD verification runs locally under `infra/tools/memguard.py`. A dashboard study is a `study` block in a definition,
published by `bom/publish_body.py`; its commit is the version.
