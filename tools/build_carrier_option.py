"""Regenerate configured spacer CAD and update a bound assembly planning ledger.

Only the loose-ring to linked-carrier option may differ from the retained
configuration. All other part records, uncertainty assumptions and limitations
are preserved. This is an incremental mass derivation, not full-assembly release.
"""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from build123d import export_step, export_stl, import_step

from lab.config import RobotConfig, load_config
from lab.robot import make_robot
from lab.arm_mechanism import arm_parts
from lab.leg_mechanism import mechanism_parts
from lab.neck_mechanism import neck_parts
from lab.mass_model import MassArtifact, PROFILES, aggregate, apply_mass_model, cad_properties, canonical, interface_digest
from lab.export import export_all
from tools.check_camera_support import copy_source, sha, write


def require_spacer_change(old, new):
    expected = old.model_dump(mode='json')
    if expected['dimensions']['yoke_spacer_mode'] != 'loose_rings':
        raise ValueError('source must contain loose rings')
    expected['dimensions']['yoke_spacer_mode'] = 'linked_carriers'
    if canonical(expected) != canonical(new.model_dump(mode='json')):
        raise ValueError('only the yoke spacer mode may change')


def all_spacers(robot):
    parts = arm_parts(robot)[0] + mechanism_parts(robot)[0] + neck_parts(robot)[0]
    return {p.name:p for p in parts if p.role == 'spacer'}


def agree_properties(actual, expected):
    return (np.isclose(actual['mass_kg'], expected['mass_kg'], rtol=1e-7, atol=1e-12)
            and np.allclose(actual['center_m'], expected['center_m'], atol=1e-9, rtol=1e-7)
            and np.allclose(actual['tensor_kg_m2'], expected['tensor_kg_m2'], atol=1e-12, rtol=1e-7))


def run(variant, source_mass, reference_cad, output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(variant)
    source = json.loads((source_mass/'report.json').read_text())
    if source['status'] != 'pass':
        raise ValueError('source mass computation did not pass')
    for name, digest in source['output_hashes'].items():
        if sha(source_mass/name) != digest:
            raise ValueError('source mass output changed: '+name)
    old_models = {p:MassArtifact.model_validate_json((source_mass/p/'mass-properties.json').read_text()) for p in PROFILES}
    old_cfg = RobotConfig.model_validate(old_models['nominal'].resolved_config)
    require_spacer_change(old_cfg, cfg)
    reference_report = json.loads((reference_cad/'report.json').read_text())
    reference_cfg = RobotConfig.model_validate_json((reference_cad/'resolved-config.json').read_text())
    if reference_cfg != old_cfg or reference_report['assembly_step_roundtrip'] != 'pass':
        raise ValueError('reference CAD configuration/export mismatch')
    robot = make_robot(cfg)
    ledger = json.loads((source_mass/'part-ledger.json').read_text())
    parts = all_spacers(robot); old_parts = all_spacers(make_robot(old_cfg))
    generic = {n:p for n,p in parts.items() if 'gripper' not in n}
    if len(generic)!=12 or len(parts)!=14:
        raise ValueError('expected twelve generic modules and two grippers')
    files = [*Path('lab').glob('*.py'),Path(__file__).relative_to(Path.cwd()),variant,Path('robot/base.yaml'),source_mass/'report.json',source_mass/'part-ledger.json',reference_cad/'report.json']
    files += [source_mass/p/'mass-properties.json' for p in PROFILES]
    files += [reference_cad/(n+'.step') for n in generic]
    files += [reference_cad/'resolved-config.json']
    for name in generic:
        if sha(reference_cad/(name+'.step')) != reference_report['output_hashes'][name+'.step']:
            raise ValueError('reference local STEP changed')
    hashes = copy_source(output,{str(p):p for p in files})
    # The retained ledger must reconstruct every old link before any replacement.
    for profile,model in old_models.items():
        if RobotConfig.model_validate(model.resolved_config) != old_cfg:
            raise ValueError('source profiles have different configurations')
        for link, prop in model.links.items():
            reconstructed = aggregate([p['profiles'][profile] for p in ledger['parts'] if p['owner']==link])
            if not agree_properties(reconstructed,prop.model_dump(mode='json')):
                raise ValueError('source ledger differs from its mass artifact')
    changed = copy.deepcopy(ledger); rows=[]
    for row in changed['parts']:
        name=row['name']
        if name not in generic:continue
        part=generic[name]; old=old_parts[name]
        if row['owner'] != part.owner or row['kind']!='cad_material' or row['material']!='polymer':
            raise ValueError('spacer owner/material mismatch')
        reference=import_step(reference_cad/(name+'.step'))
        export_step(part.shape,output/(name+'.step'))
        restored=import_step(output/(name+'.step'))
        export_stl(restored,output/(name+'.stl'),tolerance=.03,angular_tolerance=.08)
        missing=reference.cut(restored).volume;extra=restored.cut(reference).volume
        if len(restored.solids())!=2 or not restored.is_valid or max(missing,extra)>1e-4:
            raise ValueError('configured carrier differs from retained local CAD')
        for profile,density in zip(PROFILES,row['density_kg_m3'],strict=True):
            old_prop=cad_properties(old.shape,old.shape.volume*density*1e-9,np.eye(4))
            if not agree_properties(old_prop,row['profiles'][profile]):
                raise ValueError('original spacer geometry differs from source ledger')
            row['profiles'][profile]=cad_properties(restored,restored.volume*density*1e-9,np.eye(4))
        row['volume_mm3']=restored.volume
        row['spacer_mode']='linked_carriers'
        rows.append(dict(name=name,owner=part.owner,old_solids=8,new_solids=2,missing_reference_mm3=missing,extra_reference_mm3=extra,added_volume_mm3=restored.volume-old.shape.volume))
    if len(rows)!=12:raise ValueError('ledger did not replace every generic module')
    limitations=ledger['limitations']+[
        'Incremental spacer-only CAD update of the retained assembly ledger; other source parts and failures preserved.',
        'Detailed carrier STEP/STL files are link-local; no complete STEP assembly regenerated by this consumer.',
        'Exported dynamics use updated planning tensors with existing primitive contact/visual geometry; spacer contacts and detailed appearance are not represented.',
        'Physical printing, fit, strength, handling and production savings remain unqualified.'
    ]
    changed['limitations']=limitations
    write(output/'part-ledger.json',changed)
    profiles={}
    for profile, old_model in old_models.items():
        links={link:aggregate([p['profiles'][profile] for p in changed['parts'] if p['owner']==link]) for link in old_model.links}
        model=MassArtifact(schema_version=1,profile=profile,evidence='unmeasured_assembly_planning_model',resolved_config=cfg.model_dump(mode='json'),baseline_interface_sha256=interface_digest(robot),source_hashes=hashes,limitations=limitations,links=links,total_mass_kg=sum(p['mass_kg'] for p in links.values()))
        folder=output/profile;folder.mkdir();write(folder/'mass-properties.json',model.model_dump(mode='json'))
        instance=make_robot(cfg);apply_mass_model(instance,folder/'mass-properties.json');export_all(instance,folder)
        compiled=mujoco.MjModel.from_xml_path(str(folder/'supported.xml'));data=mujoco.MjData(compiled)
        # Passive supported model smoke test, not a locomotion/controller evaluation.
        for _ in range(100):mujoco.mj_step(compiled,data)
        finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite or not np.isclose(compiled.body_mass.sum(),model.total_mass_kg,rtol=1e-8):
            raise ValueError('mass export or supported model smoke failed')
        profiles[profile]=dict(mass_kg=model.total_mass_kg,added_mass_kg=model.total_mass_kg-old_model.total_mass_kg,model_load_and_100_passive_steps='pass',controller_evaluation='not_run')
        # Match the existing mass-model run boundary consumed by lab.balance.
        write(folder/'report.json',dict(
            status='pass',scope='mass tensor export and 100 passive supported steps only',
            mass_model=instance.mass_model,assembly_evidence=source['assembly_evidence'],
            not_run=['balance','manipulation','physical_mass_measurement','manufacturing_qualification'],
            output_hashes={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))
    write(output/'report.json',dict(status='partial',geometry_and_mass_status='pass',physical_status='not_run',scope=__doc__,source_hashes=hashes,source_assembly_evidence=source['assembly_evidence'],parts=rows,spacer_solids=40,unchanged_gripper_rings=16,profiles=profiles,limitations=limitations,output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}))
    print(json.dumps(profiles),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['variant','source-mass','reference-cad','output']:parser.add_argument('--'+key,type=Path,required=True)
    run(**vars(parser.parse_args()))
