"""Measure recording timing without equating host arrival with capture time.

Input timestamps ending _host_ns are explicitly mapped to host monotonic time.
An absent mapping leaves age/latency unknown, even if device timestamps exist.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import numpy as np


def analyse(rows):
    previous = {}
    last_mapped = {}
    trace = []
    for index, row in enumerate(rows):
        r = dict(row_index=index, stream_id=row['stream_id'], sequence=row.get('sequence'),
                 source_clock=row.get('clock_domain'), clock_map_id=row.get('clock_map_id'),
                 clock_uncertainty_ms=None, capture_to_receive_ms=None,
                 capture_to_inference_ms=None, sequence_gap=None, diagnostics=[])
        stream = row['stream_id']; before = previous.get(stream)
        if before:
            if before.get('boot_id') != row.get('boot_id'): r['diagnostics'].append('device_boot_changed')
            elif row.get('sequence') is not None and before.get('sequence') is not None:
                delta = row['sequence'] - before['sequence']
                r['sequence_gap'] = max(delta-1, 0)
                if delta <= 0: r['diagnostics'].append('nonincreasing_sequence')
            a,b = last_mapped.get((stream,row.get('boot_id'))),row.get('capture_host_ns')
            if a is not None and b is not None and b<a: r['diagnostics'].append('mapped_capture_regressed')
        previous[stream] = row
        if row.get('capture_host_ns') is not None:
            last_mapped[(stream,row.get('boot_id'))] = row['capture_host_ns']
        mapped = row.get('clock_map_id') is not None and row.get('capture_host_ns') is not None
        if not mapped: r['diagnostics'].append('capture_not_mapped_to_host')
        else:
            for end,field in [('host_receive_ns','capture_to_receive_ms'),('inference_end_host_ns','capture_to_inference_ms')]:
                if row.get(end) is not None:
                    r[field] = (row[end]-row['capture_host_ns'])/1e6
                    if r[field]<0: r['diagnostics'].append(field+'_negative')
            u=row.get('clock_uncertainty_ns')
            if u is not None: r['clock_uncertainty_ms']=u/1e6
            else: r['diagnostics'].append('clock_uncertainty_unknown')
        if row.get('pose_source') is None: r['diagnostics'].append('capture_pose_unavailable')
        if row.get('health') != 'available': r['diagnostics'].append('source_health_'+str(row.get('health')))
        trace.append(r)
    summaries=[]
    for stream in sorted({r['stream_id'] for r in trace}):
        cohort=[r for r in trace if r['stream_id']==stream]
        d=dict(stream_id=stream,rows=len(cohort),diagnostic_counts=dict(collections.Counter(x for r in cohort for x in r['diagnostics'])))
        for key in ['capture_to_receive_ms','capture_to_inference_ms','clock_uncertainty_ms']:
            values=[r[key] for r in cohort if r[key] is not None]
            d[key]=dict(observed=len(values),missing=len(cohort)-len(values),
                        p10_p50_p90_p99=np.percentile(values,[10,50,90,99]).tolist() if values else None)
        summaries.append(d)
    return dict(streams=summaries,trace=trace,limits='Descriptive recorder audit. No installed latency, policy stop response or clock synchronization quality is inferred from synthetic inputs; negative values remain visible.')


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=[json.loads(s) for s in a.input.read_text().splitlines() if s.strip()]
    report=analyse(data);report['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest();report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['streams']))

if __name__=='__main__':main()
