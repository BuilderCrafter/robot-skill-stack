"""Offline blind runner. No GT/reference-geometry reader or runtime connection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from . import controls,fixed_grid,reference as ref,visibility
from .inputs import CASES,PACKAGE,load_capture

METHODS={'reference':None,'fixed_grid':fixed_grid,'visibility':visibility}


def verify_seal(output):
    output=Path(output)
    seal=json.loads((output/'blind-seal.json').read_text())
    for name,expected in seal['hashes'].items():
        if ref.digest(output/name)!=expected:
            raise ValueError(f'Blind seal mismatch: {name}')
    return seal


def run(captures,output,method='visibility'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    algorithm=METHODS[method]
    policy=ref.POLICY if algorithm is None else algorithm.POLICY
    sources={p.name:ref.digest(p) for p in sorted(PACKAGE.glob('*.py'))}
    declaration=dict(declared_utc=datetime.now(timezone.utc).isoformat(),method=method,
        policy=policy,source_sha256=sources,controls=list(controls.NAMES),
        stop='First positive-scene or negative-control failure; no new rule',
        input_manifest_sha256=ref.digest(PACKAGE/'external-inputs.json'),no_gt_input=True)
    ref.save(output/'declaration.json',declaration)
    baseline,negatives,timings,hashes,failures={},{},{},{},[]
    previous=json.loads((PACKAGE/'evidence/reference/blind-results.json').read_text())
    previous_controls=json.loads((PACKAGE/'evidence/reference/negative-controls.json').read_text())
    for case in CASES:
        payload,scene,inputs=load_capture(captures,case)
        hashes.update(inputs)
        cpu=time.process_time_ns()
        original,clusters,envelopes=ref.reconstruct(payload.frames,scene,payload.robot_snapshots)
        reference_cpu=(time.process_time_ns()-cpu)/1e6
        reference_times=original.pop('timing_ms')
        if ref.plain(original)!=previous[case]:
            raise ValueError(f'Unchanged reference does not reproduce historical prediction: {case}')
        cpu,wall=time.process_time_ns(),time.perf_counter_ns()
        proposed=[dict(c) if algorithm is None else dict(component=c['component'],**algorithm.qualify(c,payload.frames))
                  for c in original['components']]
        extra_wall=(time.perf_counter_ns()-wall)/1e6
        extra_cpu=(time.process_time_ns()-cpu)/1e6
        baseline[case]=dict(reference=original,models=proposed)
        ref.save(output/(case+'-blind.json'),baseline[case])
        timings[case]=dict(reference_cpu_ms=reference_cpu,reference_wall_ms=reference_times,
                          additional_rule_cpu_ms=extra_cpu,additional_rule_wall_ms=extra_wall)
        candidates=[c for c in original['components'] if c.get('hypothesis') is not None]
        if len(candidates)!=1:
            failures.append(case+': ambiguous or missing nonrobot component')
            break
        candidate=candidates[0]
        accepted=next(c for c in proposed if c['component']==candidate['component'])
        print(case,method,accepted['status'],accepted['reason'],flush=True)
        if accepted['status']!='qualified':
            failures.append(case+': positive scene rejected')
            break
        points=sorted(clusters,key=len,reverse=True)[candidate['component']]
        negatives[case]={}
        for name,(p,frames) in controls.make(payload.frames,points,candidate['hypothesis'],payload.robot_snapshots).items():
            original_control=ref.qualify(p,frames,envelopes)
            if name in previous_controls[case]['controls'] and ref.plain(original_control)!=previous_controls[case]['controls'][name]:
                raise ValueError(f'Historical control does not reproduce: {case}/{name}')
            proposed_control=original_control if algorithm is None else algorithm.qualify(original_control,frames)
            negatives[case][name]=dict(reference=original_control,model=proposed_control)
            print(case,name,proposed_control['status'],proposed_control['reason'],flush=True)
            if proposed_control['status']=='qualified':
                failures.append(case+': invalid control accepted: '+name)
                break
        if failures:break
    for relative,expected in hashes.items():
        if ref.digest(Path(captures)/relative)!=expected:
            raise ValueError('Input changed during replay: '+relative)
    for name,expected in sources.items():
        if ref.digest(PACKAGE/name)!=expected:raise ValueError('Declared source changed: '+name)
    ref.save(output/'blind-results.json',baseline)
    ref.save(output/'negative-controls.json',negatives)
    ref.save(output/'timings.json',timings)
    ref.save(output/'input-manifest.json',hashes)
    passed=not failures and len(baseline)==2 and all(len(c)==len(controls.NAMES) for c in negatives.values())
    ref.save(output/'status.json',dict(failures=failures,completed_cases=list(baseline),all_declared_tests_pass=passed))
    names=['declaration.json','blind-results.json','negative-controls.json','timings.json','input-manifest.json','status.json']
    names.extend(case+'-blind.json' for case in baseline)
    ref.save(output/'blind-seal.json',dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
                                         hashes={n:ref.digest(output/n) for n in names}))
    print('SEALED',ref.digest(output/'blind-results.json'),'failures',failures,flush=True)
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True,help='External rgbd-base-clearance-20261113 archive root')
    parser.add_argument('--output',type=Path,required=True,help='A new directory; never overwrite historical evidence')
    parser.add_argument('--method',choices=METHODS,default='visibility')
    args=parser.parse_args()
    return 0 if run(args.captures,args.output,args.method) else 2


if __name__=='__main__':raise SystemExit(main())
