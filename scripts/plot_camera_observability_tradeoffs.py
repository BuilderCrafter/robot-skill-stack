#!/usr/bin/env python3
"""Plot separate measured visibility, runtime, motion and semantic components.

This is a report reducer, not another experiment or an opaque recommendation
score. Runtime uses three fixed1024-ray runs per layout; visibility uses only
the42 shared robot/fixture and fixture/background-clear conditions. Wrist and
single-run optional hardware configurations are not plotted as measured costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
LAYOUTS=(('A_current','Current A','baseline'),
         ('C_asymmetric_1.15_0.65','Repositioned C','baseline'),
         ('K_B_front_height_0.85_rear','Optimized three K','mapping'),
         ('H_third_side_-1.00','Current + side H','mapping'))


def collect(assets):
    names=('geometry_scene_qualifications.json','native_visibility.json','performance.json',
           'planning.json','semantic_spotcheck.json')
    documents={name:json.loads((assets/name).read_text()) for name in names}
    sources={str((assets/name).relative_to(ROOT) if (assets/name).is_relative_to(ROOT) else assets/name):
             hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in names}
    output=[]
    for layout,label,mode in LAYOUTS:
        visible=next(row for row in documents['geometry_scene_qualifications.json']['sensitivity'] if row['layout_id']==layout)['robot_and_background_disjoint']
        if visible['condition_count']!=42:
            raise ValueError('This figure requires the42-condition visibility comparison')
        groups=[row for row in documents['performance.json']['groups'] if row['layout_id']==layout
                and row['mode']==mode and row['sampling_policy']=='fixed'
                and row['global_ray_budget']==1024 and row['valid_measurements']==3]
        if len(groups)!=1 or groups[0]['metrics']['kit_update_ms']['p99']['count']!=3:
            raise ValueError('Expected exactly one three-run fixed1024-ray runtime group: '+layout)
        group=groups[0]
        if mode=='mapping' and group['third_resolution']!=[320,240]:
            raise ValueError('Third-camera comparison must use320x240')
        cases=[row for row in documents['planning.json']['cases'] if row['layout_id']==layout
            and row['cohort']=='primary' and row['strategy']=='reactive' and row['sampling_policy']=='fixed'
            and row['measurement_valid'] and row['global_ray_budget']==1024]
        semantics=[row for row in documents['semantic_spotcheck.json']['rows'] if row['layout']==layout]
        if len(semantics)!=5 or not cases:
            raise ValueError('Missing semantic or physical evidence for '+layout)
        output.append(dict(layout_id=layout,label=label,camera_count=2 if mode=='baseline' else 3,
            visibility=visible,kit_p99_ms=group['metrics']['kit_update_ms']['p99'],
            runtime_case_ids=group['case_ids'],runtime_mode=mode,
            strict_motion_successes=sum(row['strict_contact_free_success'] for row in cases),
            physical_cases=len(cases),physical_scenarios=sorted({row['scenario'] for row in cases}),
            semantic_matches=sum(row['class_count_match'] for row in semantics),semantic_postures=len(semantics),
            actual_native_calibration=documents['native_visibility.json']['actual_native_calibration'][layout]))
    baseline=output[0]['kit_p99_ms']['median']
    for row in output:
        row['kit_p99_relative_cost_percent']=100*(row['kit_p99_ms']['median']/baseline-1)
    return dict(status='MEASURED_COMPONENT_TRADEOFFS',points=output,sources=sources,
        scope=dict(visibility='Area-weighted actual rendered surface over42 matched robot/fixture and fixture/background-clear conditions; support-plane contact excluded from the inter-object qualification.',
            runtime='Median of three per-run Kit p99 values; error bars show their minimum/maximum, not confidence intervals. Ten-second warmup and30-second native headless windows.',
            physical='Primary, valid, fixed1024-ray reactive cases only; strict contact-free goal success. H contains only targeted static/partial cases, so its denominator differs.',
            semantics='Unchanged-processor single-frame original-object class-count spot-check at five postures; not a semantic holdout or manipulation qualification.',
            score='No combined score, no automatic25-percent veto, and no unmeasured wrist or optional single-run costs.'))


def plot(report, destination):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/camera-observability-feasibility/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(9.5,6.2))
    colors=('#2878b5','#e08214','#6b4c9a','#2b8c6b')
    positions=((28.,45.4),(23.1,72.5),(38.4,83.),(33.1,60.4))
    aligns=('left','left','right','left')
    for row,color,position,align in zip(report['points'],colors,positions,aligns):
        x=row['kit_p99_ms']['median'];y=100*row['visibility']['visible_surface_one_fraction']
        axis.errorbar(x,y,xerr=[[x-row['kit_p99_ms']['minimum']],[row['kit_p99_ms']['maximum']-x]],
            fmt='o' if row['camera_count']==2 else'^',color=color,markersize=9,capsize=4,elinewidth=1.7)
        targeted=' targeted' if row['physical_cases']<5 else''
        text=(row['label']+'\n'+f"Strict {row['strict_motion_successes']}/{row['physical_cases']}{targeted}; "
              f"semantic spot {row['semantic_matches']}/{row['semantic_postures']}")
        axis.annotate(text,xy=(x,y),xytext=position,textcoords='data',ha=align,va='center',fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35',fc='white',ec=color,alpha=.96),
            arrowprops=dict(arrowstyle='-',color=color,lw=.9,shrinkA=4,shrinkB=7))
    reference=report['points'][0]['kit_p99_ms']['median']*1.25
    axis.axvline(reference,color='gray',linestyle='--',linewidth=1,zorder=0)
    axis.text(reference+.15,40.7,'+25% runtime-cost reference\nnot an automatic rejection',fontsize=8,color='dimgray',va='bottom')
    axis.set(xlim=(22.2,41.6),ylim=(39.,88.),xlabel='Kit update p99 (ms): median of three native runs',
        ylabel='Native obstacle surface visible (%)\n42 robot- and background-clear conditions',
        title='Visibility gain, runtime cost and downstream usefulness')
    axis.grid(alpha=.18)
    figure.text(.11,.045,'Horizontal bars: range of three run p99 values.  ○ two cameras; △ three cameras (third 320×240).',fontsize=8)
    figure.text(.11,.018,'H has only two targeted physical cases. Wrist and optional single-run costs remain unmeasured here.',fontsize=8)
    figure.tight_layout(rect=(0,.07,1,1))
    destination.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(destination,dpi=180)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'docs/assets/camera_observability_feasibility')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/camera_observability_feasibility/tradeoffs.png')
    args=parser.parse_args()
    report=collect(args.assets_dir)
    report['plot_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plot(report,args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(args.output),points=len(report['points']))))


if __name__=='__main__':
    main()
