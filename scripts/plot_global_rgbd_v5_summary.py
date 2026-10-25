#!/usr/bin/env python3
"""Plot aggregate V5 outcomes and measured, explicitly scoped phase timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    from scripts.global_rgbd_v5_protocol import SCENES

    directory = Path(directory)
    lock = json.loads((directory/'primary64_lock.json').read_text())
    performance = json.loads((directory/'performance.json').read_text())['cases']
    cases = {row['scenario']:row for row in lock['cases'] if row['execution_condition']=='static'}
    native = {row['scenario']:row for row in lock['cases'] if row['execution_condition']=='frozen_rgbd'}
    matrix, labels = [], []
    for scene in SCENES:
        row = cases[scene]
        planned = row['plans']['frozen_rgbd']['success']
        valid = row['physical'].get('frozen_rgbd')=='ALLOW_SAMPLED_NUMERIC_CLEARANCE'
        completed = native[scene]['native_result']=='success'
        matrix.append([2,2,2 if planned else 0,2 if valid else 1 if planned else 3,2 if completed else 1 if planned else 3])
        labels.append(['PASS','completed','planned' if planned else 'base rejected',
            'sampled clear' if valid else 'collision veto' if planned else 'not reached',
            'completed*' if completed else 'vetoed' if planned else 'not run'])
    fig, ax = plt.subplots(figsize=(11,5.8))
    ax.imshow(matrix,cmap=ListedColormap(['#e79a8b','#efc56b','#a8c9ad','#dddddd']),vmin=0,vmax=3,aspect='auto')
    for i,row in enumerate(labels):
        for j,label in enumerate(row):ax.text(j,i,label,ha='center',va='center',fontsize=9)
    ax.set(yticks=range(7),yticklabels=SCENES,xticks=range(5),
        xticklabels=['GT planner','GT execution','RGB-D planner','Independent\nphysical check','RGB-D execution'],
        title='Primary64: planner success, physical validity and native completion are distinct')
    ax.tick_params(labelsize=9)
    ax.title.set_fontsize(12)
    fig.subplots_adjust(left=.085,right=.99,top=.91,bottom=.17)
    fig.text(.02,.025,'*Narrow: same two-waypoint route as free control;44 positive-separation proximity reports, zero observed physical-contact steps.',fontsize=8)
    fig.savefig(directory/'outcomes.png',dpi=180);plt.close(fig)

    selected = [row for row in performance if row['execution_condition']=='frozen_rgbd']
    canonical = [row for row in performance if row['execution_condition']=='static']
    first_read = median(row['camera_acquisition']['semantic']['elapsed_ms']['p50'] for row in selected)
    stages = [('First shared camera read',first_read),
        ('Mapping compute (worker subset)',1000*median(row['mapping_at_capture']['compute_seconds'] for row in selected)),
        ('Mapping transport (worker subset)',1000*median(row['mapping_at_capture']['transport_seconds'] for row in selected)),
        ('Mapping worker total',1000*median(row['mapping_at_capture']['worker_seconds'] for row in selected)),
        ('Original source age at capture',1000*median(row['source_age_seconds'] for row in selected)),
        ('Frozen numeric/support capture',1000*median(row['capture_seconds'] for row in selected))]
    phases = ['world_update_seconds','ik_seconds','rrt_seconds','route_validation_seconds',
        'trajectory_generation_seconds','trajectory_validation_seconds']
    names = ['registration','IK','RRT','route check','trajectory generation','spline check']
    fig, axes = plt.subplots(1,3,figsize=(17,6.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.2,1.1]})
    axes[0].barh([a for a,b in stages],[b for a,b in stages],color='#447b9e')
    axes[0].set(xlabel='median milliseconds',title='Source pipeline\n(overlapping measurements)')
    for index,(_,value) in enumerate(stages):axes[0].text(value+4,index,f'{value:.1f}',va='center',fontsize=8)
    axes[0].set_xlim(right=max(b for a,b in stages)*1.15)
    axes[0].invert_yaxis()
    positions=np.arange(14);bottom=np.zeros(14)
    entries=[(row,source) for row in sorted(canonical,key=lambda r:SCENES.index(r['scenario'])) for source in ('static','frozen_rgbd')]
    for phase,name in zip(phases,names):
        values=np.array([1000*(row['planning'][source]['phases_seconds'].get(phase) or 0.) for row,source in entries])
        axes[1].barh(positions,values,left=bottom,label=name);bottom+=values
    axes[1].set(yticks=positions,yticklabels=[row['scenario']+' '+('GT' if source=='static' else 'RGB-D') for row,source in entries],
        xlabel='milliseconds; internal production phase timers',title='Planner phases\n(unreached phases have no bar)')
    axes[1].invert_yaxis();axes[1].legend(fontsize=7,loc='lower right')
    rows=[row for row in selected if row['request_to_first_command_seconds'] is not None]
    values=[1000*rows[0]['request_to_first_command_seconds'],
        1000*sum(v.get('experimental_owner_wait_seconds') or 0. for v in rows[0]['experimental_veto'].values()),
        rows[0]['kit_interval_ms']['worst'],rows[0]['execution_control_interval_ms']['p95']]
    axes[2].barh(['Narrow request→first command','Research veto wait (both plans)','Instrumented Kit worst','Execution control p95'],values,color=['#537c9b','#d49c42','#b56a64','#689a75'])
    for index,value in enumerate(values):axes[2].text(value+4,index,f'{value:.1f}',va='center',fontsize=8)
    axes[2].set_xlim(right=max(values)*1.15)
    axes[2].set(xlabel='milliseconds',title='Experimental waits\nand native execution')
    axes[2].invert_yaxis()
    for ax in axes:
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(11)
    fig.suptitle('Measured primary64 timings: source age overlaps worker time; counterplanning and GT veto waits are experimental')
    fig.savefig(directory/'source_to_plan_timing.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,default=Path('docs/assets/motion_planning_v5'))
    render(parser.parse_args().evidence_dir)


if __name__=='__main__':
    main()
