# New: seed-aligned summaries and paired crossed seed/center bootstrap.
# Development intervals are exploratory; final intervals do not imply real-world validity.
import json
from pathlib import Path
import numpy as np
from common import read_json, write_json
from evaluation import METRICS, write_csv

def crossed_interval(difference, repetitions=2000, seed=2026):
    # difference [seed, center, metric]; sample center IDs jointly for all seeds.
    rng=np.random.default_rng(seed);s,n,k=difference.shape;draws=np.empty((repetitions,k))
    for i in range(repetitions):
        si=rng.integers(0,s,size=s);ci=rng.integers(0,n,size=n)
        draws[i]=difference[si][:,ci].mean((0,1))
    return {'mean':difference.mean((0,1)).tolist(),
            'ci95':np.quantile(draws,[.025,.975],axis=0).tolist(),
            'ci99':np.quantile(draws,[.005,.995],axis=0).tolist()}

def summarize(root,plan,phase='validation',repetitions=2000,output=None):
    folder=Path(output) if output else root/'reports'/phase
    folder.mkdir(parents=True,exist_ok=False)
    if phase=='validation':
        tasks=[(v,s,root/'runs'/f'{v}_seed{s}'/'validation') for v in plan['variants'] for s in plan['seeds']]
        baseline=root/'baselines_validation'
    else:
        final=read_json(root/'final_protocol.json')
        tasks=[(v,s,root/'final_results'/f'{v}_seed{s}') for v in final['variants'] for s in final['seeds']]
        baseline=root/'baselines_final'
    data={};rows=[];conditions=None
    for variant,seed,path in tasks:
        if not (path/'evaluation.json').exists():raise RuntimeError('Missing evaluation: '+str(path))
        meta=read_json(path/'evaluation.json')
        if conditions is None:conditions=meta['conditions']
        if conditions!=meta['conditions']:raise ValueError('Condition ordering mismatch')
        a=np.load(path/'scores.npz')['values'];data.setdefault(variant,{})[seed]=a
        for ci,(profile,m) in enumerate(conditions):
            row={'variant':variant,'seed':seed,'profile':profile,'m':m,'samples':a.shape[1]}
            row.update(dict(zip(METRICS,a[ci].mean(0).tolist())));rows.append(row)
    common_profiles={'uniform_low','uniform_medium','top_stable'}
    masks={'macro':[i for i,(p,m) in enumerate(conditions) if p in common_profiles],
           'small_m':[i for i,(p,m) in enumerate(conditions) if p in common_profiles and m in [2,3]],
           'top_stable_small_m':[i for i,(p,m) in enumerate(conditions) if p=='top_stable' and m in [2,3]]}
    extra=[i for i,(p,m) in enumerate(conditions) if p not in common_profiles]
    if extra:masks['unseen_parameters']=extra
    perseed=[];summary=[];paired=[]
    def packed(variant,indices):
        return np.stack([data[variant][s][indices].mean(0) for s in sorted(data[variant])])
    for subset,indices in masks.items():
        if not indices:continue
        for variant in data:
            a=packed(variant,indices);seedmeans=a.mean(1)
            for seed,value in zip(sorted(data[variant]),seedmeans):
                row={'variant':variant,'seed':seed,'subset':subset};row.update(dict(zip(METRICS,value.tolist())));perseed.append(row)
            row={'variant':variant,'subset':subset,'seeds':len(a)}
            for j,metric in enumerate(METRICS):
                row[metric+'_mean']=float(seedmeans[:,j].mean())
                row[metric+'_seed_sd']=float(seedmeans[:,j].std(ddof=1)) if len(a)>1 else None
            summary.append(row)
            if variant!='A0' and 'A0' in data:
                if set(data[variant])!=set(data['A0']):raise ValueError('Seed mismatch with A0')
                paired.append({'contrast':variant+' - A0','subset':subset,
                    **crossed_interval(a[:,:,:2]-packed('A0',indices)[:,:,:2],repetitions)})
        if all(v in data for v in ['A0','P10','P01','P11']):
            interaction=packed('P11',indices)-packed('P10',indices)-packed('P01',indices)+packed('A0',indices)
            paired.append({'contrast':'P11 - P10 - P01 + A0','subset':subset,
                           **crossed_interval(interaction[:,:,:2],repetitions)})
    baselines=[]
    if (baseline/'complete.json').exists():
        meta=read_json(baseline/'complete.json');b=np.load(baseline/'scores.npz')['values']
        if meta['conditions']!=conditions:raise ValueError('Baseline grid mismatch')
        for subset,indices in masks.items():
            if not indices:continue
            for j,method in enumerate(meta['methods']):
                center=b[j,indices].mean(0)
                row={'method':method,'subset':subset};row.update(dict(zip(METRICS,center.mean(0).tolist())));baselines.append(row)
                for variant in data:
                    paired.append({'contrast':variant+' - '+method,'subset':subset,
                        **crossed_interval(packed(variant,indices)[:,:,:2]-center[None,:,:2],repetitions)})
    write_csv(folder/'by_condition_and_seed.csv',rows);write_csv(folder/'per_seed.csv',perseed)
    write_csv(folder/'summary.csv',summary);write_csv(folder/'baselines.csv',baselines)
    write_json(folder/'paired_intervals.json',{'phase':phase,'metric_order':['kendall','rbo'],'replicates':repetitions,
        'method':'crossed seed and center resampling; common center indices across seeds and methods',
        'interpretation':'Exploratory intervals. 99% intervals may serve the five prespecified core Kendall contrasts; other contrasts are not multiplicity-corrected.',
        'contrasts':paired})
    lines=['# Ablation results','',f'Phase: {phase}. Distances lower are better.','',
           '| Variant | Kendall mean | Seed SD | RBO mean | Seed SD |','|---|---:|---:|---:|---:|']
    for r in summary:
        if r['subset']=='macro':lines.append(f'| {r["variant"]} | {r["kendall_mean"]:.6f} | {r["kendall_seed_sd"]} | {r["rbo_mean"]:.6f} | {r["rbo_seed_sd"]} |')
    lines += ['', 'All conditions and seeds are retained in CSV files. No best-seed selection.',
              'Development results reuse the validation set for checkpoint selection and are exploratory.',
              'Intervals describe sampled centers and these training seeds, not new domains or all possible datasets.',
              'A5 removes encoder attention only. A2 removes explicit summaries, not all statistical information.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report: '+str(folder),flush=True)

