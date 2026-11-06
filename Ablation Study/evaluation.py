# New: aligned per-center metrics and baseline caching; never tune using test.
import csv, json, time
from pathlib import Path
import numpy as np
import torch
from common import synchronize, write_json, digest, read_json
from reference_v3.metrics import distances
from reference_v3.baselines import METHODS, baseline_predictions

METRICS=['kendall','rbo','exact','top10_overlap','top10_displacement']

def scores(truth,predicted):
    kd,rd=distances(truth,predicted,.9)
    k=min(10,len(truth)); pos={item:i for i,item in enumerate(predicted)}
    return [kd,rd,float(truth==predicted),len(set(truth[:k])&set(predicted[:k]))/k,
            sum(abs(pos[item]-i) for i,item in enumerate(truth[:k]))/k]

def grid(section,ms):
    return [(profile,int(m)) for profile in section['conditions'] for m in ms]

def model_scores(model,section,ms,device,batch=32,output=None,metadata=None):
    truth=section['true_rankings'].tolist(); conditions=grid(section,ms)
    values=np.empty((len(conditions),len(truth),len(METRICS)),dtype=np.float64)
    rows=[]; model.eval()
    handle=None
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        handle=(output/'per_sample.jsonl').open('x',encoding='utf-8')
    try:
        with torch.inference_mode():
            for ci,(profile,m) in enumerate(conditions):
                x=section['conditions'][profile]['noisy_positions'][:,:,:m]
                predictions=[];synchronize(device);started=time.perf_counter()
                for start in range(0,len(truth),batch):
                    predictions.extend(model.predict(x[start:start+batch].to(device)).cpu().tolist())
                synchronize(device);elapsed=time.perf_counter()-started
                for index,(target,prediction) in enumerate(zip(truth,predictions)):
                    values[ci,index]=scores(target,prediction)
                    if handle:
                        record=dict(metadata or {},profile=profile,m=m,center_id=index,
                                    true_ranking=target,prediction=prediction)
                        record.update(dict(zip(METRICS,values[ci,index].tolist())))
                        handle.write(json.dumps(record)+'\n')
                row={'profile':profile,'m':m,'samples':len(truth),'inference_seconds':elapsed}
                row.update(dict(zip(METRICS,values[ci].mean(0).tolist())));rows.append(row)
    finally:
        if handle:handle.close()
    if output:
        with (output/'scores.npz').open('xb') as f:np.savez_compressed(f,values=values)
        write_json(output/'evaluation.json',{'metadata':metadata or {},'conditions':conditions,
                   'metrics':METRICS,'summary':rows,'center_count':len(truth)})
        write_csv(output/'summary.csv',rows)
    return values,rows

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def baseline_cache(section,ms,folder,cache_key,seed=2026):
    folder=Path(folder)
    if (folder/'complete.json').exists():
        meta=read_json(folder/'complete.json')
        if meta['cache_key']!=cache_key or digest(folder/'scores.npz')!=meta['scores_sha256']:
            raise ValueError('Baseline cache mismatch')
        return
    if folder.exists(): raise RuntimeError('Incomplete baseline cache; inspect before retry: '+str(folder))
    folder.mkdir(parents=True)
    truth=section['true_rankings'].tolist();conditions=grid(section,ms)
    values=np.empty((len(METHODS)-1,len(conditions),len(truth),len(METRICS)))
    rows=[]
    with (folder/'per_sample.jsonl').open('x',encoding='utf-8') as f:
        for ci,(profile,m) in enumerate(conditions):
            started=time.perf_counter()
            for i,target in enumerate(truth):
                predictions=baseline_predictions(section['conditions'][profile]['noisy_positions'][i,:,:m].numpy(),seed+i)
                for j,method in enumerate(METHODS[1:]):
                    prediction=predictions[method];values[j,ci,i]=scores(target,prediction)
                    row={'method':method,'profile':profile,'m':m,'center_id':i,
                         'true_ranking':target,'prediction':prediction}
                    row.update(dict(zip(METRICS,values[j,ci,i].tolist())))
                    f.write(json.dumps(row)+'\n')
            for j,method in enumerate(METHODS[1:]):
                row={'method':method,'profile':profile,'m':m,'samples':len(truth)}
                row.update(dict(zip(METRICS,values[j,ci].mean(0).tolist())));rows.append(row)
            print(f'Baselines {profile} m={m}: {time.perf_counter()-started:.1f}s',flush=True)
    with (folder/'scores.npz').open('xb') as f:np.savez_compressed(f,values=values)
    write_csv(folder/'summary.csv',rows)
    write_json(folder/'complete.json',{'cache_key':cache_key,'scores_sha256':digest(folder/'scores.npz'),
               'conditions':conditions,'methods':list(METHODS[1:]),'metrics':METRICS,
               'center_count':len(truth),'summary':rows})

