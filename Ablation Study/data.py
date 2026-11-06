# New: immutable development export, equal-exposure noise controls, held-out centers.
# Never use the historical test for selection; read its centers only for exclusion.
import math
from pathlib import Path
import numpy as np
import torch
from common import load_torch, save_torch, digest
from reference_v3.data import sample_orders, orders_to_positions, theta_profile, validate_bundle

PROFILES = ['uniform_low','uniform_medium','top_stable']
MS = [2,3,5,10]

def expected_kendall(theta):
    n = len(theta)+1
    return sum(float((torch.softmax(-float(t)*torch.arange(n-j,dtype=torch.float64),0)*
                      torch.arange(n-j,dtype=torch.float64)).sum()) for j,t in enumerate(theta))/(n*(n-1)/2)

def matched_uniform(n):
    target = expected_kendall(theta_profile(n,'top_stable'))
    lo, hi = 0., 64.
    for _ in range(90):
        mid = (lo+hi)/2
        if expected_kendall([mid]*(n-1)) > target: lo=mid
        else: hi=mid
    value=(lo+hi)/2
    return value, target, expected_kendall([value]*(n-1))

def condition(truth, theta, seed, maximum=10):
    orders=sample_orders(truth, maximum, theta, seed)
    return {'noisy_rankings':orders, 'noisy_positions':orders_to_positions(orders), 'theta_vector':theta}

def center_rows(payload):
    if isinstance(payload, torch.Tensor): yield from payload.tolist()
    elif isinstance(payload, dict) and 'splits' in payload:
        for section in payload['splits'].values(): yield from section['true_rankings'].tolist()
    elif isinstance(payload, dict) and 'true_rankings' in payload: yield from payload['true_rankings'].tolist()
    else: raise ValueError('Not a supported ranking-center dataset')

def historical_datasets(root):
    # Only known data filenames; never load model checkpoints as datasets.
    root=Path(root)
    names={'dataset.pt','ranking_dataset.pt','true_rankings.pt','development.pt','test.pt'}
    return sorted(p for p in root.rglob('*.pt') if p.name in names or p.name.startswith('generalized_mallows_v1_'))

def prepare_development(source, folder, noise_controls=False, noise_seed=2026110607):
    bundle=load_torch(source); validate_bundle(bundle)
    if set(bundle['config']['profiles'])!=set(PROFILES) or not set(MS)<=set(bundle['config']['m_values']):
        raise ValueError('Expected three standard profiles and m=2/3/5/10')
    sections={name:bundle['splits'][name] for name in ['train','validation']}
    # Drop historical test observations entirely. Preserve only exclusion centers.
    oldcenters=torch.cat([s['true_rankings'] for s in bundle['splits'].values()])
    save_torch(folder/'excluded_centers.pt', oldcenters)
    result={'format':'ablation_development_v1','num_items':bundle['config']['num_items'],
            'splits':sections, 'source_sha256':digest(source), 'train_controls':{}}
    if noise_controls:
        n=result['num_items']; scalar,target,actual=matched_uniform(n)
        truth=sections['train']['true_rankings']
        for mode,theta in [('uniform',[scalar]*(n-1)),('stage',theta_profile(n,'top_stable'))]:
            # Three independent noise groups in each case, matching baseline exposure.
            result['train_controls'][mode]={f'group_{i}':condition(truth,theta,noise_seed+i)
                                           for i in range(3)}
        result['noise_matching']={'uniform_theta':scalar,'stage_expected_kendall':target,
            'uniform_expected_kendall':actual,'noise_seed':noise_seed,'groups_per_center':3}
    save_torch(folder/'development.pt',result)
    return result

def make_schedule(count, groups, epochs, batch, seed):
    generator=torch.Generator().manual_seed(seed+1)
    rng=np.random.default_rng(seed+2)
    permutations=[]; choices=[]
    for _ in range(epochs):
        permutations.append(torch.randperm(count*groups,generator=generator))
        choices.append(torch.tensor(rng.integers(0,len(MS),size=math.ceil(count*groups/batch)),dtype=torch.long))
    return {'permutations':torch.stack(permutations),'m_indices':torch.stack(choices),'m_values':MS}

def fresh_test(count,n,seed,excluded,robustness=False):
    occupied={tuple(map(int,r)) for r in excluded if len(r)==n}
    if count<1 or count+len(occupied)>math.factorial(n): raise ValueError('Insufficient distinct centers')
    gen=torch.Generator().manual_seed(seed); rows=[]
    while len(rows)<count:
        row=tuple(torch.randperm(n,generator=gen).tolist())
        if row not in occupied: occupied.add(row); rows.append(row)
    truth=torch.tensor(rows,dtype=torch.long)
    profiles={p:theta_profile(n,p) for p in PROFILES}
    if robustness:
        for name,values in [('unseen_lower',(2.5,.8,.25)),('unseen_upper',(3.5,1.2,.5))]:
            profiles[name]=[values[0] if j<math.ceil(.2*n) else values[1] if j<math.ceil(.6*n) else values[2] for j in range(n-1)]
    return {'true_rankings':truth,'conditions':{p:condition(truth,t,seed+100003+i*10007)
            for i,(p,t) in enumerate(profiles.items())}}

