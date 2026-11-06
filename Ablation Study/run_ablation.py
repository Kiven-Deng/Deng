# New: independent CLI for controlled ablations; original project files stay unchanged.
# Commands default to preview. Training/evaluation requires explicit --execute.
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
import torch
from common import (PROJECT,HERE,digest,fingerprints,read_json,write_json,load_torch,
                    save_torch,load_plan,lock,device_for)
from model import VARIANTS,GROUPS,ModelConfig,shared_initialization,initialize
from data import (MS,PROFILES,prepare_development,make_schedule,historical_datasets,
                  center_rows,fresh_test)
from training import train_task
from evaluation import model_scores,baseline_cache
from analysis import summarize

DEFAULT_DATA=PROJECT/'outputs/v3_run_ad482c219a0944858edc0efdb7b0f574/data/dataset_20261106_121433_096724_9867d22d/dataset.pt'

def selected(args):
    variants=args.variants or GROUPS[args.group]
    if len(variants)!=len(set(variants)):raise ValueError('Repeated variant')
    return variants

def prepare(args):
    root=Path(args.study).resolve()
    if root.exists():raise ValueError('Study already exists; use a NEW study directory.')
    source=Path(args.dataset).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    variants=selected(args)
    if min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Invalid seeds')
    if min(args.epochs,args.batch_size,args.patience)<1 or args.lr<=0:raise ValueError('Invalid training parameters')
    root.mkdir(parents=True)
    dev=prepare_development(source,root,any(v.startswith('D') for v in variants))
    config=ModelConfig(hidden=args.hidden,heads=args.heads,encoder_layers=args.encoder_layers,decoder_layers=args.decoder_layers)
    # Check config before expensive runs.
    shared_initialization(config,args.seeds[0])
    for seed in args.seeds:
        save_torch(root/f'initialization/seed{seed}.pt',shared_initialization(config,seed))
        save_torch(root/f'schedules/seed{seed}.pt',make_schedule(len(dev['splits']['train']['true_rankings']),3,args.epochs,args.batch_size,seed))
    artifacts={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*.pt')}
    plan={'format':'ranking_ablation_v1','created':time.strftime('%Y-%m-%dT%H:%M:%S'),
          'source_dataset':str(source),'source_dataset_sha256':digest(source),
          'source_hashes':fingerprints(),'artifacts':artifacts,'variants':variants,'seeds':args.seeds,
          'model':config.to_dict(),'eval_m_values':MS,
          'training':{'epochs':args.epochs,'batch_size':args.batch_size,'patience':args.patience,'lr':args.lr},
          'protocol':'validation-only development; same shared initialization and sample schedule; no historical test evaluation',
          'variant_definitions':{v:VARIANTS[v] for v in variants},
          'noise_matching':dev.get('noise_matching')}
    write_json(root/'plan.json',plan)
    print(f'Prepared: {root}\nTasks: {len(variants)*len(args.seeds)}. No training started.',flush=True)

def run(args):
    root,plan=load_plan(args.study)
    variants=args.variants or plan['variants'];seeds=args.seeds or plan['seeds']
    if not set(variants)<=set(plan['variants']) or not set(seeds)<=set(plan['seeds']):raise ValueError('Tasks outside frozen plan')
    for v in variants:
        for s in seeds:
            state=root/'runs'/f'{v}_seed{s}'/'state.json'
            print(f'{v} seed={s}: '+(read_json(state)['status'] if state.exists() else 'pending'))
    if not args.execute:
        print('Preview only. Add --execute to train serially.');return
    with lock(root/'run.lock'):
        dev=load_torch(root/'development.pt');device=device_for(args.device)
        for v in variants:
            for s in seeds:train_task(root,plan,dev,v,s,device)
        print('Selected tasks complete. Run summarize after all planned tasks finish.',flush=True)

def baselines(args):
    root,plan=load_plan(args.study)
    if not args.execute:print('Preview: baseline evaluation on validation only. Add --execute.');return
    with lock(root/'run.lock'):
        dev=load_torch(root/'development.pt')
        baseline_cache(dev['splits']['validation'],plan['eval_m_values'],root/'baselines_validation',
                       plan['artifacts']['development.pt']+':validation')

def summarize_command(args):
    root,plan=load_plan(args.study)
    if args.phase=='final' and not (root/'final_complete.json').exists():raise RuntimeError('Final evaluation is incomplete')
    output=Path(args.output).resolve() if args.output else root/'reports'/f'{args.phase}_{time.strftime("%Y%m%d_%H%M%S")}'
    summarize(root,plan,args.phase,args.bootstrap,output)

def discover_exclusions(root,plan,extra):
    paths=set(historical_datasets(PROJECT))
    paths.update(Path(p).resolve() for p in extra)
    paths.add(root/'excluded_centers.pt'); paths.add(root/'development.pt')
    paths.add(Path(plan['source_dataset']))
    occupied=set(); provenance=[]
    for path in sorted(paths):
        if not path.exists():raise FileNotFoundError('Historical dataset missing: '+str(path))
        payload=load_torch(path)
        current=[tuple(map(int,r)) for r in center_rows(payload)]
        occupied.update(current)
        provenance.append({'path':str(path),'sha256':digest(path),'centers_read':len(current)})
    return occupied,provenance

def freeze_final(args):
    root,plan=load_plan(args.study)
    if (root/'final_protocol.json').exists():raise ValueError('Final protocol already frozen')
    variants=args.variants or plan['variants'];seeds=args.seeds or plan['seeds']
    if 'A0' not in variants:raise ValueError('Include A0 in final comparisons')
    if not set(variants)<=set(plan['variants']) or not set(seeds)<=set(plan['seeds']):
        raise ValueError('Final variants/seeds must already belong to this study; prepare a new 5-seed study if needed.')
    checkpoints=[]
    for v in variants:
        for seed in seeds:
            folder=root/'runs'/f'{v}_seed{seed}'
            state=read_json(folder/'state.json')
            if state['status']!='complete':raise ValueError('Incomplete training')
            for relative,expected in state['files'].items():
                if digest(folder/relative)!=expected:raise ValueError('Training artifact changed')
            best=read_json(folder/'best.json');path=folder/best['checkpoint']
            checkpoints.append({'variant':v,'seed':seed,'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    occupied,provenance=discover_exclusions(root,plan,args.exclude)
    dev=load_torch(root/'development.pt');n=dev['num_items']
    protocol={'format':'ablation_final_protocol_v1','variants':variants,'seeds':seeds,
        'samples':args.samples,'data_seed':args.data_seed,'num_items':n,'robustness':args.robustness,
        'eval_m_values':plan['eval_m_values'],'primary':'macro normalized Kendall over 3 standard profiles x 4 m',
        'checkpoints':checkpoints,'historical_datasets':provenance,
        'inference':'conditional on these training seeds; no hyperparameter selection on this new test',
        'plan_sha256':digest(root/'plan.json')}
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'run.lock'):
        save_torch(root/'final_excluded_centers.pt',{'true_rankings':torch.tensor(sorted(r for r in occupied if len(r)==n),dtype=torch.long)})
        protocol['exclusions_sha256']=digest(root/'final_excluded_centers.pt')
        write_json(root/'final_protocol.json',protocol)
    print('Final protocol frozen. No new test observations generated. Next: make-final-test.',flush=True)

def final_protocol(root,plan):
    protocol=read_json(root/'final_protocol.json')
    if protocol['plan_sha256']!=digest(root/'plan.json'):raise ValueError('Final plan mismatch')
    if digest(root/'final_excluded_centers.pt')!=protocol['exclusions_sha256']:raise ValueError('Exclusions changed')
    for row in protocol['checkpoints']:
        if digest(root/row['path'])!=row['sha256']:raise ValueError('Frozen checkpoint changed')
    return protocol

def make_final_test(args):
    root,plan=load_plan(args.study);protocol=final_protocol(root,plan)
    if not args.execute:print('Preview: generate sealed test using frozen protocol. Add --execute.');return
    with lock(root/'run.lock'):
        folder=root/'final_test'
        if folder.exists():raise ValueError('Final test directory already exists; do not regenerate after viewing.')
        folder.mkdir()
        excluded=load_torch(root/'final_excluded_centers.pt')['true_rankings'].tolist()
        section=fresh_test(protocol['samples'],protocol['num_items'],protocol['data_seed'],excluded,protocol['robustness'])
        overlaps=set(map(tuple,excluded)) & set(map(tuple,section['true_rankings'].tolist()))
        if overlaps:raise ValueError('Center overlap detected')
        save_torch(folder/'test.pt',{'format':'ablation_final_test_v1','splits':{'test':section}})
        write_json(folder/'manifest.json',{'protocol_sha256':digest(root/'final_protocol.json'),
            'dataset_sha256':digest(folder/'test.pt'),'samples':len(section['true_rankings']),
            'center_overlap':0,'historical_centers_checked':len(set(map(tuple,excluded))),
            'conditions':list(section['conditions']),'data_seed':protocol['data_seed']})
        print('New test saved; no model results computed.',flush=True)

def evaluate_final(args):
    root,plan=load_plan(args.study);protocol=final_protocol(root,plan)
    manifest=read_json(root/'final_test/manifest.json')
    if manifest['protocol_sha256']!=digest(root/'final_protocol.json') or manifest['dataset_sha256']!=digest(root/'final_test/test.pt'):
        raise ValueError('Final test fingerprint mismatch')
    if not args.execute:print(f'Preview: {len(protocol["checkpoints"])} frozen models and 6 baselines. Add --execute.');return
    with lock(root/'run.lock'):
        section=load_torch(root/'final_test/test.pt')['splits']['test'];device=device_for(args.device)
        baseline_cache(section,protocol['eval_m_values'],root/'baselines_final',manifest['dataset_sha256']+':test')
        for row in protocol['checkpoints']:
            output=root/'final_results'/f'{row["variant"]}_seed{row["seed"]}'
            done=output/'complete.json'
            if done.exists():
                state=read_json(done)
                if state['checkpoint_sha256']!=row['sha256'] or state['dataset_sha256']!=manifest['dataset_sha256']:
                    raise ValueError('Final result mismatch')
                if digest(output/'scores.npz')!=state['scores_sha256']:raise ValueError('Final scores changed')
                continue
            if output.exists():raise RuntimeError('Incomplete final result; inspect before retry: '+str(output))
            saved=load_torch(root/row['path'])
            model=initialize(ModelConfig(**saved['model_config']),row['variant'],
                             shared_initialization(ModelConfig(**saved['model_config']),row['seed'])).to(device)
            model.load_state_dict(saved['model_state_dict'])
            metadata={'variant':row['variant'],'seed':row['seed'],'split':'test',
                'best_epoch':saved['epoch'],'dataset_sha256':manifest['dataset_sha256'],
                'checkpoint_sha256':row['sha256'],'plan_sha256':digest(root/'plan.json')}
            model_scores(model,section,protocol['eval_m_values'],device,plan['training']['batch_size'],output,metadata)
            write_json(done,dict(metadata,scores_sha256=digest(output/'scores.npz')))
            print('Final evaluated: '+output.name,flush=True)
        if not (root/'final_complete.json').exists():
            write_json(root/'final_complete.json',{'status':'complete','protocol_sha256':digest(root/'final_protocol.json'),
                                                 'models':len(protocol['checkpoints'])})

def parser():
    p=argparse.ArgumentParser(description=__doc__ or 'Standalone ranking recovery ablation study')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--study',required=True);q.add_argument('--dataset',default=str(DEFAULT_DATA))
    q.add_argument('--group',choices=GROUPS,default='core');q.add_argument('--variants',nargs='+',choices=VARIANTS)
    q.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);q.add_argument('--epochs',type=int,default=50)
    q.add_argument('--patience',type=int,default=10);q.add_argument('--batch-size',type=int,default=32);q.add_argument('--lr',type=float,default=1e-4)
    q.add_argument('--hidden',type=int,default=64);q.add_argument('--heads',type=int,default=4)
    q.add_argument('--encoder-layers',type=int,default=2);q.add_argument('--decoder-layers',type=int,default=2);q.set_defaults(func=prepare)
    q=sub.add_parser('run');q.add_argument('--study',required=True);q.add_argument('--variants',nargs='+',choices=VARIANTS)
    q.add_argument('--seeds',nargs='+',type=int);q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');q.add_argument('--execute',action='store_true');q.set_defaults(func=run)
    q=sub.add_parser('baselines');q.add_argument('--study',required=True);q.add_argument('--execute',action='store_true');q.set_defaults(func=baselines)
    q=sub.add_parser('summarize');q.add_argument('--study',required=True);q.add_argument('--phase',choices=['validation','final'],default='validation')
    q.add_argument('--bootstrap',type=int,default=2000);q.add_argument('--output');q.set_defaults(func=summarize_command)
    q=sub.add_parser('freeze-final');q.add_argument('--study',required=True);q.add_argument('--variants',nargs='+',choices=VARIANTS)
    q.add_argument('--seeds',nargs='+',type=int);q.add_argument('--samples',type=int,default=1000)
    q.add_argument('--data-seed',type=int,default=2026111601);q.add_argument('--exclude',nargs='*',default=[])
    q.add_argument('--robustness',action='store_true');q.set_defaults(func=freeze_final)
    for name,fn in [('make-final-test',make_final_test),('evaluate-final',evaluate_final)]:
        q=sub.add_parser(name);q.add_argument('--study',required=True);q.add_argument('--execute',action='store_true')
        if name=='evaluate-final':q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        q.set_defaults(func=fn)
    q=sub.add_parser('list');q.set_defaults(func=lambda a: print(json.dumps(VARIANTS,indent=2,ensure_ascii=False)))
    return p

if __name__=='__main__':
    torch.set_num_threads(1)
    args=parser().parse_args()
    if getattr(args,'bootstrap',1)<1 or getattr(args,'samples',1)<1:raise ValueError('Count must be positive')
    args.func(args)

