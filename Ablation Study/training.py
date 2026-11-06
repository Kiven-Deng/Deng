# New: immutable shared initialization/schedules; validation-only model selection.
# A completed task is reused; failed/interrupted tasks are never silently restarted.
import json, random, time, traceback
from pathlib import Path
import numpy as np
import torch
from common import load_torch, save_torch, write_json, read_json, digest, seed_all, synchronize
from model import VARIANTS, initialize, ModelConfig
from evaluation import model_scores

def train_task(root,plan,development,variant,seed,device):
    folder=root/'runs'/f'{variant}_seed{seed}'
    if folder.exists():
        state=read_json(folder/'state.json') if (folder/'state.json').exists() else {'status':'incomplete'}
        if state['status']=='complete':
            if state['plan_sha256']!=digest(root/'plan.json'):raise ValueError('Task plan changed')
            for relative,expected in state['files'].items():
                if digest(folder/relative)!=expected:raise ValueError('Completed task file changed')
            return folder
        raise RuntimeError(f'{folder}: {state["status"]}; inspect state/logs. Use a new study for retry.')
    folder.mkdir(parents=True)
    write_json(folder/'state.json',{'status':'running','variant':variant,'seed':seed})
    try:
        seed_all(seed);h=plan['training'];flags=VARIANTS[variant]
        config=ModelConfig(**plan['model'])
        shared=load_torch(root/f'initialization/seed{seed}.pt')
        model=initialize(config,variant,shared).to(device)
        active=model.active_parameter_count(); total=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=h['lr'],weight_decay=0)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,min_lr=1e-6)
        train=development['splits']['train'];truth=train['true_rankings']
        conditions=development['train_controls'][flags['data']] if 'data' in flags else train['conditions']
        profiles=list(conditions); count=len(truth)
        schedule=load_torch(root/f'schedules/seed{seed}.pt')
        best=(float('inf'),float('inf'));stale=0;best_path=None;steps=0
        started=time.perf_counter()
        for epoch in range(h['epochs']):
            model.train(); loss_sum=0.;samples=0
            permutation=schedule['permutations'][epoch]
            for bi,start in enumerate(range(0,len(permutation),h['batch_size'])):
                flat=permutation[start:start+h['batch_size']];ids=flat%count;groups=flat//count
                m=flags.get('train_m',[schedule['m_values'][int(schedule['m_indices'][epoch,bi])]])[0]
                x=torch.stack([conditions[profiles[int(g)]]['noisy_positions'][int(i),:,:m]
                               for i,g in zip(ids,groups)]).to(device)
                target=truth[ids].to(device)
                optimizer.zero_grad(set_to_none=True);loss=model.loss(x,target)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();steps+=1;samples+=len(ids);loss_sum+=float(loss.detach())*len(ids)
                if (bi+1)%50==0:print(f'{variant}/{seed} epoch={epoch+1} batch={bi+1} CE={loss_sum/samples:.5f}',flush=True)
            _,rows=model_scores(model,development['splits']['validation'],plan['eval_m_values'],device,h['batch_size'])
            key=(float(np.mean([r['kendall'] for r in rows])),float(np.mean([r['rbo'] for r in rows])))
            scheduler.step(key[0]); improved=key<best
            if improved:
                best=key;stale=0;best_path=folder/f'best_epoch_{epoch+1:03d}.pt'
                save_torch(best_path,{'format':'ablation_checkpoint_v1','variant':variant,'seed':seed,
                    'model_config':plan['model'],'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'epoch':epoch+1,'steps_at_checkpoint':steps,'validation_key':list(key),
                    'plan_sha256':digest(root/'plan.json'),'development_sha256':plan['artifacts']['development.pt']})
            else:stale+=1
            record={'epoch':epoch+1,'train_ce':loss_sum/samples,'macro_kendall':key[0],'macro_rbo':key[1],
                    'validation':rows,'lr':optimizer.param_groups[0]['lr'],'improved':improved,
                    'steps':steps,'elapsed_seconds':time.perf_counter()-started}
            with (folder/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(f'{variant}/{seed} epoch={epoch+1} CE={record["train_ce"]:.5f} valK={key[0]:.6f} valRBO={key[1]:.6f}',flush=True)
            if stale>=h['patience']:break
        synchronize(device);train_seconds=time.perf_counter()-started
        checkpoint=load_torch(best_path);model.load_state_dict(checkpoint['model_state_dict'])
        metadata={'variant':variant,'seed':seed,'split':'validation','purpose':'exploratory_development',
                  'best_epoch':checkpoint['epoch'],'steps':steps,'train_seconds':train_seconds,
                  'parameter_count':total,'active_parameter_count':active,
                  'dataset_sha256':plan['artifacts']['development.pt'],'plan_sha256':digest(root/'plan.json')}
        model_scores(model,development['splits']['validation'],plan['eval_m_values'],device,h['batch_size'],folder/'validation',metadata)
        write_json(folder/'best.json',{'checkpoint':best_path.name,'validation_key':list(best),
                                     'epochs_completed':epoch+1,'metadata':metadata})
        paths=[best_path,folder/'best.json',folder/'history.jsonl',folder/'validation/scores.npz',
               folder/'validation/evaluation.json',folder/'validation/per_sample.jsonl']
        write_json(folder/'state.json',{'status':'complete','variant':variant,'seed':seed,
            'plan_sha256':digest(root/'plan.json'),'files':{p.relative_to(folder).as_posix():digest(p) for p in paths}},replace=True)
        return folder
    except BaseException:
        write_json(folder/'state.json',{'status':'error','traceback':traceback.format_exc()},replace=True)
        raise

