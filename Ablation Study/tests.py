# New: CPU-only verification of all gates, initialization, causal properties and CLI pipeline.
# No production training or historical test evaluation is performed by these checks.
import argparse, copy, hashlib, io, json, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import HERE, PROJECT, load_torch, save_torch, read_json, digest, write_json
from model import ModelConfig,AblationTransformer,VARIANTS,shared_initialization,initialize
from reference_v3.model import RecoveryTransformer
from reference_v3.data import build_bundle,validate_bundle
from data import matched_uniform,expected_kendall,make_schedule
from analysis import crossed_interval
import run_ablation as cli

def structural():
    torch.set_num_threads(1)
    config=ModelConfig(hidden=16,heads=2,encoder_layers=1,decoder_layers=1)
    bundle=build_bundle(6,(24,8,8),(2,3,5,10),('uniform_low','uniform_medium','top_stable'),903)
    validate_bundle(bundle)
    truth=bundle['splits']['train']['true_rankings'][:6]
    x=bundle['splits']['train']['conditions']['top_stable']['noisy_positions'][:6]
    shared=shared_initialization(config,42)
    reference=RecoveryTransformer(config)
    reference.load_state_dict({k:shared[k] for k in reference.state_dict()})
    model=initialize(config,'A0',shared)
    assert torch.equal(reference(x,truth),model(x,truth))
    assert torch.equal(reference.loss(x,truth),model.loss(x,truth))
    assert torch.equal(reference.predict(x),model.predict(x))
    for variant in VARIANTS:
        model=initialize(config,variant,shared).eval()
        for k,v in model.state_dict().items():assert torch.equal(v,shared[k])
        logits=model(x,truth)
        assert torch.allclose(logits,model(x[:,:,torch.arange(9,-1,-1)],truth),atol=3e-6,rtol=1e-5),variant
        assert torch.allclose(logits[:1],model(x[:1],truth[:1]),atol=3e-6,rtol=1e-5),variant
        model.train()
        assert torch.allclose(logits,model(x,truth),atol=3e-6,rtol=1e-5),variant
        model.eval()
        perm=torch.tensor([2,5,0,4,1,3]);inverse=perm.argsort()
        assert torch.allclose(model(x[:,perm],inverse[truth]),logits[:,:,perm],atol=3e-6,rtol=1e-5),variant
        changed=truth.clone();changed[:,3:]=changed[:,3:].flip(1)
        assert torch.allclose(logits[:,:4],model(x,changed)[:,:4],atol=3e-6,rtol=1e-5),variant
        memory,ranks,summary=model.encode(x)
        for t in range(6):
            assert torch.allclose(logits[:,t],model.prefix_logits(memory,ranks,summary,truth[:,:t])[:,-1],atol=3e-6,rtol=1e-5),variant
        for m in [2,3,5,10]:
            pred=model.predict(x[:,:,:m]);assert torch.equal(pred.sort(1).values,torch.arange(6).expand_as(pred)),variant
        model.zero_grad();model.loss(x,truth).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        other=initialize(config,variant,shared);other.load_state_dict(torch.load(stream,weights_only=True))
        assert torch.equal(model.predict(x),other.predict(x)),variant
        # Changing removed branches cannot change outputs.
        before=model(x,truth).detach().clone()
        with torch.no_grad():
            for name,p in model.named_parameters():
                disabled=((variant in ['A1','A2'] and name.startswith('rank_skip.')) or
                          (variant=='A2' and name.startswith('summary_map.')) or
                          (variant=='A4' and name.startswith('observation_map.')) or
                          (variant=='A5' and name.startswith('encoders.') and ('.attention.' in name or '.norm1.' in name)))
                if disabled:p.add_(123.)
        assert torch.equal(before,model(x,truth)),variant
        if variant=='A3':
            shifted=x.clone();shifted[:,:,0]+=.125;shifted[:,:,1]-=.125
            assert torch.equal(model(x,truth),model(shifted,truth))
    scalar,target,actual=matched_uniform(50);assert abs(target-actual)<1e-12 and scalar>0
    a=make_schedule(24,3,3,8,42);b=make_schedule(24,3,3,8,42)
    assert torch.equal(a['permutations'],b['permutations']) and torch.equal(a['m_indices'],b['m_indices'])
    constant=np.ones((3,7,2))*.03
    interval=crossed_interval(constant,50)
    assert np.allclose(interval['ci95'],.03)
    print('PASS: exact A0, 13 variants, shared initialization, gates, causal/invariance checks, schedules, noise matching, bootstrap.',flush=True)
    return config,bundle

def microfit(config,bundle):
    truth=bundle['splits']['train']['true_rankings'][:12]
    x=torch.empty_like(truth).scatter_(1,truth,torch.arange(1,7).expand_as(truth)).float()[:,:,None].repeat(1,1,2)
    result={}
    for variant in ['A0','A1','A2','A3','A4','A5','P10','P01','P11']:
        model=initialize(config,variant,shared_initialization(config,42))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
        passed=False
        for step in range(1,301):
            model.train();optimizer.zero_grad();loss=model.loss(x,truth);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%25==0:
                model.eval();exact=float((model.predict(x)==truth).all(1).float().mean())
                if exact==1 and float(loss.detach())<.15:passed=True;break
        if not passed:raise AssertionError(f'{variant}: micro-fit failed at step {step}, CE={float(loss.detach())}, exact={exact}')
        result[variant]={'steps':step,'ce':float(loss.detach()),'exact':exact}
        print(f'PASS micro-fit {variant}: step={step}, exact={exact:.3f}',flush=True)
    return result

def pipeline(bundle):
    with tempfile.TemporaryDirectory(prefix='ranking_ablation_check_') as directory:
        base=Path(directory);source=base/'dataset.pt';save_torch(source,bundle)
        root=base/'study'
        def invoke(*args):
            parsed=cli.parser().parse_args(list(args));parsed.func(parsed)
        invoke('prepare','--study',str(root),'--dataset',str(source),'--group','all','--seeds','7',
               '--epochs','1','--batch-size','24','--hidden','16','--heads','2','--encoder-layers','1','--decoder-layers','1')
        invoke('run','--study',str(root),'--device','cpu')
        assert not (root/'runs').exists(),'Preview launched training'
        invoke('run','--study',str(root),'--device','cpu','--execute')
        before={p:digest(p) for p in (root/'runs').rglob('*.pt')}
        invoke('run','--study',str(root),'--device','cpu','--execute')
        assert before=={p:digest(p) for p in (root/'runs').rglob('*.pt')},'Completed run was rewritten'
        invoke('baselines','--study',str(root),'--execute')
        invoke('summarize','--study',str(root),'--bootstrap','50','--output',str(base/'dev_report'))
        for path in (root/'runs').rglob('*.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():json.loads(line)
        invoke('freeze-final','--study',str(root),'--variants','A0','A1','--seeds','7','--samples','12','--data-seed','918','--robustness')
        invoke('make-final-test','--study',str(root),'--execute')
        invoke('evaluate-final','--study',str(root),'--device','cpu','--execute')
        invoke('summarize','--study',str(root),'--phase','final','--bootstrap','50','--output',str(base/'final_report'))
        assert read_json(root/'final_test/manifest.json')['center_overlap']==0
        # Fingerprints must reject later mutation of a frozen data artifact.
        path=root/'development.pt'
        with path.open('ab') as f:f.write(b'tampered')
        try:invoke('run','--study',str(root))
        except ValueError:pass
        else:raise AssertionError('Changed artifact was accepted')
    print('PASS: all 13 variants train/evaluate, resumable queue, baseline cache, summaries, frozen fresh test and integrity rejection.',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-microfit',action='store_true')
    parser.add_argument('--skip-pipeline',action='store_true')
    parser.add_argument('--report')
    args=parser.parse_args();torch.set_num_threads(1);started=time.perf_counter()
    original={p:digest(p) for p in (PROJECT/'ranking_recovery_v3').glob('*.py')}
    config,bundle=structural()
    fits={} if args.skip_microfit else microfit(config,bundle)
    if not args.skip_pipeline:pipeline(bundle)
    assert original=={p:digest(p) for p in (PROJECT/'ranking_recovery_v3').glob('*.py')}
    report={'status':'passed','seconds':time.perf_counter()-started,'variants':list(VARIANTS),
            'microfit':fits,'pipeline_checked':not args.skip_pipeline,'device':'cpu','original_v3_unchanged':True}
    if args.report:write_json(Path(args.report).resolve(),report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

