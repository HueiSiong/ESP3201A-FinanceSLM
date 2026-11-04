"""Portable verification, inference, evaluation and controlled training. Outputs go to work/."""
from local_runtime import configure
configure()
from pathlib import Path
import argparse,contextlib,gc,hashlib,json,math,random,sys,time,warnings
import numpy as np
import torch
from tokenizers import Tokenizer
from final_model import FinanceSLM
from artifacts import sha,verify as verify_assets,manifest as asset_manifest
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/final'
CONFIGS={label:dict(vocab_size=8000,dim=d,n_layers=l,n_heads=h,max_seq_len=256) for label,d,l,h in [('15m',384,7,6),('30m',512,8,8),('50m',640,9,10)]}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def output_path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT/'work'):raise ValueError('Outputs must be under work/ to preserve frozen evidence.')
    return p

def setup(device):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    warnings.filterwarnings('ignore',message='.*Torch was not compiled with flash attention.*')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Install the CUDA wheel or use --device cpu.')
    return torch.device(device)

def load_model(path,device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('tokenizer_sha256') and state['tokenizer_sha256']!=sha(DATA/'finance_tokenizer.json'):raise ValueError('Tokenizer mismatch')
    model=FinanceSLM(**state['model_config']).to(device);model.load_state_dict(state['model_state_dict']);model.eval()
    return model,state

@torch.inference_mode()
def loss_on_docs(model,docs,device,limit=None):
    model.eval();rows=[];total=0.;count=0;start=time.perf_counter()
    for doc in docs:
        ids=doc['ids'];subtotal=0.;n=0
        for i in range(1,len(ids),256):
            if limit is not None and count>=limit:break
            end=min(i+256,len(ids))
            if limit is not None:end=min(end,i+limit-count)
            x=torch.tensor([ids[i-1:end-1]],device=device);y=torch.tensor([ids[i:end]],device=device)
            value=model(x,y)[1].item()*y.numel();subtotal+=value;n+=y.numel();count+=y.numel()
        if n:rows.append(dict(ticker=doc['ticker'],year=doc['year'],nll_sum=subtotal,tokens=n,loss=subtotal/n));total+=subtotal
        if limit is not None and count>=limit:break
    if not count:raise ValueError('No evaluation tokens')
    return dict(loss=total/count,perplexity=math.exp(total/count),tokens=count,seconds=time.perf_counter()-start,documents=rows,partial=limit is not None)

def smoke(args):
    device=setup(args.device);fingerprints=read(ROOT/'checksums.json')
    for name,expected in fingerprints.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen input/evidence changed: '+name)
    verify_assets(['demo'])
    from app import DemoModel
    runtime=DemoModel(device=str(device));text,a=runtime.generate("The company's revenue increased",max_tokens=24);text2,b=runtime.generate("The company's revenue increased",max_tokens=24)
    assert text==text2 and a['generated_tokens']>0
    assert runtime.parameters==49737600 and runtime.model.max_seq_len==256
    tokens=runtime.tokenizer.encode('Revenue grew by 12.5% to $15 million.').ids
    assert runtime.tokenizer.decode(tokens)=='Revenue grew by 12.5% to $15 million.'
    score=loss_on_docs(runtime.model,read(DATA/'test_documents.json'),device,1024)
    # Compare logits directly with a numerical fixture, not stochastic text across different devices.
    fixture=read(ROOT/'configs/reference_logits.json')
    with torch.inference_mode():logits=runtime.model(torch.tensor([fixture['input_ids']],device=device))[0][0,-1]
    errors=[abs(logits[i].item()-value) for i,value in zip(fixture['indices'],fixture['values'])]
    assert max(errors)<.02, 'Reference logits differ beyond tolerance'
    # Ensure this process did not import packages from the old project environment.
    forbidden=[p for p in sys.path if 'venv/Lib/site-packages' in p.replace('\\','/') and not Path(p).is_relative_to(Path(sys.prefix))]
    assert not forbidden,forbidden
    result=dict(passed=True,device=str(device),torch=torch.__version__,python=sys.version,python_executable=sys.executable,torch_file=torch.__file__,frozen_files_verified=len(fingerprints),parameters=runtime.parameters,tokenizer_round_trip=True,fixed_seed_repeatable=True,reference_logits_max_error=max(errors),partial_test=score,generation=text,note='Partial test perplexity is a smoke check, not the published full-test result. CPU/GPU sampled text need not match.')
    write(output_path(args.output),result);print(json.dumps(result,indent=2))

def evaluate(args):
    device=setup(args.device);model,state=load_model((ROOT/args.checkpoint).resolve(),device)
    result=loss_on_docs(model,read(DATA/f'{args.split}_documents.json'),device,args.max_tokens)
    result.update(checkpoint=str(args.checkpoint),run=state.get('run'),split=args.split,device=str(device),torch=torch.__version__)
    expected_files={f['path']:f['sha256'] for group in asset_manifest()['groups'].values() for f in group['files']}
    checkpoint=(ROOT/args.checkpoint).resolve()
    relative=checkpoint.relative_to(ROOT).as_posix() if checkpoint.is_relative_to(ROOT) else ''
    frozen=relative in expected_files and sha(checkpoint)==expected_files[relative]
    if args.max_tokens is None and state.get('run') and frozen:
        expected=read(ROOT/'results/final/runs'/state['run']/'evaluation.json')[args.split]['perplexity']
        result.update(recorded_perplexity=expected,relative_difference=abs(result['perplexity']-expected)/expected)
        if result['relative_difference']>.01:raise RuntimeError('Full-test perplexity differs by more than 1%; investigate environment/data.')
    write(output_path(args.output),result);print(json.dumps({k:v for k,v in result.items() if k!='documents'},indent=2))

def train(args):
    device=setup(args.device);folder=output_path(args.output)
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('Training output is not empty; choose a new work/ directory.')
    folder.mkdir(parents=True,exist_ok=True)
    if args.steps<1 or args.batch_size<1:raise ValueError('Positive steps/batch size required')
    canonical=device.type=='cuda' and args.steps==3072 and args.batch_size==16
    if device.type=='cuda':
        if not torch.cuda.is_bf16_supported():raise RuntimeError('Original recipe requires BF16-capable GPU; use CPU only for short functional checks.')
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    config=CONFIGS[args.model];model=FinanceSLM(**config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,**({'fused':True} if device.type=='cuda' else {}))
    rng=np.random.RandomState(args.seed);data=np.load(DATA/f'train_{args.budget}.npy')
    probe=[dict(d,ids=d['ids'][:2049]) for d in read(DATA/'validation_documents.json')]
    best=float('inf');best_step=0;started=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with (folder/'history.jsonl').open('w',encoding='utf-8',buffering=1) as log:
      for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        positions=rng.randint(0,len(data)-256,size=args.batch_size)
        indices=positions[:,None]+np.arange(257)[None,:]
        values=torch.from_numpy(data[indices].astype(np.int64)).to(device)
        progress=max(0,(step-100)/max(1,args.steps-100));lr=3e-4*min(1,step/100)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext():
            _,loss=model(values[:,:-1].contiguous(),values[:,1:].contiguous())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        optimizer.step();entry=dict(step=step,loss=loss.item(),gradient_norm=float(grad),lr=lr,elapsed_seconds=time.perf_counter()-started)
        if step==128 or step%256==0 or step==args.steps:
            val=loss_on_docs(model,probe,device, None if canonical else 1024)['loss'];entry['validation_probe_loss']=val
            if val<best:
                best=val;best_step=step
                torch.save(dict(model_config=config,model_state_dict=model.state_dict(),training_steps=step,architecture='final_model',tokenizer_path='data/final/finance_tokenizer.json',tokenizer_sha256=sha(DATA/'finance_tokenizer.json'),run=f'{args.model}_{args.budget}_seed{args.seed}',seed=args.seed,budget=args.budget,training_tokens_processed=step*args.batch_size*256,validation_probe_loss=best),folder/'best.pt')
            print(json.dumps(entry),flush=True)
        log.write(json.dumps(entry)+'\n')
    torch.save(dict(model_config=config,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),training_steps=args.steps,architecture='final_model',tokenizer_path='data/final/finance_tokenizer.json'),folder/'last.pt')
    summary=dict(model=args.model,budget=args.budget,seed=args.seed,steps=args.steps,batch_size=args.batch_size,parameters=sum(p.numel() for p in model.parameters()),selected_step=best_step,best_validation_probe_loss=best,seconds=time.perf_counter()-started,canonical_recipe=canonical,device=str(device),training_tokens_processed=args.steps*args.batch_size*256,training_stream_tokens=len(data),dataset_sha256=sha(DATA/f'train_{args.budget}.npy'),tokenizer_sha256=sha(DATA/'finance_tokenizer.json'))
    write(folder/'summary.json',summary);print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('smoke');s.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');s.add_argument('--output',default='work/smoke.json')
    e=sub.add_parser('evaluate');e.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');e.add_argument('--checkpoint',default='checkpoints/final/financeslm_final.pt');e.add_argument('--split',choices=['validation','test'],default='test');e.add_argument('--max-tokens',type=int);e.add_argument('--output',default='work/evaluation.json')
    t=sub.add_parser('train');t.add_argument('--device',choices=['cpu','cuda'],default='cuda');t.add_argument('--model',choices=CONFIGS,default='15m');t.add_argument('--budget',choices=['small','medium','large'],default='large');t.add_argument('--seed',type=int,default=42);t.add_argument('--steps',type=int,default=3072);t.add_argument('--batch-size',type=int,default=16);t.add_argument('--output',default='work/training')
    args=p.parse_args()
    if getattr(args,'max_tokens',None) is not None and args.max_tokens<1:p.error('--max-tokens must be positive')
    {'smoke':smoke,'evaluate':evaluate,'train':train}[args.command](args)

if __name__=='__main__':main()
