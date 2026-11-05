"""Portable side-by-side inference and full-test comparison; never trains."""
from pathlib import Path
import argparse, gc, hashlib, html, json, logging, math, os, sys, threading, time, uuid, warnings
from datetime import datetime, timezone
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
from app import DemoModel
import gradio as gr
import torch

DEFAULT_PROMPT = "The company's revenue increased"
BASE = '15m_large_seed42'
PRESETS = ['Model size', 'Training data', 'Temperature', 'Top-K', 'Generation seed', 'Maximum new tokens', 'Training seed']
PRESET_LABELS = {
    'Model size':'Model parameters: 15,464,064 vs 29,401,600 vs 49,737,600',
    'Training data':'Training text: 357,288 vs 706,370 vs 1,048,452 tokens',
    'Temperature':'Temperature: 0.3 vs 0.8 vs 1.2 (same trained model)',
    'Top-K':'Top-K candidates: 10 vs 40 vs 100 (same trained model)',
    'Generation seed':'Generation random seed: 42 vs 123 vs 7 (same trained model)',
    'Maximum new tokens':'Maximum new tokens: 40 vs 80 vs 120 (same trained model)',
    'Training seed':'Training random seed: 42 vs 123 (only two trained seeds)',
}
OUT = ROOT / 'work/comparison'

def verify_frozen_inputs():
    """Reject changed data/evidence before presenting historical comparisons."""
    for name, expected in read(ROOT/'checksums.json').items():
        path = ROOT/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Frozen input/evidence differs: '+name+'. Restore it from the repository.')

def checked_checkpoint(run):
    relative = f'checkpoints/final/{run}.pt'
    entries = {f['path']: f for group in read(ROOT/'assets.json')['groups'].values() for f in group['files']}
    if relative not in entries:
        raise ValueError('Unknown published checkpoint: '+run)
    path = ROOT/relative
    if not path.is_file():
        group = run.split('_')[0]
        raise ValueError(f'Missing {relative}. In the project terminal, use your virtual environment Python: '
                         f'python artifacts.py fetch --repo HueiSiong/FinanceSLM --group {group}. '
                         'Install groups 15m, 30m and 50m for every comparison. See docs/LIVE_COMPARISON.md.')
    entry = entries[relative]
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError('Checkpoint checksum differs: '+relative+'. Move that file aside and reinstall its release group.')
    return path, entry['sha256']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan(preset):
    """Each preset varies exactly one configured factor, retaining the same input prompt."""
    common = dict(run=BASE, max_tokens=80, temperature=.8, top_k=40, seed=42)
    variants = {
        'Model size': [('15.46M', {'run':'15m_large_seed42'}), ('29.40M', {'run':'30m_large_seed42'}), ('49.74M', {'run':'50m_large_seed42'})],
        'Training data': [('357,288 training tokens', {'run':'15m_small_seed42'}), ('706,370 training tokens', {'run':'15m_medium_seed42'}), ('1,048,452 training tokens', {})],
        'Temperature': [('Temperature 0.3', {'temperature':.3}), ('Temperature 0.8', {}), ('Temperature 1.2', {'temperature':1.2})],
        'Top-K': [('Top-K 10', {'top_k':10}), ('Top-K 40', {}), ('Top-K 100', {'top_k':100})],
        'Generation seed': [('Generation seed 42', {}), ('Generation seed 123', {'seed':123}), ('Generation seed 7', {'seed':7})],
        'Maximum new tokens': [('40-token limit', {'max_tokens':40}), ('80-token limit', {}), ('120-token limit', {'max_tokens':120})],
        'Training seed': [('Training seed 42', {}), ('Training seed 123', {'run':'15m_large_seed123'})],
    }
    if preset not in variants:
        raise ValueError('Choose a known comparison.')
    return [dict(common, **changes, label=label) for label, changes in variants[preset]]

def description(preset):
    fixed = {
        'Model size':'Large training pool; training seed 42. Only architecture size changes (width, layers and heads together).',
        'Training data':'15.46M architecture; training seed 42. Only the nested training pool changes; report composition changes too.',
        'Training seed':'15.46M architecture; large training pool. Only training seed changes; generation seed stays 42.',
    }
    return (fixed.get(preset, 'The same 15.46M / large-data / training-seed-42 checkpoint in every column.')
            + ' Common starting words; defaults: 80 new tokens, temperature 0.8, top-K 40, generation seed 42. Only the named factor differs.'
            + ' Model/data/training-seed comparisons use the same validation-selection rule, not necessarily the same selected training step.')

class Comparison:
    def __init__(self, device='auto'):
        verify_frozen_inputs()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        if self.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Start with --device cpu.')
        self.lock = threading.Lock()
        self.cache = {}
        self.hashes = {}
        self.evidence = {p.parent.name:read(p) for p in (ROOT/'results/final/runs').glob('*/evaluation.json')}
        self.tokenizer_hash = sha(ROOT/'data/final/finance_tokenizer.json')
        self.test_hash = sha(ROOT/'data/final/test_documents.json')

    def runtime(self, run):
        if run not in self.cache:
            if len(self.cache)>=3:
                self.cache.clear();gc.collect()
                if self.device=='cuda':torch.cuda.empty_cache()
            path, self.hashes[run] = checked_checkpoint(run)
            self.cache[run] = DemoModel(device=self.device, checkpoint_path=path, tokenizer_path=ROOT/'data/final/finance_tokenizer.json')
        return self.cache[run]

    def save(self, result, prefix):
        OUT.mkdir(parents=True, exist_ok=True)
        path=OUT/f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.json'
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        return path

    def generate(self, preset, prompt):
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Enter a starting sentence.')
        rows=[];started=time.perf_counter()
        with self.lock:
            for item in plan(preset):
                runtime=self.runtime(item['run'])
                continuation,record=runtime.generate(prompt,**{k:item[k] for k in ['max_tokens','temperature','top_k','seed']})
                ev=self.evidence[item['run']]
                rows.append(dict(record, label=item['label'], run=item['run'], parameters=ev['parameters'], training_seed=ev['seed'], training_tokens=ev['training_stream_tokens'], selected_step=ev['selected_step'], checkpoint_sha256=self.hashes[item['run']], recorded_test_perplexity=ev['test']['perplexity']))
        result=dict(kind='fresh_generation',timestamp_utc=datetime.now(timezone.utc).isoformat(),preset=preset,prompt=prompt,device=self.device,torch=torch.__version__,tokenizer_sha256=self.tokenizer_hash,controls=description(preset),total_seconds_including_load=time.perf_counter()-started,rows=rows)
        path=self.save(result,'generation')
        return result,path

    @torch.inference_mode()
    def evaluate(self, preset):
        rows=[];docs=read(ROOT/'data/final/test_documents.json');started=time.perf_counter()
        with self.lock:
            for run in dict.fromkeys(x['run'] for x in plan(preset)):
                runtime=self.runtime(run);total=0.;count=0;per_doc=[];t=time.perf_counter()
                for doc in docs:
                    ids=doc['ids'];subtotal=0.;n=0
                    for i in range(1,len(ids),256):
                        end=min(i+256,len(ids))
                        x=torch.tensor([ids[i-1:end-1]],device=self.device)
                        y=torch.tensor([ids[i:end]],device=self.device)
                        subtotal+=runtime.model(x,y)[1].item()*y.numel();n+=y.numel()
                    total+=subtotal;count+=n
                    per_doc.append(dict(ticker=doc['ticker'],year=doc['year'],tokens=n,nll_sum=subtotal))
                ppl=math.exp(total/count);expected=self.evidence[run]['test']['perplexity']
                assert count==128685
                rows.append(dict(run=run,perplexity=ppl,mean_nll=total/count,nll_sum=total,recorded_perplexity=expected,relative_difference=abs(ppl-expected)/expected,within_one_percent=abs(ppl-expected)/expected<.01,tokens=count,seconds=time.perf_counter()-t,checkpoint_sha256=self.hashes[run],documents=per_doc))
        result=dict(kind='fresh_full_test_evaluation',partial=False,device=self.device,torch=torch.__version__,timestamp_utc=datetime.now(timezone.utc).isoformat(),test_sha256=self.test_hash,preset=preset,rows=rows,total_seconds_including_load=time.perf_counter()-started,note='Full held-out text, not the typed generation prompt. FP32; TF32 off; document boundaries retained. Sampling settings do not affect teacher-forced perplexity.')
        return result,self.save(result,'evaluation')

@lru_cache(maxsize=3)
def inventory(budget):
    path=ROOT/'data/final'/f'train_{budget}.jsonl'
    return [dict(ticker=r['ticker'],year=r['year']) for line in path.read_text(encoding='utf-8').splitlines() if line.strip() for r in [json.loads(line)]]

def configuration(preset, evidence):
    """Preview every actual value before running; no inference or sampling happens here."""
    variants=plan(preset);manifest=read(ROOT/'data/final/manifest.json')
    headings=['Setting']+[x['label'] for x in variants]
    values=[]
    for item in variants:
        ev=evidence[item['run']];config=ev['config'];budget=manifest['budgets'][ev['budget']]
        values.append({
            'Learned parameters':f"{ev['parameters']:,}",
            'Width / layers / attention heads':f"{config['dim']} / {config['n_layers']} / {config['n_heads']}",
            'Training pool tokens / documents':f"{ev['training_stream_tokens']:,} tokens / {budget['documents']} documents",
            'Actual training text file':f"data/final/train_{ev['budget']}.jsonl",
            'Training seed':str(ev['seed']), 'Selected checkpoint step':str(ev['selected_step']),
            'Processed tokens at selected step':f"{ev['selected_step']*16*256:,}",
            'Maximum training steps / exposure':'3,072 / 12,582,912 tokens',
            'Temperature':str(item['temperature']), 'Top-K candidates':str(item['top_k']),
            'Generation seed':str(item['seed']), 'Maximum new tokens':str(item['max_tokens']),
            'Tokenizer / context':'Same 8,000-token BPE / 256 tokens',
            'Checkpoint file':item['run']+'.pt',
        })
    pieces=['<div class="settings-wrap"><table class="settings-table"><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headings)+'</tr></thead><tbody>']
    for key in values[0]:
        row=[v[key] for v in values];changed=len(set(row))>1
        pieces.append('<tr'+(' class="changed"' if changed else '')+'><th>'+html.escape(key)+(' · DIFFERS' if changed else ' · FIXED')+'</th>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>')
    pieces.append('</tbody></table></div><p class="setting-note">Highlighted rows show every difference, including checkpoint-selection consequences. Checkpoints are already trained; switching options never starts training. Every output uses your same typed starting words. Full-test scoring always uses the same 15 held-out documents / 128,685 targets.</p>')
    pieces.append('<p class="setting-note"><b>Training companies (same 25 across pools):</b> '+html.escape(', '.join(manifest['splits']['train']['companies']))+'.<br><b>Validation companies:</b> NVDA, PYPL. <b>Test companies:</b> AAPL, AMZN, KO. Source: curated SEC HTML Item 7 narratives, not PDFs.</p>')
    pieces.append('<details class="data-inventory"><summary>Show the exact company/year reports used by these training pools</summary>')
    for budget in dict.fromkeys(evidence[v['run']]['budget'] for v in variants):
        reports=inventory(budget)
        assert len(reports)==manifest['budgets'][budget]['documents']
        names=', '.join(f"{r['ticker']} {r['year']}" for r in reports)
        pieces.append(f'<p><b>{html.escape(budget)}: {len(reports)} reports</b><br>{html.escape(names)}</p>')
    pieces.append('</details>')
    if preset=='Training seed':pieces.append('<p><b>Only two training seeds were run. There is no third trained seed to compare.</b></p>')
    return ''.join(pieces)

def score_cards(result, preset):
    by_run={r['run']:r for r in result['rows']};variants=plan(preset)
    base=by_run[variants[0]['run']]['perplexity'];pieces=['<div class="score-grid">']
    for v in variants:
        r=by_run[v['run']];reduction=100*(base-r['perplexity'])/base
        pieces.append(f'''<article><h3>{html.escape(v['label'])}</h3><div class="score-number">{r['perplexity']:.4f}</div><p>Fresh full-test perplexity ↓</p><p>Mean negative log probability: {r['mean_nll']:.6f}<br>{r['tokens']:,} target tokens<br>Recorded reference: {r['recorded_perplexity']:.4f}<br>Match within 1%: <b>{'PASS' if r['within_one_percent'] else 'INVESTIGATE'}</b><br>{reduction:+.2f}% PPL reduction vs left column</p></article>''')
    if len(variants)==2:pieces.append('<article class="unused"><h3>No third trained seed</h3><p>The experiment trained seeds 42 and 123 only. No result is invented for a third model.</p></article>')
    pieces.append('</div>')
    if len(by_run)<len(variants):pieces.append('<p><b>Same checkpoint in these columns:</b> the full test was computed once and shown for each setting. Sampling controls change generated text, not teacher-forced perplexity.</p>')
    return ''.join(pieces)

def cards(result, scores=None):
    pieces=[f'<div class="result-heading"><b>Results actually run: {html.escape(result["preset"])}</b><br>Each column shows the <b>starting words in bold</b>, followed by the newly generated text, then its model’s full-test score.</div>','<div class="comparison-grid">']
    by_run={r['run']:r for r in scores['rows']} if scores else {}
    for r in result['rows']:
        sc=by_run.get(r['run'])
        if sc:
            baseline=by_run[result['rows'][0]['run']]['perplexity']
            reduction=100*(baseline-sc['perplexity'])/baseline
            scoreline=f'''<div class="fresh-score"><b>Fresh full-test perplexity ↓</b><div class="score-number">{sc['perplexity']:.4f}</div><div>Mean negative log probability: {sc['mean_nll']:.6f}<br>{sc['tokens']:,} held-out target tokens<br>Recorded reference: {sc['recorded_perplexity']:.4f}<br>Match within 1%: <b>{'PASS' if sc['within_one_percent'] else 'INVESTIGATE'}</b><br>{reduction:+.2f}% PPL reduction vs left column</div></div>'''
        else:
            scoreline='<div class="fresh-score pending-score">Fresh full-test score is not available yet. Run comparison computes it below this text.</div>'
        pieces.append(f'''<article><h2>{html.escape(r['label'])}</h2><div class="meta">{r['parameters']:,} parameters · {r['training_tokens']:,} training tokens<br>Training seed {r['training_seed']} · selected step {r['selected_step']}<br>Limit {r['max_new_tokens']} · T {r['temperature']} · K {r['top_k']} · generation seed {r['seed']}<br>{html.escape(r['run'])}.pt</div><h3 class="passage-label">Actual generated passage</h3><div class="generated-passage"><strong class="prompt-text">{html.escape(r['prompt'])}</strong><span class="generated-text">{html.escape(r['continuation']) or ' [End of document; no new text]'}</span></div><div class="timing">{r['generated_tokens']} new tokens · {r['generation_seconds']:.2f}s generation · {r['tokens_per_second']:.1f} tokens/s</div>{scoreline}</article>''')
    if len(result['rows'])==2:pieces.append('<article class="unused"><h2>No third trained seed</h2><p>Two genuine outputs are available: training seeds 42 and 123. Generation seed 7 is a sampling option, not a third trained model.</p></article>')
    return ''.join(pieces)+'</div>'

CSS=''' .gradio-container {max-width:1500px!important;margin:auto} .comparison-grid,.score-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px} .comparison-grid article,.score-grid article{background:#f1f6f8;border-top:5px solid #007f83;border-radius:10px;padding:18px;color:#163047} .comparison-grid h2{font-size:23px;margin:0 0 10px} .comparison-grid .meta{font-size:14px;color:#4b6374;overflow-wrap:anywhere} .generated-text{font-size:19px;line-height:1.5;white-space:pre-wrap;min-height:240px;overflow-wrap:anywhere} .comparison-grid .timing{font-size:15px;font-weight:600;margin:12px 0} .inline-score{font-weight:bold;color:#00696c;margin-top:12px}.score-number{font-size:32px;font-weight:bold;color:#007f83}.score-grid p{font-size:15px}.result-heading{padding:12px 0;font-size:18px}.settings-wrap{overflow-x:auto}.settings-table{width:100%;border-collapse:collapse;font-size:14px}.settings-table th,.settings-table td{padding:8px 10px;border:1px solid #dbe3e8;text-align:left;overflow-wrap:anywhere}.settings-table thead{background:#163047;color:white}.settings-table .changed{background:#fff0d8}.settings-table tbody th{width:29%}.setting-note{font-size:14px}.unused{opacity:.75} @media(max-width:850px){.comparison-grid,.score-grid{grid-template-columns:1fr}.generated-text{min-height:0}}'''
CSS+=''' .comparison-grid .generated-passage{font-size:19px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;min-height:240px}.comparison-grid .generated-text{min-height:0}.comparison-grid .passage-label{font-size:16px;margin:18px 0 8px;color:#00696c}.fresh-score{border-top:2px solid #b8d8d5;margin-top:18px;padding-top:14px;font-size:14px;line-height:1.65}.fresh-score .score-number{font-size:34px}.pending-score{color:#526877}'''
# Custom HTML keeps light card/table backgrounds in both Gradio themes. Set all
# foregrounds explicitly so Gradio's dark-mode prose styles cannot turn them white.
CSS+='''
.gradio-container .comparison-grid article,
.gradio-container .score-grid article {background:#f1f6f8!important;color:#163047!important;opacity:1!important}
.gradio-container .comparison-grid article *,
.gradio-container .score-grid article * {color:#163047!important}
.gradio-container .comparison-grid .meta {color:#425a6b!important}
.gradio-container .comparison-grid .passage-label,
.gradio-container .comparison-grid .score-number,
.gradio-container .score-grid .score-number {color:#00696c!important}
.gradio-container .settings-table tbody th,
.gradio-container .settings-table tbody td {background:#f8fafc!important;color:#163047!important}
.gradio-container .settings-table tbody tr.changed th,
.gradio-container .settings-table tbody tr.changed td {background:#fff0d8!important;color:#163047!important}
.gradio-container .settings-table thead th {background:#163047!important;color:#ffffff!important}
.gradio-container .result-heading {background:#f1f6f8!important;color:#163047!important;padding:12px;border-radius:8px}
.gradio-container .result-heading * {color:#163047!important}
'''

def build(engine):
    def generate(preset,prompt):
        try:
            result,path=engine.generate(preset,prompt)
            return cards(result),f"**Fresh inference completed** on {engine.device.upper()} in {result['total_seconds_including_load']:.2f}s including model loading. Repeat with unchanged settings to check repeatability. Log: `{path.name}`",str(path),None
        except ValueError as e:raise gr.Error(str(e))
    def compare_all(preset,prompt):
        try:
            generation,gpath=engine.generate(preset,prompt)
            # Show fresh text immediately while the full test runs; no download needed.
            yield cards(generation),'The generated text is visible. Computing full-test scores beneath each passage…',str(gpath),None
            scores,spath=engine.evaluate(preset)
            yield cards(generation,scores),f"**Complete: fresh text and full-test scores are displayed here.** Each score appears underneath its generated passage. Device: {engine.device.upper()}. No download is needed. Text run: {generation['timestamp_utc']}.",str(gpath),str(spath)
        except ValueError as e:raise gr.Error(str(e))
    def changed(preset):
        return description(preset),configuration(preset,engine.evidence),'<p>Settings changed. Click Run comparison to fill the columns with fresh results.</p>','Ready for the displayed configuration.',None,None
    with gr.Blocks(title='FinanceSLM | Live comparison v2.1',analytics_enabled=False) as demo:
        gr.Markdown('# FinanceSLM · Live comparison v2.1\n**Three generated passages side by side, with each model’s fresh score directly underneath. No JSON download is required.**')
        with gr.Row():
            preset=gr.Dropdown([(PRESET_LABELS[k],k) for k in PRESETS],value='Model size',label='Choose the exact values to compare')
            prompt=gr.Textbox(value=DEFAULT_PROMPT,label='Same starting words for every column',lines=2,max_length=8000)
        desc=gr.Markdown(description('Model size'))
        preview=gr.HTML(configuration('Model size',engine.evidence))
        with gr.Row():
            combined=gr.Button('Run comparison: text + full-test scores',variant='primary')
            button=gr.Button('Generate text only (faster)')
        gr.Markdown('## Generated passages and model scores\nThe bold starting words continue into newly generated text in each column. The score underneath evaluates the model on **15 held-out documents / 128,685 targets**; it is not a score of that generated passage. Sampling settings change the text, but the same checkpoint retains the same teacher-forced test perplexity.')
        output=gr.HTML('<p>Choose a comparison and generate. No saved continuation is displayed as a live result.</p>')
        status=gr.Markdown('Ready. The first run includes checkpoint loading. Logs save automatically in the project folder.')
        with gr.Accordion('Optional evidence downloads — not needed for the demo',open=False):
            genlog=gr.File(label='Generation JSON (optional)',interactive=False)
            scorelog=gr.File(label='Evaluation JSON (optional)',interactive=False)
        gr.Markdown('**Interpretation:** model/data comparisons hold the selection rule fixed; selected steps may differ. Generation seed and training seed are different controls. Generated numbers are unverified; lower perplexity does not prove factual accuracy or coherence.')
        preset.change(changed,inputs=preset,outputs=[desc,preview,output,status,genlog,scorelog],concurrency_id='inference',concurrency_limit=1)
        combined.click(compare_all,inputs=[preset,prompt],outputs=[output,status,genlog,scorelog],api_name='compare_all',concurrency_limit=1,concurrency_id='inference')
        button.click(generate,inputs=[preset,prompt],outputs=[output,status,genlog,scorelog],api_name='compare',concurrency_limit=1,concurrency_id='inference')
    return demo.queue(default_concurrency_limit=1,max_size=8)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--port',type=int,default=7863)
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'server.log',encoding='utf-8'),logging.StreamHandler()])
    logging.getLogger('httpx').setLevel(logging.WARNING)
    engine=Comparison(args.device)
    print(f'Live comparison on {engine.device}: http://127.0.0.1:{args.port}',flush=True)
    build(engine).launch(server_name='127.0.0.1',server_port=args.port,share=False,inbrowser=args.open_browser,css=CSS,theme=gr.themes.Soft(primary_hue='teal',font=['Arial','sans-serif']),footer_links=[])

if __name__=='__main__':main()
