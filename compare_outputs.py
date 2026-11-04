"""Create exact same-prompt Word/Markdown/HTML comparisons from saved outputs; never trains."""
from pathlib import Path
import argparse,hashlib,html,json
ROOT=Path(__file__).resolve().parent

def collect():
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'results/final/runs').glob('*/evaluation.json'))]
    assert len(rows)==18
    prompts=[s['prompt'] for s in rows[0]['samples']]
    assert len(prompts)==20 and all([s['prompt'] for s in r['samples']]==prompts for r in rows)
    assert len({r['config']['max_seq_len'] for r in rows})==1
    return rows,prompts

def build(output):
    from docx import Document
    from docx.shared import Inches,Pt
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    rows,prompts=collect();lookup={(r['model'],r['budget'],r['seed']):r for r in rows}
    output.mkdir(parents=True,exist_ok=True);doc=Document();section=doc.sections[0]
    section.orientation=WD_ORIENT.LANDSCAPE
    section.page_width=Inches(11.69);section.page_height=Inches(8.27)
    section.top_margin=section.bottom_margin=Inches(.55);section.left_margin=section.right_margin=Inches(.6)
    doc.styles['Normal'].font.name='Calibri';doc.styles['Normal'].font.size=Pt(10)
    doc.core_properties.title='FinanceSLM Same-Prompt Model Comparison'
    doc.add_heading('FinanceSLM: same-prompt comparison',0)
    intro='Exact saved continuations from the completed experiment, not regenerated or edited. Model-size comparisons hold the largest dataset (1,048,452 token IDs) fixed. Both training seeds are shown separately. Every model receives the same prompt, tokenizer, 256-token context, 80-token output limit, temperature 0.8, top-K 40 and generation seed 42. Different continuations are expected because model weights differ.'
    doc.add_paragraph(intro)
    doc.add_paragraph('Training seed and generation seed are different controls. These pages compare matching prompts within one training seed; do not compare unrelated prompts as a model-quality test. Perplexity is computed independently on the same full held-out documents. A higher Distinct-2 score measures variety, not correctness. Text can invent amounts and contradict itself; these are research outputs, not financial facts.')
    doc.add_paragraph('Part A contains all 20 prompts for both training seeds at the largest dataset. Part B contains all 20 prompts for both seeds at fixed 29.40M parameters while varying dataset size. The accompanying offline HTML explorer covers every model/data/seed combination (360 saved continuations). The historical final report explains the experiment and its limitations.')
    md=['# FinanceSLM same-prompt comparison','',intro,'','Settings: max new tokens 80; temperature 0.8; top-K 40; generation seed 42; context 256. All text is copied exactly from evaluation.json.','']
    count=0
    for axis in ['model size','dataset size']:
      for seed in [42,123]:
       candidates=[lookup[(m,'large',seed)] for m in ['15m','30m','50m']] if axis=='model size' else [lookup[('30m',b,seed)] for b in ['small','medium','large']]
       labels=[f'{r["parameters"]/1e6:.2f}M parameters / {r["training_stream_tokens"]:,} data tokens' for r in candidates]
       for i,prompt in enumerate(prompts):
        doc.add_page_break();doc.add_heading(f'{axis.title()} · training seed {seed} · prompt {i+1}',1);doc.add_paragraph(prompt,'Subtitle')
        table=doc.add_table(rows=1,cols=3);table.style='Light Shading Accent 1'
        repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for cell,label in zip(table.rows[0].cells,labels):cell.text=label
        cells=table.add_row().cells
        md += [f'## {axis.title()} | training seed {seed} | prompt {i+1}', '', '**Prompt:** '+prompt,'']
        for cell,r,label in zip(cells,candidates,labels):
            sample=r['samples'][i];cell.text=sample['continuation'] or '[End of document emitted immediately]'
            for p in cell.paragraphs:
                for run in p.runs:run.font.size=Pt(10)
            md += ['### '+label,'',sample['continuation'],'',f"Run: `{r['run']}` · Generated tokens: {sample['generated_tokens']} · Distinct-2: {sample['distinct_2']:.3f}",'']
        metric=table.add_row().cells
        for cell,r in zip(metric,candidates):
            s=r['samples'][i];cell.text=f"Full test PPL: {r['test']['perplexity']:.2f}\nGenerated tokens: {s['generated_tokens']}\nDistinct-2: {s['distinct_2']:.3f}\nRepeated trigrams: {s['repeated_trigram_rate']:.3f}\nRun: {r['run']}"
            for p in cell.paragraphs:
                for run in p.runs:run.font.size=Pt(8)
        doc.add_paragraph('Output length is capped, so a final unfinished sentence may simply reflect the limit. Financial claims are unverified. Source: results/final/runs/<run>/evaluation.json.','Caption')
        count+=1
    doc.save(output/'Same_Prompt_Comparison.docx')
    (output/'Same_Prompt_Comparison.md').write_text('\n'.join(md),encoding='utf-8')
    # Embed JSON without allowing saved text to terminate the script element.
    payload=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinanceSLM comparison</title>
<style>body{font:16px system-ui;margin:2rem;color:#19343b;background:#f5f8fa}h1{margin-bottom:.4rem}label{display:inline-block;margin:.8rem 1rem .8rem 0;max-width:100%}select{padding:.6rem;max-width:100%}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}article{background:white;padding:1.2rem;border:1px solid #cbdadd;border-radius:8px}pre{white-space:pre-wrap;font:inherit;line-height:1.6}.meta{color:#50646a;font-size:.85rem;overflow-wrap:anywhere}#prompt{font-size:1.15rem;font-weight:600}button{padding:.6rem}@media(max-width:850px){main{grid-template-columns:1fr}}</style>
<h1>FinanceSLM: compare the same prompt</h1><p>Exact saved outputs, no model or server needed. 18 runs × 20 prompts. Text may contain invented figures or contradictions.</p>
<label>Compare <select id="axis"><option value="model">Model sizes</option><option value="data">Dataset sizes</option></select></label>
<label>Fixed dataset <select id="budget"><option>large</option><option>medium</option><option>small</option></select></label>
<label>Fixed model <select id="model"><option>30m</option><option>15m</option><option>50m</option></select></label>
<label>Training seed <select id="seed"><option>42</option><option>123</option></select></label><br>
<label>Prompt <select id="index"></select></label><p id="prompt"></p>
<p class="meta">Same tokenizer and 256-token context. Generation: maximum 80 new tokens, temperature 0.8, top-K 40, seed 42. Full test perplexity uses the same held-out documents.</p><main id="cards"></main>
<script id="records" type="application/json">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('records').textContent),ids=['axis','budget','model','seed','index'];
const fields=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));
rows[0].samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'. '+s.prompt;fields.index.append(o)});
function render(){const i=Number(fields.index.value),seed=Number(fields.seed.value),axis=fields.axis.value;
fields.budget.disabled=axis!=='model';fields.model.disabled=axis!=='data';
const candidates=axis==='model'?['15m','30m','50m'].map(m=>rows.find(r=>r.model===m&&r.budget===fields.budget.value&&r.seed===seed)):['small','medium','large'].map(b=>rows.find(r=>r.model===fields.model.value&&r.budget===b&&r.seed===seed));
document.getElementById('prompt').textContent=rows[0].samples[i].prompt;const cards=document.getElementById('cards');cards.replaceChildren();
for(const r of candidates){const s=r.samples[i],a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('pre'),m=document.createElement('p');h.textContent=(r.parameters/1e6).toFixed(2)+'M / '+r.training_stream_tokens.toLocaleString()+' data tokens';p.textContent=s.continuation||'[End of document]';m.className='meta';m.textContent=r.run+' | Test PPL '+r.test.perplexity.toFixed(2)+' | '+s.generated_tokens+' generated tokens | Distinct-2 '+s.distinct_2.toFixed(3);a.append(h,p,m);cards.append(a)}}
ids.forEach(id=>fields[id].addEventListener('change',render));render();</script></html>'''.replace('PAYLOAD',payload)
    (output/'Same_Prompt_Comparison.html').write_text(page,encoding='utf-8')
    audit=dict(runs=len(rows),prompts_per_run=len(prompts),saved_continuations=sum(len(r['samples']) for r in rows),word_comparison_tables=count,identical_prompt_order=True,generation_settings=json.loads((ROOT/'results/final/comparison.json').read_text(encoding='utf-8'))['protocol']['generation'],source_checksums={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'results/final/runs').glob('*/evaluation.json'))})
    (output/'comparison_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_checksums'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/comparison');args=parser.parse_args();build(args.output)
