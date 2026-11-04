"""Audit and curate a separate, paragraph-preserving FinanceSLM corpus."""
from local_runtime import configure
configure()
from pathlib import Path
import csv, hashlib, json, re, unicodedata
from collections import Counter
from bs4 import BeautifulSoup
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/final'
DATA = ROOT / 'data/final'
START = re.compile(r'^\s*ITEM\s+7\s*[.\-–:]*\s*MANAGEMENT[’\']?S?\s+DISCUSSION', re.I | re.M)
END = re.compile(r'^\s*ITEM\s+(?:7\s*A\s*[.\-–:]*\s*QUANTITATIVE|8\s*[.\-–:]*\s*FINANCIAL\s+STATEMENTS)', re.I | re.M)
TERMS = ['revenue', 'cash flow', 'liquidity', 'net income', 'operating', 'results of operations']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def originals():
    files = sorted((ROOT/'data/Raw').rglob('*.html'))
    files += [ROOT/'data/historical_metadata.csv', ROOT/'data/expanded_metadata.csv']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files if p.is_file()}


def norm(text):
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    return re.sub(r'\s+', ' ', text).strip()

def extract(html):
    soup = BeautifulSoup(html, 'html.parser')
    counts = Counter()
    for tag in list(soup.find_all(True)):
        if not tag.name or tag.attrs is None: continue
        name = tag.name.lower()
        style = (tag.get('style') or '').replace(' ', '').lower()
        if name in {'head','script','style','ix:header','ix:hidden','ix:resources','noscript'} or name.startswith(('xbrli:', 'link:')) or 'display:none' in style or 'visibility:hidden' in style:
            counts['hidden_or_metadata_elements_removed'] += 1
            tag.decompose()
    # Remove numerical tables, retain narrative/layout tables and inline financial numbers.
    for table in reversed(soup.find_all('table')):
        if not table.name or table.attrs is None: continue
        cells = table.find_all(['td','th'])
        strings = [norm(c.get_text(' ', strip=True)) for c in cells]
        numeric = sum(bool(re.search(r'\d', s)) and len(s.split()) <= 6 for s in strings)
        long_prose = sum(len(s.split()) >= 30 for s in strings)
        if len(cells) >= 4 and numeric >= 3 and numeric/max(1,len(cells)) >= .2 and long_prose == 0:
            counts['numeric_tables_removed'] += 1
            table.replace_with('\n\n')
    for tag in soup.find_all(['p','div','h1','h2','h3','h4','li','tr','br']):
        tag.insert_before('\n'); tag.insert_after('\n')
    text = (soup.body or soup).get_text(' ', strip=False)
    lines = [norm(line) for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    starts, ends = list(START.finditer(text)), list(END.finditer(text))
    candidates = []
    for start in starts:
        after = [end for end in ends if end.start() > start.end()]
        if not after: continue
        end = after[0]
        section = text[start.start():end.start()]
        words = len(section.split())
        score = sum(term in section.lower() for term in TERMS)
        if 1200 <= words <= 35000 and score >= 3:
            candidates.append((score, words, section, start.start(), end.start(), end.group()))
    if not candidates:
        return None, dict(counts, start_matches=len(starts), end_matches=len(ends), reason='No bounded Item 7 candidate passing length and keyword checks')
    best = max(candidates)
    if re.search(r'critical audit matter|report of independent registered', best[2], re.I):
        return None, dict(counts, start_matches=len(starts), end_matches=len(ends), reason='Auditor-report content inside candidate: ambiguous incorporated-report boundary; excluded conservatively')
    paragraphs = []
    for line in best[2].splitlines():
        if re.search(r'table\s+of\s+contents',line,re.I) or START.search(line):
            counts['navigation_or_heading_lines_removed'] += 1; continue
        # Retain substantial prose; omit standalone headings, page labels and flattened table fragments.
        words = line.split()
        if len(words) < 12 or sum(c.isalpha() for c in line)/max(1,len(line)) < .45 or re.match(r'Amounts in millions|Dollars in millions', line, re.I):
            counts['short_or_numeric_lines_removed'] += 1; continue
        if re.search(r'\b(?:us-gaap|dei|xbrli):',line):
            counts['metadata_lines_removed'] += 1; continue
        paragraphs.append(line)
    return paragraphs, dict(counts, start_matches=len(starts), end_matches=len(ends), candidate_count=len(candidates), start=best[3], end=best[4], end_heading=best[5], candidate_words=best[1], keyword_score=best[0])

def main():
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists(): raise RuntimeError('Final data already exists; refusing to overwrite it.')
    for stale in (DATA/'extracted').glob('*.txt'): stale.unlink()
    if not (OUT/'original_hashes.json').exists(): dump(OUT/'original_hashes.json', originals())
    source_meta = {}
    for p in [ROOT/'data/historical_metadata.csv', ROOT/'data/expanded_metadata.csv']:
        for row in csv.DictReader(p.open(encoding='utf-8')): source_meta[(row['ticker'],int(row['year']))]=row
    # The original-corpus audit is historical evidence in results/final/.
    # Rebuilding from raw snapshots does not require that obsolete corpus.
    records=[]; audit=[]
    paths=sorted((ROOT/'data/Raw/historical').glob('*.html'))+sorted((ROOT/'data/Raw/expanded').glob('*.html'))
    for i,p in enumerate(paths):
        ticker,year=p.stem.split('_')[:2]; year=int(year)
        paragraphs, stats=extract(p.read_text(encoding='utf-8',errors='replace'))
        stat=dict(file=str(p.relative_to(ROOT)),ticker=ticker,year=year,**stats)
        if paragraphs:
            text='\n\n'.join(paragraphs)
            if len(text.split()) >= 1000:
                # Hold out companies, keeping the original held-out companies isolated.
                split='test' if ticker in {'AAPL','AMZN','JPM','KO'} else 'validation' if ticker in {'NVDA','PYPL','BAC','PEP'} else 'train'
                r=dict(ticker=ticker,year=year,text=text,words=len(text.split()),split=split,source_file=str(p.relative_to(ROOT)),source_url=source_meta.get((ticker,year),{}).get('source_url',''),paragraphs=len(paragraphs))
                records.append(r); stat.update(status='accepted_before_dedup',words=r['words'],paragraphs=len(paragraphs))
                dest=DATA/'extracted'/f'{ticker}_{year}_item7.txt';dest.parent.mkdir(exist_ok=True);dest.write_text(text,encoding='utf-8')
            else: stat.update(status='excluded',reason='Fewer than 1000 prose words after cleaning')
        else: stat['status']='excluded'
        audit.append(stat)
        print(f'{i+1}/{len(paths)} {p.name}: {stat["status"]}',flush=True)
    # Exact paragraph deduplication: test, then validation, then training. This prevents exact leakage.
    seen=set(); dedup=[]; final=[]
    for split in ['test','validation','train']:
        for r in sorted([r for r in records if r['split']==split],key=lambda r:(r['ticker'],-r['year'])):
            retained=[]; removed=0
            for p in r['text'].split('\n\n'):
                key=hashlib.sha256(norm(p).lower().encode()).hexdigest()
                if key in seen: removed+=1
                else: seen.add(key); retained.append(p)
            r=dict(r,text='\n\n'.join(retained),words=sum(len(p.split()) for p in retained),paragraphs=len(retained))
            dedup.append(dict(ticker=r['ticker'],year=r['year'],split=split,removed_paragraphs=removed,retained_words=r['words']))
            if r['words']>=500: final.append(r)
    for split in ['train','validation','test']:
        subset=[r for r in final if r['split']==split]
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
    tokenizer=Tokenizer(BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=ByteLevel(add_prefix_space=False);tokenizer.decoder=ByteLevelDecoder()
    train=[r for r in final if r['split']=='train']
    # Shared tokenizer fitted only on the largest training pool; vocabulary exposure is a stated control.
    tokenizer.train_from_iterator((r['text'] for r in train),trainer=BpeTrainer(vocab_size=8000,min_frequency=2,initial_alphabet=ByteLevel.alphabet(),special_tokens=['<pad>','<unk>','<bos>','<eos>']))
    tokenizer.save(str(DATA/'finance_tokenizer.json'))
    eos=tokenizer.token_to_id('<eos>')
    # Stable shuffled, company-round-robin order gives nested sets spanning the available training companies.
    rng=np.random.RandomState(2026); companies=sorted({r['ticker'] for r in train});rng.shuffle(companies)
    buckets={c: sorted([r for r in train if r['ticker']==c],key=lambda r:r['year']) for c in companies}
    for b in buckets.values():rng.shuffle(b)
    ordered=[]
    while any(buckets.values()):
        for c in companies:
            if buckets[c]:ordered.append(buckets[c].pop())
    token_docs=[tokenizer.encode(r['text']).ids+[eos] for r in ordered]
    total=sum(map(len,token_docs)); budgets={}
    for label,fraction in [('small',1/3),('medium',2/3),('large',1)]:
        selected=[];n=0
        for r,ids in zip(ordered,token_docs):
            selected.append(r);n+=len(ids)
            if n>=total*fraction:break
        ids=[i for doc in token_docs[:len(selected)] for i in doc]
        np.save(DATA/f'train_{label}.npy',np.array(ids,dtype=np.uint16))
        (DATA/f'train_{label}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
        budgets[label]=dict(tokens=len(ids),documents=len(selected),companies=sorted({r['ticker'] for r in selected}))
    for split in ['validation','test']:
        subset=[r for r in final if r['split']==split]
        docs=[tokenizer.encode(r['text']).ids+[eos] for r in subset]
        np.save(DATA/f'{split}.npy',np.array([i for d in docs for i in d],dtype=np.uint16))
        dump(DATA/f'{split}_documents.json',[dict(ticker=r['ticker'],year=r['year'],ids=ids) for r,ids in zip(subset,docs)])
    dump(OUT/'extraction_audit.json',audit);dump(OUT/'deduplication_audit.json',dedup)
    summary=dict(source_html_files=len(paths),accepted_before_dedup=len(records),final_documents=len(final),excluded_sources=[r for r in audit if r['status']=='excluded'],splits={s:dict(documents=sum(r['split']==s for r in final),words=sum(r['words'] for r in final if r['split']==s),companies=sorted({r['ticker'] for r in final if r['split']==s})) for s in ['train','validation','test']},budgets=budgets,vocab_size=tokenizer.get_vocab_size(),removed_duplicate_paragraphs=sum(r['removed_paragraphs'] for r in dedup),limitations=['Heuristic extraction requires sampled manual review; no guarantee of zero noise.','Numeric tables excluded: narrative completion scope, not tabular reasoning.','Exact paragraph deduplication does not eliminate paraphrases or near duplicates.','Shared tokenizer fitted on largest training pool; smaller runs share vocabulary exposure.','Dataset sizes are available cleaned tokens, not proposed 20M/60M/120M.'])
    dump(DATA/'manifest.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'work/rebuilt-data')
    parser.add_argument('--audit-dir',type=Path,default=ROOT/'work/rebuilt-audit')
    args=parser.parse_args()
    DATA=args.output_dir.resolve();OUT=args.audit_dir.resolve()
    if not DATA.is_relative_to(ROOT/'work') or not OUT.is_relative_to(ROOT/'work') or DATA==OUT:
        raise SystemExit('Use two separate directories under work/ to preserve frozen evidence.')
    if any(p.exists() and any(p.iterdir()) for p in [DATA,OUT]):
        raise SystemExit('Rebuild output directories must be empty; choose new work/ directories.')
    if not (ROOT/'data/Raw/historical').is_dir():
        raise SystemExit('Install the raw-reports asset group first: python artifacts.py fetch --repo OWNER/REPO --group raw')
    main()

