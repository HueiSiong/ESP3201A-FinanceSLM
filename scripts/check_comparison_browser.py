"""Optional real-browser verification. Requires Playwright and a browser, plus model groups.

Example: python scripts/check_comparison_browser.py --device cuda --browser-channel chrome
Install Playwright separately for this developer check; it is not needed to run the demo.
"""
from pathlib import Path
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
MEASURE = r'''() => {
 const selectors=['.settings-table thead th','.settings-table tbody th','.settings-table tbody td','.result-heading','.comparison-grid h2','.comparison-grid .meta','.passage-label','.prompt-text','.generated-text','.timing','.fresh-score','.score-number'];
 function rgb(s){return (s.match(/[\d.]+/g)||[]).map(Number)}
 function lum(c){return c.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4}).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0)}
 const rows=[];
 for(const selector of selectors)for(const el of document.querySelectorAll(selector)){
  let bg=null;
  for(let node=el;node;node=node.parentElement){let c=rgb(getComputedStyle(node).backgroundColor);if(c.length===3||c[3]===1){bg=c;break}}
  if(!bg)throw Error('No opaque background: '+selector);
  const fg=rgb(getComputedStyle(el).color),a=lum(fg),b=lum(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
  rows.push({selector,contrast:ratio});
 }
 return rows;
}'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable, help='Interpreter for the demo server')
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    parser.add_argument('--browser-channel', default=None, help='For example chrome; default uses Playwright Chromium')
    args = parser.parse_args()
    out = ROOT/'work/comparison-browser'; out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    with (out/'server.log').open('w',encoding='utf-8') as log:
        server = subprocess.Popen([args.python,'-B','-u',str(ROOT/'compare_app.py'),'--device',args.device,'--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            for _ in range(90):
                if server.poll() is not None: raise RuntimeError('Server exited; inspect work/comparison-browser/server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}',timeout=1) as r:
                        if r.status==200: break
                except Exception: time.sleep(1)
            else: raise RuntimeError('Server startup timed out')
            checks=[]; themes={}; errors=[]; downloads=[]
            with sync_playwright() as p:
                browser=p.chromium.launch(channel=args.browser_channel,headless=True)
                page=browser.new_page(viewport={'width':1600,'height':1100},color_scheme='dark')
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('download',lambda d:downloads.append(d.suggested_filename))
                page.goto(f'http://127.0.0.1:{port}/?__theme=dark',wait_until='networkidle')
                cases=[('Model size','Model parameters:'),('Training data','Training text:'),('Temperature','Temperature:'),('Top-K','Top-K candidates:'),('Generation seed','Generation random seed:'),('Maximum new tokens','Maximum new tokens:'),('Training seed','Training random seed:')]
                for index,(key,label) in enumerate(cases):
                    if index:
                        page.get_by_role('combobox').click()
                        page.get_by_role('option',name=re.compile('^'+re.escape(label))).click()
                        expect(page.locator('.comparison-grid article')).to_have_count(0,timeout=15000)
                    preview=page.locator('.settings-table').inner_text()
                    assert all(s in preview for s in ['DIFFERS','FIXED','Checkpoint file','Selected checkpoint step'])
                    page.get_by_role('button',name='Run comparison: text + full-test scores',exact=True).click()
                    expect(page.get_by_text('Complete: fresh text and full-test scores are displayed here.',exact=True)).to_be_visible(timeout=900000)
                    expected_count=2 if key=='Training seed' else 3
                    expect(page.locator('.generated-text')).to_have_count(expected_count)
                    texts=page.locator('.generated-text').all_text_contents()
                    assert all(text.strip() for text in texts)
                    assert all(text=="The company's revenue increased" for text in page.locator('.prompt-text').all_text_contents())
                    scores=[float(s) for s in page.locator('.score-number').all_text_contents()]
                    assert len(scores)==expected_count
                    assert page.locator('.comparison-grid').inner_text().count('PASS')==expected_count
                    assert page.locator('.comparison-grid').inner_text().count('128,685 held-out target tokens')==expected_count
                    for col in page.locator('.comparison-grid article:not(.unused)').all():
                        assert col.evaluate('(el)=>el.querySelector(".fresh-score").getBoundingClientRect().top>el.querySelector(".generated-passage").getBoundingClientRect().bottom')
                    if key in ['Temperature','Top-K','Generation seed','Maximum new tokens']: assert len(set(scores))==1
                    if key=='Training seed': assert page.locator('.unused').inner_text().startswith('No third trained seed')
                    checks.append(dict(preset=key,outputs=expected_count,scores=scores,full_test_tokens=128685,all_within_one_percent=True))
                    if index==0:
                        for theme in ['dark','light']:
                            if theme=='light':
                                page.emulate_media(color_scheme='light')
                                page.evaluate('document.querySelectorAll(".dark").forEach(e=>e.classList.remove("dark"))')
                            rows=page.evaluate(MEASURE)
                            assert rows and min(r['contrast'] for r in rows)>=4.5
                            themes[theme]=dict(checked_elements=len(rows),minimum_contrast=min(r['contrast'] for r in rows))
                            page.locator('.comparison-grid').screenshot(path=str(out/f'outputs_{theme}.png'))
                    print('PASS',key,scores,flush=True)
                before=page.locator('.generated-text').all_text_contents()
                page.get_by_role('button',name='Generate text only (faster)',exact=True).click()
                expect(page.get_by_text('Fresh inference completed',exact=True)).to_be_visible(timeout=180000)
                assert page.locator('.generated-text').all_text_contents()==before
                assert page.locator('.score-number').count()==0
                page.set_viewport_size({'width':1280,'height':720})
                assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
                assert not errors and not downloads,(errors,downloads)
                browser.close()
            result=dict(passed=True,device=args.device,all_seven_presets=checks,themes=themes,no_downloads_required=True,stale_results_cleared=True,repeatable_generation=True,text_only_clears_scores=True,projector_no_horizontal_overflow=True,javascript_errors=errors)
            (out/'browser_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        finally:
            if server.poll() is None:
                if os.name=='nt': subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else: server.terminate()
                server.wait(timeout=30)

if __name__=='__main__': main()
