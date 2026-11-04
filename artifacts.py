"""Fetch or install checksum-verified GitHub release assets. No third-party dependencies."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,os,re,shutil,tempfile,urllib.request,zipfile
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def manifest():return json.loads((ROOT/'assets.json').read_text(encoding='utf-8'))

def verify(groups):
    result=[]
    for group in groups:
        for f in manifest()['groups'][group]['files']:
            path=ROOT/f['path']
            if not path.is_file() or path.stat().st_size!=f['bytes'] or sha(path)!=f['sha256']:
                raise RuntimeError('Missing or incorrect asset: '+f['path'])
            result.append(f['path'])
    return result

def install(archive,entry):
    if not archive.is_file() or archive.stat().st_size!=entry['bytes'] or sha(archive)!=entry['sha256']:
        raise RuntimeError('Archive checksum/size mismatch: '+str(archive))
    expected={f['path']:f for f in entry['files']}
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(expected) or set(z.namelist())!=set(expected):raise RuntimeError('Unexpected archive members')
        for name,f in expected.items():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or rel.parts[:2] not in [('checkpoints','final'),('data','Raw')]:raise RuntimeError('Unsafe asset path')
            dest=(ROOT/Path(*rel.parts)).resolve()
            if not dest.is_relative_to(ROOT):raise RuntimeError('Asset path leaves project')
            if dest.exists():
                if sha(dest)==f['sha256']:continue
                raise RuntimeError('Existing file differs; move it aside before installing: '+str(dest))
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_suffix(dest.suffix+'.installing')
            try:
                if z.getinfo(name).file_size!=f['bytes']:raise RuntimeError('Member size mismatch')
                with z.open(name) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target,1024*1024)
                if sha(temporary)!=f['sha256']:raise RuntimeError('Member checksum mismatch')
                temporary.replace(dest)
            finally:temporary.unlink(missing_ok=True)

def main():
    m=manifest();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['fetch','install','verify']);parser.add_argument('--group',choices=list(m['groups'])+['all'],default='demo')
    parser.add_argument('--repo',help='GitHub OWNER/REPOSITORY');parser.add_argument('--tag',default=m['default_tag'])
    parser.add_argument('--from-dir',type=Path);parser.add_argument('--base-url',help='Override download base, e.g. http://127.0.0.1:PORT for local testing')
    args=parser.parse_args();groups=list(m['groups']) if args.group=='all' else [args.group]
    if args.action=='verify':print(json.dumps({'verified':verify(groups)},indent=2));return
    if args.action=='install' and not args.from_dir:parser.error('install needs --from-dir')
    if args.action=='fetch':
        if args.base_url:
            base=args.base_url.rstrip('/')
            if not (base.startswith('https://') or re.match(r'^http://(127\.0\.0\.1|localhost)(:\d+)?$',base)):parser.error('Use HTTPS or a loopback test server')
        else:
            if not args.repo or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo):parser.error('fetch needs --repo OWNER/REPOSITORY')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.tag):parser.error('Invalid release tag')
            base=f'https://github.com/{args.repo}/releases/download/{args.tag}'
        cache=ROOT/'work/downloads';cache.mkdir(parents=True,exist_ok=True)
    for group in groups:
        entry=m['groups'][group]
        if args.action=='install':archive=args.from_dir/entry['archive']
        else:
            archive=cache/entry['archive']
            if not archive.exists() or sha(archive)!=entry['sha256']:
                partial=archive.with_suffix('.part')
                try:
                    request=urllib.request.Request(base+'/'+entry['archive'],headers={'User-Agent':'FinanceSLM-reproducibility/1.0'})
                    with urllib.request.urlopen(request,timeout=90) as response,partial.open('wb') as target:
                        total=0
                        while block:=response.read(1024*1024):
                            total+=len(block)
                            if total>entry['bytes']:raise RuntimeError('Download exceeds expected size')
                            target.write(block)
                    if partial.stat().st_size!=entry['bytes'] or sha(partial)!=entry['sha256']:raise RuntimeError('Downloaded archive failed checksum verification')
                    partial.replace(archive)
                finally:partial.unlink(missing_ok=True)
        install(archive,entry);print('Installed and verified:',group,flush=True)
    verify(groups)

if __name__=='__main__':main()
