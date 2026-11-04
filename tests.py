"""Offline regression tests for asset integrity, prompt alignment, and training isolation."""
from pathlib import Path
import json,tempfile,unittest,zipfile
import artifacts
ROOT=Path(__file__).resolve().parent

class PublicationTests(unittest.TestCase):
    def test_same_prompts_and_controls(self):
        from compare_outputs import collect
        rows,prompts=collect()
        self.assertEqual(len(rows),18);self.assertEqual(len(prompts),20)
        for seed in [42,123]:
            for budget in ['small','medium','large']:
                rs=[r for r in rows if r['seed']==seed and r['budget']==budget]
                self.assertEqual(len({r['training_stream_tokens'] for r in rs}),1)
                self.assertEqual(len({r['config']['max_seq_len'] for r in rs}),1)

    def test_frozen_inputs_unchanged(self):
        for path,expected in json.loads((ROOT/'checksums.json').read_text(encoding='utf-8')).items():
            self.assertEqual(artifacts.sha(ROOT/path),expected,path)

    def test_archive_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError,'checksum'):artifacts.install(p,{'bytes':9,'sha256':'0'*64,'files':[]})

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../outside.txt',b'bad')
            entry={'bytes':p.stat().st_size,'sha256':artifacts.sha(p),'files':[{'path':'../outside.txt','bytes':3,'sha256':'0'*64}]}
            with self.assertRaisesRegex(RuntimeError,'Unsafe'):artifacts.install(p,entry)
            self.assertFalse((ROOT.parent/'outside.txt').exists())

    def test_training_cannot_overwrite_evidence(self):
        from research import output_path
        for name in ['results/final','data/final','../outside','checkpoints/final']:
            with self.assertRaises(ValueError):output_path(name)

if __name__=='__main__':unittest.main(verbosity=2)
