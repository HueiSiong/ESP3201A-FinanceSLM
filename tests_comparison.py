"""Source-only regression checks; no weight downloads or inference required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import compare_app as app


class ComparisonTests(unittest.TestCase):
    def test_matched_training_controls(self):
        evidence = {p.parent.name: app.read(p) for p in (app.ROOT/'results/final/runs').glob('*/evaluation.json')}
        rows = [evidence[x['run']] for x in app.plan('Model size')]
        self.assertEqual({r['training_stream_tokens'] for r in rows}, {1048452})
        self.assertEqual({r['seed'] for r in rows}, {42})
        self.assertEqual([r['parameters'] for r in rows], [15464064, 29401600, 49737600])
        rows = [evidence[x['run']] for x in app.plan('Training data')]
        self.assertEqual({r['parameters'] for r in rows}, {15464064})
        self.assertEqual([r['training_stream_tokens'] for r in rows], [357288, 706370, 1048452])
        self.assertEqual({r['seed'] for r in rows}, {42})

    def test_sampling_changes_one_control(self):
        keys = {'Temperature':'temperature', 'Top-K':'top_k', 'Generation seed':'seed', 'Maximum new tokens':'max_tokens'}
        for preset, changed in keys.items():
            rows = app.plan(preset)
            for key in ['run', 'temperature', 'top_k', 'seed', 'max_tokens']:
                self.assertEqual(len({r[key] for r in rows}), 3 if key == changed else 1)
        self.assertEqual(len(app.plan('Training seed')), 2)
        with self.assertRaises(ValueError): app.plan('Unknown')

    def test_inventory_and_preview(self):
        evidence = {p.parent.name: app.read(p) for p in (app.ROOT/'results/final/runs').glob('*/evaluation.json')}
        for budget, expected in [('small',36), ('medium',73), ('large',109)]:
            self.assertEqual(len(app.inventory(budget)), expected)
        for preset in app.PRESETS:
            preview = app.configuration(preset, evidence)
            self.assertIn('Selected checkpoint step', preview)
            self.assertIn('128,685', preview)
            self.assertIn('DIFFERS', preview)

    def test_missing_and_corrupt_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'assets.json').write_text((app.ROOT/'assets.json').read_text(encoding='utf-8'), encoding='utf-8')
            with patch.object(app, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, '--group 15m'): app.checked_checkpoint(app.BASE)
                target = root/'checkpoints/final'/f'{app.BASE}.pt'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'corrupted checkpoint')
                with self.assertRaisesRegex(ValueError, 'checksum'): app.checked_checkpoint(app.BASE)
                with self.assertRaisesRegex(ValueError, 'Unknown'): app.checked_checkpoint('../outside')

    def test_frozen_evidence_checked(self):
        app.verify_frozen_inputs()
        with patch.object(app, 'sha', return_value='incorrect'):
            with self.assertRaisesRegex(ValueError, 'Frozen input'): app.verify_frozen_inputs()

    def test_inline_text_scores_and_escaping(self):
        rows = []
        scores = []
        for i, variant in enumerate(app.plan('Model size')):
            rows.append(dict(label=variant['label'],run=variant['run'],parameters=1,training_tokens=2,training_seed=42,selected_step=512,max_new_tokens=80,temperature=.8,top_k=40,seed=42,prompt='<script>prompt</script>',continuation='<b>continuation</b>',generated_tokens=10,generation_seconds=1,tokens_per_second=10))
            scores.append(dict(run=variant['run'],perplexity=60+i,mean_nll=4,tokens=128685,recorded_perplexity=60+i,within_one_percent=True))
        rendered = app.cards(dict(preset='Model size',rows=rows),dict(rows=list(reversed(scores))))
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;b&gt;continuation&lt;/b&gt;', rendered)
        self.assertEqual(rendered.count('class="generated-text"'),3)
        self.assertEqual(rendered.count('class="score-number"'),3)
        articles = rendered.split('<article>')[1:]
        for i, article in enumerate(articles):
            self.assertIn(f'{60+i:.4f}',article)
            self.assertLess(article.index('generated-text'),article.index('fresh-score'))
        pending = app.cards(dict(preset='Model size',rows=rows))
        self.assertNotIn('class="score-number"',pending)

    def test_logs_stay_under_work(self):
        self.assertTrue(app.OUT.is_relative_to(app.ROOT/'work'))


if __name__ == '__main__': unittest.main(verbosity=2)
