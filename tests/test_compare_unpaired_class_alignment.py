import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import distributed as dist

from blue import Blue, BlueConfig
from blue.data import JointDataset
from blue.training import save_checkpoint
from tools.compare_unpaired_class_alignment import audit_class_exposure


class CompareUnpairedClassAlignmentTests(unittest.TestCase):
    def test_class_exposure_counts_actual_pairs_and_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data.jsonl'
            records = [dict(text='zero', class_label=0), dict(text='one', class_label=1),
                       dict(series=[[0., 0., 0.]], class_label=0),
                       dict(series=[[1., 1., 1.]], class_label=1)]
            def audit():
                data.write_text(''.join(json.dumps(row) + '\n' for row in records))
                dataset = JointDataset(data, BlueConfig.preset('tiny'), allow_mixed=True)
                return audit_class_exposure(dataset, batch_size=2, seed=3, steps=1)
            counts = audit()
            self.assertEqual(counts, dict(cross_modal_steps=1, shared_class_steps=1, contributing_steps=1,
                                          steps_with_negative_pairs=1, eligible_text_views=2,
                                          eligible_series_views=2, positive_pairs=2, negative_pairs=2))
            records[1]['class_label'] = -100
            counts = audit()
            self.assertEqual(counts['shared_class_steps'], 1)
            self.assertEqual(counts['contributing_steps'], 0)
            self.assertEqual(counts['negative_pairs'], 0)
            records[0]['series'] = [[0., 0., 0.]]
            with self.assertRaisesRegex(ValueError, 'texte ou série seules'):
                audit()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_study_compares_modes_on_disjoint_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / 'parent.pt', root / 'study'
            torch.manual_seed(31)
            save_checkpoint(checkpoint, Blue(BlueConfig.preset('tiny')))
            script = Path(__file__).resolve().parents[1] / 'tools/compare_unpaired_class_alignment.py'
            command = [sys.executable, str(script), '--checkpoint', str(checkpoint), '--output-dir', str(output),
                       '--seeds', '17', '--steps', '2', '--train-examples-per-modality', '8', '--batch-size', '2',
                       '--text-seed', '100', '--series-seed', '101', '--test-seed', '2000000',
                       '--test-batches', '1', '--test-batch-size', '4']
            run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout[-8000:])
            report = json.loads((output / 'comparison.json').read_text())
            rows = {row['mode']: row for row in report['runs']}
            self.assertEqual(set(rows), {'control', 'class'})
            self.assertEqual(rows['control']['nonzero_alignment_steps'], 0)
            self.assertEqual(rows['control']['class_exposure'], rows['class']['class_exposure'])
            for row in rows.values():
                self.assertEqual(row['step'], 2)
                self.assertEqual(row['exposure']['sample_views'], 8)
                self.assertEqual(row['metrics']['examples'], 4)
            records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 16)
            self.assertTrue(all(('text' in row) != ('series' in row) for row in records))
            before = (output / 'comparison.json').read_bytes()
            rejected = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual((output / 'comparison.json').read_bytes(), before)

    def test_rejects_overlapping_data_and_test_seeds(self):
        script = Path(__file__).resolve().parents[1] / 'tools/compare_unpaired_class_alignment.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = subprocess.run([sys.executable, str(script), '--checkpoint', str(root / 'unused.pt'),
                                  '--output-dir', str(root / 'study'), '--text-seed', '100',
                                  '--series-seed', '100'], text=True, capture_output=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('doivent être distincts', run.stderr)
            self.assertFalse((root / 'study').exists())


if __name__ == '__main__':
    unittest.main()
