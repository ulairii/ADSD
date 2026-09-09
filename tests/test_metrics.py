"""CPU-only checks for scientific aggregation and portable cache configuration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/inference'))
import run


class MetricsTests(unittest.TestCase):
    def test_global_ratios_differ_from_request_means(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for index, (tokens, calls, drafts, accepted, seconds) in enumerate([
                (4, 1, 3, 3, 2), (6, 3, 9, 3, 6)
            ]):
                (path / f'row_{index:04d}.json').write_text(json.dumps({
                    'generated_ids': list(range(tokens)), 'elapsed_seconds': seconds,
                    'correct_last_number': index == 0, 'reached_cap': False,
                    'counts': {'sample_length': [tokens], 'total_step': [calls],
                               'draft_eval': [drafts], 'accepted_draft_tokens': [accepted]}
                }))
            (path / 'stdout.log').write_text('num_q 2 correct 0 ratio 0.0\n')
            summary = run.summarize(path)
            self.assertEqual(summary['block_efficiency_global'], 2.5)
            self.assertEqual(summary['block_efficiency_mean'], 3.0)
            self.assertEqual(summary['aggregate_acceptance_global'], 0.5)
            self.assertAlmostEqual(summary['aggregate_acceptance_mean'], 2/3)
            self.assertEqual(summary['throughput_global'], 1.25)
            self.assertEqual(summary['accuracy'], 0.5)
            self.assertEqual(summary['native_parser']['accuracy'], 0.0)

    def test_impossible_token_accounting_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'row_0000.json').write_text(json.dumps({
                'generated_ids': [1], 'elapsed_seconds': 1,
                'counts': {'sample_length': [2]}
            }))
            with self.assertRaisesRegex(ValueError, 'token accounting'):
                run.summarize(path)

    def test_external_cache_and_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(cache_root=root/'scratch', output=root/'results'/'trial',
                                   runtime=root/'runtime', train=root/'train.jsonl', test=root/'test.jsonl')
            env = run.environment(args)
            self.assertTrue(env['HF_HOME'].startswith(str(root/'scratch'/'trial')))
            self.assertEqual(env['ADSD_TEST_JSONL'], str(root/'test.jsonl'))
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
