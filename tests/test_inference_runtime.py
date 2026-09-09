"""Check the assembled inference runtime without loading GPU models."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/inference'))
import run


class InferenceRuntimeTests(unittest.TestCase):
    def test_prepare_keeps_evaluator_and_required_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'transformers'
            package.mkdir()
            (package / '__init__.py').write_text('')
            distribution = SimpleNamespace(locate_file=lambda _: package)
            with patch('importlib.metadata.distribution', return_value=distribution), \
                 patch('importlib.metadata.version', return_value='4.46.3'):
                run.prepare(SimpleNamespace(runtime=root / 'runtime'))
            evaluator = root / 'runtime/workspace/chain-of-thought-hub/gsm8k/eval_speculative_decoding_llm.py'
            compile(evaluator.read_text(), str(evaluator), 'exec')
            protocol = json.loads((ROOT / 'configs/protocol.json').read_text())
            self.assertEqual(hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                             protocol['runtime']['evaluator_sha256'])
            for name in ('prompt_original.txt', 'prompt_hardest.txt', 'validation_index.npy'):
                self.assertTrue((evaluator.parent / 'lib_prompt' / name).is_file())
            for path in (root / 'runtime/overlay').rglob('*.py'):
                compile(path.read_text(), str(path), 'exec')

    def test_fixed_suffix_reaches_inference_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'runtime/workspace/chain-of-thought-hub/gsm8k'
            (base / 'lib_prompt').mkdir(parents=True)
            (base / 'eval_speculative_decoding_llm.py').write_text('')
            output = root / 'output'
            output.mkdir()
            args = SimpleNamespace(runtime=root / 'runtime', output=output,
                target=root / 'target', draft=root / 'draft', samples=1,
                cap=512, sample_offset=0, seed=2027, mode='tokenwise',
                attack=ROOT / 'configs/suffix.json')

            def generated(command, **kwargs):
                suffix = json.loads(args.attack.read_text())['suffix_text']
                self.assertEqual(command[command.index('--prompt-suffix') + 1], suffix)
                self.assertIn('--speculative', command)
                (output / 'row_0000.json').write_text(json.dumps({
                    'generated_ids': [1, 2], 'elapsed_seconds': 1,
                    'correct_last_number': True, 'reached_cap': False,
                    'counts': {'sample_length': [2], 'total_step': [1],
                               'draft_eval': [1], 'accepted_draft_tokens': [1]}}))
                kwargs['stdout'].write('num_q 1 correct 1 ratio 1.0\n')
                return SimpleNamespace(returncode=0)

            with patch.object(run, 'provenance'), patch.object(run, 'environment', return_value={}), \
                 patch.object(run.subprocess, 'run', side_effect=generated):
                run.evaluate(args)
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
