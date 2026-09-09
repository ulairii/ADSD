"""Check search packaging, inference export and failed-result handling."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adsd_search_runner', ROOT / 'scripts/search/run.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


class SearchPipelineTests(unittest.TestCase):
    def exercise_search(self, roundtrip=True, returncode=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(**{k: root / k for k in
                ('runtime', 'output', 'target', 'draft', 'train', 'test')}, attack=None, cache_root=root / 'cache')
            protocol = json.loads((ROOT / 'configs/protocol.json').read_text())
            digest = search.inference.digest

            def input_digest(path):
                for split in ('train', 'test'):
                    if Path(path) == getattr(args, split):
                        return protocol['dataset'][split + '_sha256']
                return digest(path)

            def provenance(_):
                args.output.mkdir()
                search.inference.atomic(args.output / 'manifest.json', {
                    'model_files': json.loads((ROOT / 'configs/model_hashes.json').read_text()),
                    'source_file_sha256': {}})

            def execute(command, **kwargs):
                self.assertIn(str(ROOT / 'src'), kwargs['env']['PYTHONPATH'])
                self.assertEqual(kwargs['env']['ADSD_TRAIN_JSONL'], str(args.train))
                self.assertEqual(command[command.index('--config') + 1], str(ROOT / 'configs/search.json'))
                result = json.loads((ROOT / 'results/search.json').read_text())
                result['deployment_text_roundtrip_valid'] = roundtrip
                search.inference.atomic(args.output / 'result.json', result)
                return SimpleNamespace(returncode=returncode)

            with patch.object(search.inference, 'digest', side_effect=input_digest), \
                 patch.object(search.inference, 'provenance', side_effect=provenance), \
                 patch.object(search.subprocess, 'run', side_effect=execute):
                if not roundtrip or returncode:
                    with self.assertRaises((ValueError, RuntimeError)):
                        search.search(args)
                    self.assertFalse((args.output / 'suffix.json').exists())
                    self.assertEqual(json.loads((args.output / 'status.json').read_text())['status'], 'failed')
                else:
                    search.search(args)
                    self.assertEqual(json.loads((args.output / 'suffix.json').read_text()),
                                     json.loads((ROOT / 'configs/suffix.json').read_text()))
                    self.assertEqual(json.loads((args.output / 'status.json').read_text())['status'], 'complete')

    def test_export_inference_suffix(self):
        self.exercise_search()

    def test_reject_non_roundtripping_result(self):
        self.exercise_search(roundtrip=False)

    def test_failed_search_does_not_export_suffix(self):
        self.exercise_search(returncode=1)

    def test_verifier_rejects_different_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            result = json.loads((ROOT / 'results/search.json').read_text())
            result['best_projected']['suffix_token_ids'][0] += 1
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(result))
            completed = subprocess.run([sys.executable, str(ROOT / 'scripts/search/verify.py'),
                                        '--result', str(path)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('suffix_token_ids', completed.stderr)
