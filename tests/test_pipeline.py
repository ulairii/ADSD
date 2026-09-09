"""Check full-dataset launch commands and paired report collection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/inference'))
import report


class PipelineTests(unittest.TestCase):
    def test_launcher_runs_two_complete_conditions_then_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / 'capture-python'
            interpreter.write_text(
                '#!' + sys.executable + '\n'
                'import json, os, sys\n'
                'with open(os.environ["COMMAND_LOG"], "a") as f:\n'
                '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            interpreter.chmod(0o755)
            env = os.environ.copy()
            for key in ('TARGET', 'DRAFT', 'TRAIN', 'TEST', 'RUNTIME', 'OUTPUT', 'CACHE_ROOT'):
                env['ADSD_' + key] = str(root / key.lower())
            env.update(ADSD_PYTHON=str(interpreter), COMMAND_LOG=str(root / 'commands.jsonl'))
            subprocess.run(['bash', str(ROOT / 'scripts/run_inference.sh')], env=env, check=True)
            commands = [json.loads(line) for line in (root / 'commands.jsonl').read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            config = json.loads((ROOT / 'configs/protocol.json').read_text())['evaluation']
            for command, condition in zip(commands[:2], ('benign', 'adsd')):
                self.assertEqual(command[1], 'evaluate')
                for option, expected in (('--samples', config['samples']), ('--seed', config['seed']), ('--cap', config['cap'])):
                    self.assertEqual(command[command.index(option) + 1], str(expected))
                self.assertNotIn('--sample-offset', command)
                self.assertEqual(command[command.index('--output') + 1], str(root / 'output' / condition))
            self.assertNotIn('--attack', commands[0])
            self.assertEqual(commands[1][commands[1].index('--attack') + 1], str(ROOT / 'configs/suffix.json'))
            self.assertEqual(Path(commands[2][0]).name, 'report.py')
            self.assertNotIn('--job', commands[2])

    def test_report_reads_complete_pair_and_rejects_mismatched_seed_or_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            models = {'target': {}, 'draft': {}}
            save(root / 'configs/model_hashes.json', models)
            save(root / 'configs/suffix.json', {'suffix_text': 'test'})
            save(root / 'configs/samples.json', {'question_sha256': [
                hashlib.sha256(f'Question {i}'.encode()).hexdigest() for i in range(2)]})
            protocol = {'evaluation': {'samples': 2, 'offset': 0, 'seed': 2027},
                        'runtime': {'executed_manifest_sha256': 'runtime'},
                        'dataset': {'train_sha256': 'train', 'test_sha256': 'test'}}
            for condition, elapsed in (('benign', 1.0), ('adsd', 2.0)):
                run = root / 'runs' / condition
                save(run / 'status.json', {'status': 'complete', 'exit_code': 0})
                save(run / 'manifest.json', {
                    'runtime_manifest_sha256': 'runtime', 'model_files': models,
                    'dataset_sha256': {'train': 'train', 'test': 'test'},
                    'gpu': 'NVIDIA A100 80GB',
                    'args': {'samples': 2, 'sample_offset': 0, 'seed': 2027,
                             'cap': 512, 'mode': 'tokenwise', 'attack': condition == 'adsd'},
                    'attack_sha256': report.digest(root / 'configs/suffix.json')})
                save(run / 'effective_policy.json', {
                    'args': {'gamma': 10, 'temperature': 1.0, 'max_new_tokens': 512,
                             'num_samples': 2, 'sample_offset': 0, 'seed': 2027},
                    'models': {role: {'dtype': 'torch.float16', 'attention': 'sdpa'}
                               for role in ('model1', 'model2')}})
                save(run / 'summary.json', {'native_parser': {'samples': 2, 'correct': 1}})
                for i in range(2):
                    save(run / f'row_{i:04d}.json', {
                        'index': i, 'question': f'Question {i}', 'elapsed_seconds': elapsed,
                        'generated_ids': [1, 2], 'correct_last_number': i == 0, 'reached_cap': False,
                        'counts': {'sample_length': [2], 'draft_eval': [10],
                                   'accepted_draft_tokens': [1], 'total_step': [1]}})
            with patch.object(report, 'ROOT', root):
                metrics, records = report.collect(root / 'runs', protocol)
                self.assertEqual(len(records), 2)
                self.assertEqual(metrics['benign']['samples'], 2)
                self.assertEqual(metrics['adsd']['latency_seconds'] / metrics['benign']['latency_seconds'], 2.0)
                for condition in ('benign', 'adsd'):
                    path = root / f'runs/{condition}/manifest.json'
                    manifest = json.loads(path.read_text())
                    manifest['gpu'] = 'NVIDIA H100 80GB'
                    save(path, manifest)
                other_metrics, other_records = report.collect(root / 'runs', protocol)
                self.assertEqual(other_metrics, metrics)
                self.assertTrue(all(r['manifest']['gpu'] == 'NVIDIA H100 80GB' for r in other_records))
                path = root / 'runs/adsd/manifest.json'
                manifest = json.loads(path.read_text())
                manifest['args']['seed'] = 2115
                save(path, manifest)
                with self.assertRaisesRegex(ValueError, 'seed'):
                    report.collect(root / 'runs', protocol)
                manifest['args']['seed'] = 2027
                save(path, manifest)
                path = root / 'runs/adsd/row_0001.json'
                row = json.loads(path.read_text())
                row['index'] = 0
                save(path, row)
                with self.assertRaisesRegex(ValueError, 'Missing/duplicate rows'):
                    report.collect(root / 'runs', protocol)


if __name__ == '__main__':
    unittest.main()
