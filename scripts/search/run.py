#!/usr/bin/env python3
"""Run ADSD suffix search with the bundled configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/inference'))
import run as inference


def build_command(args):
    return [sys.executable, str(ROOT / 'scripts/search/entry.py'),
            '--config', str((ROOT / 'configs/search.json').resolve()),
            '--target-model', str(args.target.resolve()),
            '--draft-model', str(args.draft.resolve()),
            '--output', str(args.output.resolve() / 'result.json'),
            '--progress-output', str(args.output.resolve() / 'progress.json')]


def search(args):
    protocol = json.loads((ROOT / 'configs/protocol.json').read_text())
    for split in ('train', 'test'):
        if inference.digest(getattr(args, split)) != protocol['dataset'][split + '_sha256']:
            raise ValueError(f'{split} data differs from the pinned GSM8K input; run prepare_data.py')
    inference.provenance(args)
    try:
        manifest_path = args.output / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        expected_models = json.loads((ROOT / 'configs/model_hashes.json').read_text())
        if manifest['model_files'] != expected_models:
            raise ValueError('Model files differ from the pinned snapshots in configs/model_hashes.json')
        files = [ROOT / 'configs/search.json', ROOT / 'configs/protocol.json',
                 ROOT / 'external/hsd/chain-of-thought-hub/gsm8k/lib_prompt/prompt_original.txt']
        for directory in ('scripts/search', 'scripts/prompt_search', 'src/sd_robustness'):
            files.extend((ROOT / directory).glob('*.py'))
        manifest['source_file_sha256'].update(
            {str(p.relative_to(ROOT)): inference.digest(p) for p in files})
        inference.atomic(manifest_path, manifest)
        command = build_command(args)
        inference.atomic(args.output / 'command.json', command)
        env = inference.environment(args)
        env['PYTHONPATH'] += os.pathsep + str(ROOT / 'src')
        with (args.output / 'stdout.log').open('w') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Search exited with code {completed.returncode}; see stdout.log')
        result = json.loads((args.output / 'result.json').read_text())
        if result.get('deployment_text_roundtrip_valid') is not True:
            raise ValueError('Search output text does not round-trip to its token IDs')
        best = result['best_projected']
        suffix = {key: best[key] for key in ('suffix_text', 'suffix_token_ids')}
        inference.atomic(args.output / 'suffix.json', suffix)
        inference.atomic(args.output / 'status.json', {'status': 'complete', 'exit_code': 0})
        print(f'Search complete: {args.output / "suffix.json"}')
    except Exception as exc:
        inference.atomic(args.output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'output', 'target', 'draft', 'train', 'test'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, help='Defaults to ADSD_CACHE_ROOT')
    args = parser.parse_args()
    args.attack = None
    search(args)


if __name__ == '__main__':
    main()
