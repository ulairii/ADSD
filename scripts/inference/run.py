#!/usr/bin/env python3
"""ADSD inference with the HSD speculative decoding engine."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_hooks import atomic


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location('workspace_tools', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare(args):
    """Copy the installed package; never patch a shared Python installation."""
    import importlib.metadata
    installed = Path(importlib.metadata.distribution('transformers').locate_file('transformers'))
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=False)
    overlay = runtime / 'overlay' / 'transformers'
    shutil.copytree(installed, overlay, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    helper = module(ROOT / 'scripts/eval/internal/prepare_workspace.py')
    source = ROOT / 'external/hsd'
    workspace = runtime / 'workspace'
    helper.copy_workspace(source, workspace)
    evaluator = workspace / 'chain-of-thought-hub/gsm8k/eval_speculative_decoding_llm.py'
    helper.patch_eval_script(evaluator)
    helper.patch_generation_utils(workspace / 'transformers/generation/utils.py')
    helper.ensure_outputs_dir(workspace)
    for src in (workspace / 'transformers').rglob('*.py'):
        dest = overlay / src.relative_to(workspace / 'transformers')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text().expandtabs(4))
    text = evaluator.read_text()
    # A snapshot path's basename is a revision hash, not its model name.
    text = text.replace('model_ref = os.path.basename(model2_name.rstrip("/"))',
                        'model_ref = model2_name')
    text = text.replace("gsm8k = load_dataset('gsm8k', 'main')", '''from datasets import Dataset, DatasetDict
gsm8k = DatasetDict({split: Dataset.from_list([
    json.loads(line) for line in open(os.environ['ADSD_' + split.upper() + '_JSONL'])
    if line.strip()]) for split in ('train', 'test')})''')
    marker = '            end = time.time()\n'
    require(text.count(marker) == 1, 'Expected one evaluator insertion point')
    text = text.replace(marker, marker + '            from native_hooks import capture\n            capture(globals())\n')
    marker = '    print("start training")'
    require(text.count(marker) == 1, 'Expected one evaluator insertion point')
    text = text.replace(marker, '    from native_hooks import policy\n    policy(globals())\n' + marker)
    evaluator.write_text(text)
    atomic(runtime / 'manifest.json', {
        'transformers_version': importlib.metadata.version('transformers'),
        'base_package': str(installed),
        'evaluator_sha256': digest(evaluator),
        'overlay_hashes': {str(p.relative_to(overlay)): digest(p)
                           for p in overlay.rglob('*.py')},
        'note': 'Transformers runtime with HSD inference components.'})


def environment(args):
    env = os.environ.copy()
    cache_base = getattr(args, 'cache_root', None) or os.environ.get('ADSD_CACHE_ROOT')
    cache = Path(cache_base).expanduser().resolve() / args.output.name if cache_base else args.output.resolve() / 'cache'
    for key in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
                'HF_DATASETS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME',
                'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'SHARED_CACHE_ROOT'):
        env[key] = str(cache / key.lower())
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED'):
        env[key] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env.pop('ADSD_ACCEPTANCE_FALLBACK_THRESHOLD', None)
    env['PYTHONPATH'] = os.pathsep.join([str(args.runtime.resolve() / 'overlay'),
                                       str(HERE), str(ROOT)])
    env['ADSD_TRAIN_JSONL'] = str(args.train.resolve())
    env['ADSD_TEST_JSONL'] = str(args.test.resolve())
    env['ADSD_RUN_OUTPUT'] = str(args.output.resolve())
    return env


def provenance(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if not (args.runtime / 'manifest.json').is_file():
        raise ValueError('Run prepare first with this Python environment')
    manifest = {'command': sys.argv, 'args': {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
                'runtime_manifest_sha256': digest(args.runtime / 'manifest.json'),
                'dataset_sha256': {split: digest(getattr(args, split)) for split in ('train', 'test')},
                'model_files': {}}
    for role in ('target', 'draft'):
        path = getattr(args, role).resolve()
        if not (path / 'config.json').is_file():
            raise ValueError(f'cache preload incomplete: {path}')
        required = list(path.glob('*.safetensors'))
        if not required:
            raise ValueError(f'cache preload incomplete: no safetensors in {path}')
        manifest['model_files'][role] = {
            p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(path.iterdir()) if p.is_file() and not p.name.startswith('.')}
    source_paths = [HERE/'run.py', HERE/'native_hooks.py',
                    ROOT/'scripts/eval/internal/prepare_workspace.py']
    manifest['source_file_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in source_paths if p.exists()}
    manifest['source_commit_hint'] = os.environ.get('ADSD_SOURCE_COMMIT')
    if args.attack:
        manifest['attack_sha256'] = digest(args.attack)
    manifest['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                               '--format=csv,noheader'], text=True).strip()
    atomic(args.output / 'manifest.json', manifest)


def evaluate(args):
    provenance(args)
    base = args.runtime.resolve() / 'workspace/chain-of-thought-hub/gsm8k'
    work = args.output / 'native'
    work.mkdir()
    shutil.copytree(base / 'lib_prompt', work / 'lib_prompt')
    shutil.copy2(base / 'eval_speculative_decoding_llm.py', work)
    (work / 'outputs').mkdir()
    suffix = ''
    if args.attack:
        best = json.loads(args.attack.read_text())
        suffix = best.get('suffix_text')
        if not isinstance(suffix, str) or not suffix:
            raise ValueError('Attack artifact lacks nonempty suffix/prefix text')
    command = [sys.executable, str(work / 'eval_speculative_decoding_llm.py'),
               '--target-model', str(args.target.resolve()), '--draft-model', str(args.draft.resolve()),
               '--num-samples', str(args.samples), '--max-new-tokens', str(args.cap),
               '--sample-offset', str(args.sample_offset),
               '--seed', str(args.seed), '--gamma', '10', '--temperature', '1.0',
               '--prompt-tag', 'inference']
    if args.mode == 'tokenwise':
        command.append('--speculative')
    if suffix:
        command += ['--prompt-suffix', suffix]
    atomic(args.output / 'command.json', command)
    with (args.output / 'stdout.log').open('w') as log:
        completed = subprocess.run(command, cwd=work, env=environment(args), stdout=log,
                                   stderr=subprocess.STDOUT)
    rows = sorted(args.output.glob('row_*.json'))
    ok = completed.returncode == 0 and len(rows) == args.samples
    if not ok:
        atomic(args.output / 'status.json', {'status': 'failed',
            'exit_code': completed.returncode, 'rows': len(rows)})
        raise SystemExit(completed.returncode or 1)
    try:
        summarize(args.output)
    except Exception as exc:
        atomic(args.output / 'status.json', {'status': 'failed', 'stage': 'summarization',
            'error': str(exc), 'exit_code': 1, 'rows': len(rows)})
        raise
    atomic(args.output / 'progress.json', {'status': 'complete', 'completed': len(rows),
                                         'expected': args.samples})
    atomic(args.output / 'status.json', {'status': 'complete', 'exit_code': 0, 'rows': len(rows)})


def summarize(path):
    rows = [json.loads(p.read_text()) for p in sorted(path.glob('row_*.json'))]
    def total(row, field):
        return sum(row['counts'].get(field, []))
    for row in rows:
        require(row['elapsed_seconds'] > 0, 'Nonpositive generation time')
        if row['counts']:
            require(total(row, 'sample_length') == len(row['generated_ids']), 'Native token accounting mismatch')
            require(0 <= total(row, 'accepted_draft_tokens') <= total(row, 'draft_eval'), 'Invalid accepted-token accounting')
    seconds = sum(r['elapsed_seconds'] for r in rows)
    tokens = sum(len(r['generated_ids']) for r in rows)
    drafted = sum(total(r, 'draft_eval') for r in rows)
    accepted = sum(total(r, 'accepted_draft_tokens') for r in rows)
    calls = sum(total(r, 'total_step') for r in rows)
    block = [len(r['generated_ids']) / total(r, 'total_step') for r in rows if total(r, 'total_step')]
    acceptance = [total(r, 'accepted_draft_tokens') / total(r, 'draft_eval')
                  for r in rows if total(r, 'draft_eval')]
    summary = dict(samples=len(rows), correct=sum(r['correct_last_number'] for r in rows),
                   accuracy=sum(r['correct_last_number'] for r in rows)/len(rows),
                   latency_mean=seconds/len(rows), generated_tokens_mean=tokens/len(rows),
                   throughput_global=tokens/seconds, capped=sum(r['reached_cap'] for r in rows),
                   aggregate_acceptance_global=accepted/drafted if drafted else None,
                   aggregate_acceptance_mean=sum(acceptance)/len(acceptance) if acceptance else None,
                   block_efficiency_mean=sum(block)/len(block) if block else None,
                   block_efficiency_global=tokens/calls if calls else None,
                   scoring='last number of generated continuation; native full-text parser retained in stdout.log')
    import re
    native_score = re.findall(r'num_q (\d+) correct (\d+) ratio ([\d.]+)',
                             (path / 'stdout.log').read_text(errors='replace'))
    if native_score:
        n, correct, rounded = native_score[-1]
        summary['native_parser'] = dict(samples=int(n), correct=int(correct),
                                        accuracy=int(correct)/int(n), reported_rounded=float(rounded))
    atomic(path / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--runtime', type=Path, required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--cache-root', type=Path, help='Cache directory; defaults to ADSD_CACHE_ROOT')
    for name in ('runtime', 'output', 'target', 'draft', 'train', 'test'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attack', type=Path, help='Fixed suffix JSON; omit for benign inference')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--samples', type=int, default=263)
    p.add_argument('--sample-offset', type=int, default=0)
    p.add_argument('--cap', type=int, default=512)
    p.add_argument('--mode', choices=['tokenwise', 'target_only'], default='tokenwise')
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate}[args.action](args)


if __name__ == '__main__':
    main()
