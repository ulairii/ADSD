#!/usr/bin/env python3
"""Summarize paired ADSD inference over the 263-question benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, native_correct):
    seconds = sum(r['elapsed_seconds'] for r in rows)
    tokens = sum(len(r['generated_ids']) for r in rows)
    calls = sum(sum(r['counts']['total_step']) for r in rows)
    drafted = sum(sum(r['counts']['draft_eval']) for r in rows)
    accepted = sum(sum(r['counts']['accepted_draft_tokens']) for r in rows)
    kept = kept_tokens = speed_numerator_tokens = 0
    weighted_seconds = 0.0
    for row in rows:
        c = row['counts']
        sizes = c['sample_length']
        require(len(sizes) == len(c['draft_eval']), 'Inconsistent native block lists')
        require(sum(sizes) == len(row['generated_ids']), 'Generated token accounting mismatch')
        require(row['elapsed_seconds'] > 0, 'Nonpositive latency')
        require(0 <= sum(c['accepted_draft_tokens']) <= sum(c['draft_eval']), 'Impossible acceptance')
        valid = [s for d, s in zip(c['draft_eval'], sizes) if d == 10]
        if valid:
            kept += len(valid)
            kept_tokens += sum(valid)
            speed_numerator_tokens += sum(sizes)
            weighted_seconds += row['elapsed_seconds'] * sum(valid) / sum(sizes)
    require(seconds > 0 and calls > 0 and kept > 0 and weighted_seconds > 0, 'Insufficient counters')
    return {
        'samples': len(rows), 'native_accuracy': native_correct / len(rows),
        'continuation_last_number_accuracy': sum(r['correct_last_number'] for r in rows) / len(rows),
        'latency_seconds': seconds / len(rows),
        'block_efficiency_gamma_matched': kept_tokens / kept,
        'block_efficiency_all_blocks': tokens / calls,
        'speed_gamma_scaled': speed_numerator_tokens / weighted_seconds * 10,
        'throughput_real_tokens_per_second': tokens / seconds,
        'acceptance_global': accepted / drafted,
        'generated_tokens_mean': tokens / len(rows),
        'capped_requests': sum(r['reached_cap'] for r in rows),
        'gamma_matched_blocks': kept,
    }


def collect(run_root, protocol, runtime_manifest=None):
    expected_models = json.loads((ROOT / 'configs/model_hashes.json').read_text())
    sample_manifest = json.loads((ROOT / 'configs/samples.json').read_text())
    runtime_hash = protocol['runtime']['executed_manifest_sha256']
    if runtime_manifest:
        value = json.loads(runtime_manifest.read_text())
        content = {k: value[k] for k in ('transformers_version', 'evaluator_sha256', 'overlay_hashes')}
        fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        require(fingerprint == protocol['runtime']['content_sha256'], 'Different runtime content')
        runtime_hash = digest(runtime_manifest)
    records, outputs = [], {}
    for condition, parity in [('benign', 0), ('adsd', 1)]:
        offset = protocol['evaluation']['offset']
        count = protocol['evaluation']['samples']
        seed = protocol['evaluation']['seed']
        run = run_root / condition
        status = json.loads((run / 'status.json').read_text())
        require(status['status'] == 'complete' and status['exit_code'] == 0, f'Incomplete {run}')
        manifest = json.loads((run / 'manifest.json').read_text())
        require(manifest['runtime_manifest_sha256'] == runtime_hash, 'Unexpected runtime manifest')
        args = manifest['args']
        for key, value in {'samples': count, 'sample_offset': offset, 'seed': seed,
                           'cap': 512, 'mode': 'tokenwise'}.items():
            require(args[key] == value, f'Unexpected {key} in {run}')
        for split in ('train', 'test'):
            require(manifest['dataset_sha256'][split] == protocol['dataset'][split + '_sha256'],
                    f'Wrong {split} dataset')
        for role in ('target', 'draft'):
            require(manifest['model_files'][role] == expected_models[role], f'Wrong {role} bytes')
        require(bool(args['attack']) == bool(parity), 'Wrong attack condition')
        if parity:
            require(manifest['attack_sha256'] == digest(ROOT / 'configs/suffix.json'),
                    'Unexpected attack artifact')
        paths = sorted(run.glob('row_*.json'))
        chunk = [json.loads(p.read_text()) for p in paths]
        require([r['index'] for r in chunk] == list(range(offset, offset + count)), 'Missing/duplicate rows')
        for row in chunk:
            require(hashlib.sha256(row['question'].encode()).hexdigest() == sample_manifest['question_sha256'][row['index']],
                    'Question differs from frozen sample list')
        policy = json.loads((run / 'effective_policy.json').read_text())
        for key, value in {'gamma': 10, 'temperature': 1.0, 'max_new_tokens': 512,
                           'num_samples': count, 'sample_offset': offset, 'seed': seed}.items():
            require(policy['args'][key] == value, f'Wrong effective policy: {key}')
        for role in ('model1', 'model2'):
            require(policy['models'][role]['dtype'] == 'torch.float16', 'Wrong precision')
            require(policy['models'][role]['attention'] == 'sdpa', 'Wrong attention kernel')
        summary = json.loads((run / 'summary.json').read_text())
        require(summary['native_parser']['samples'] == count, 'Native parser count mismatch')
        native_correct = summary['native_parser']['correct']
        rows = chunk
        records.append({'run': str(run), 'manifest': manifest, 'status': status,
                        'summary': summary, 'effective_policy': policy,
                        'row_sha256': {p.name: digest(p) for p in paths}})
        outputs[condition] = metrics(rows, native_correct)
    return outputs, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, help='Portable path remapping; content must match pinned runtime')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'configs/protocol.json').read_text())
    try:
        results, records = collect(args.run_root, protocol, args.runtime_manifest)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit('Cannot summarize evaluation: ' + str(e))
    benign, attack = results['benign'], results['adsd']
    ratio = attack['latency_seconds'] / benign['latency_seconds']
    result = {
        'samples': protocol['evaluation']['samples'],
        'seed': protocol['evaluation']['seed'],
        'results': results,
        'latency_ratio': ratio,
        'slowdown_percent': (ratio - 1) * 100,
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
