#!/usr/bin/env python3
"""Prepare the GSM8K JSONL files used by ADSD inference."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def validate(content, split, protocol):
    if hashlib.sha256(content).hexdigest() != protocol[split + '_sha256']:
        raise ValueError(f'{split}.jsonl does not match the configured GSM8K version')
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    expected = {'train': 7473, 'test': 1319}[split]
    if len(rows) != expected:
        raise ValueError(f'{split}.jsonl: expected {expected} questions, got {len(rows)}')
    return len(rows)


def prepare(data_dir):
    protocol = json.loads((ROOT / 'configs/protocol.json').read_text())['dataset']
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        path = data_dir / f'{split}.jsonl'
        if path.exists():
            count = validate(path.read_bytes(), split, protocol)
        else:
            url = (f"https://raw.githubusercontent.com/{protocol['source']}/"
                   f"{protocol['revision']}/grade_school_math/data/{split}.jsonl")
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            count = validate(content, split, protocol)
            with path.open('xb') as output:
                output.write(content)
        print(f'{path}: {count} questions, verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_dir)


if __name__ == '__main__':
    main()
