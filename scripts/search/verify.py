#!/usr/bin/env python3
"""Compare a search result with the default inference suffix."""
import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--default', type=Path, default=root / 'configs/suffix.json')
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    candidate = result.get('best_projected', result)
    expected = json.loads(args.default.read_text())
    for field in ('suffix_text', 'suffix_token_ids'):
        if field not in candidate or field not in expected:
            raise SystemExit(f'Missing required field: {field}')
        if candidate[field] != expected[field]:
            raise SystemExit(f'Suffix mismatch: {field}')
    print('MATCH: suffix text and token IDs equal the default inference suffix.')


if __name__ == '__main__':
    main()
