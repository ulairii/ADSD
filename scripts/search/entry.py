"""Load local GSM8K data and run discrete suffix search."""
import json
import os
from pathlib import Path
import runpy
import sys
import datasets

original_load = datasets.load_dataset


def load_dataset(name, *args, **kwargs):
    if name != 'gsm8k':
        return original_load(name, *args, **kwargs)
    split = kwargs.get('split', 'train')
    path = os.environ['ADSD_' + split.upper() + '_JSONL']
    return datasets.Dataset.from_list([json.loads(line) for line in open(path) if line.strip()])


datasets.load_dataset = load_dataset
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts/prompt_search'))
runpy.run_path(str(root / 'scripts/prompt_search/discrete_search.py'), run_name='__main__')
