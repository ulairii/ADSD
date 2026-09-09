"""Save per-request outputs and decoding configuration after generation."""
import hashlib
import json
import os
from pathlib import Path
import re


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def capture(ns):
    root = Path(os.environ['ADSD_RUN_OUTPUT'])
    index = ns['args'].sample_offset + ns['progress'] - 1
    ids = ns['outputs'][0].detach().cpu().tolist()
    prompt = ns['input_ids'][0].detach().cpu().tolist()
    continuation = ids[len(prompt):]
    text = ns['tokenizer1'].decode(continuation, skip_special_tokens=True)
    def last_number(value):
        matches = re.findall(r'-?\d+(?:\.\d+)?', value.replace(',', ''))
        return matches[-1] if matches else None
    counts = ns.get('counts', {}) if ns['args'].speculative else {}
    row = dict(index=index, question=ns['q'], reference=ns['a'],
               prompt_ids=prompt, generated_ids=continuation, generated_text=text,
               elapsed_seconds=ns['end']-ns['start'], counts=counts,
               correct_last_number=last_number(text) == last_number(ns['a']),
               reached_cap=len(continuation) >= ns['args'].max_new_tokens,
               prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest())
    atomic(root / f'row_{index:04d}.json', row)
    atomic(root / 'progress.json', dict(status='running', completed=ns['progress'],
                                       expected=ns['num_samples']))


def policy(ns):
    root = Path(os.environ['ADSD_RUN_OUTPUT'])
    import torch
    import transformers
    import signal
    def interrupted(signum, _frame):
        atomic(root / 'interrupted.json', {'signal': signum,
               'completed_rows': len(list(root.glob('row_*.json'))),
               'recovery': 'Retain partial rows; restart in a new output directory.'})
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    atomic(root / 'effective_policy.json', {
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'models': {name: dict(dtype=str(ns[name].dtype),
                            attention=ns[name].config._attn_implementation,
                            generation_config=ns[name].generation_config.to_dict())
                   for name in ('model1', 'model2')},
        'args': vars(ns['args'])})
