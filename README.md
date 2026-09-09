# ADSD

[![arXiv](https://img.shields.io/badge/arXiv-2607.21804-b31b1b.svg)](https://arxiv.org/abs/2607.21804)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Project%20Page-FFD21E.svg)](https://huggingface.co/spaces/Ulairi/ADSD)

Suffix search, inference and evaluation for Adversarial Prompts for Acceptance Collapse in Speculative Decoding.

Run benign and fixed-suffix speculative decoding with Qwen2.5-14B and Qwen2.5-0.5B on GSM8K. The default suffix is stored in [configs/suffix.json](configs/suffix.json).

## Project status

- [x] GSM8K data preparation and verification.
- [x] Runtime preparation and benign inference.
- [x] ADSD inference with the provided suffix.
- [x] Paired benign/ADSD evaluation and metric reporting.
- [x] CPU tests for runtime preparation, launching and metrics.
- [x] Discrete suffix search with the provided configuration.

## Installation

Run the commands below from the repository root, using Python 3.10 and CUDA-enabled PyTorch. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The measured environment is recorded in [configs/environment.freeze.txt](configs/environment.freeze.txt). Model and dataset revisions are in [configs/protocol.json](configs/protocol.json).

## Prepare data

We use GSM8K (`main`), following [HSD](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding). Prepare the data in a directory outside the repository:

```bash
export ADSD_DATA_DIR=/path/to/datasets/gsm8k
python scripts/prepare_data.py --data-dir "$ADSD_DATA_DIR"
export ADSD_TRAIN="$ADSD_DATA_DIR/train.jsonl"
export ADSD_TEST="$ADSD_DATA_DIR/test.jsonl"
```

The script downloads and verifies the data, reusing existing files. The evaluator selects the first 263 test questions automatically.

## Prepare models

Download the pinned model snapshots into an external cache:

```bash
export ADSD_CACHE_ROOT=/path/to/cache
export ADSD_TARGET="$(hf download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 \
  --revision 801d1799b049765aebe175acb5d529c372e7b1b9 \
  --cache-dir "$ADSD_CACHE_ROOT/huggingface/hub" --quiet)"
export ADSD_DRAFT="$(hf download Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8 \
  --revision c68601e5424e69cdaa6e073673e3c94db27b4397 \
  --cache-dir "$ADSD_CACHE_ROOT/huggingface/hub" --quiet)"
```

If these snapshots are already available locally, set `ADSD_TARGET` and `ADSD_DRAFT` to their directories instead.

## Run

After preparing the data and models, set the runtime and output directories:

```bash
export ADSD_PYTHON=python
export ADSD_RUNTIME=/path/to/runtime
export ADSD_OUTPUT=/path/to/runs

python scripts/inference/run.py prepare --runtime "$ADSD_RUNTIME"
bash scripts/run_inference.sh
```

`prepare` creates the HSD runtime from the installed Transformers package. Use a new runtime directory for this step.

The command runs all 263 test questions once for benign inference and once with the ADSD suffix, then saves the comparison to `$ADSD_OUTPUT/comparison.json`. Use a new output directory for each run. Decoding settings are in [configs/protocol.json](configs/protocol.json).

For a smaller inference run:

```bash
python scripts/inference/run.py evaluate \
  --runtime "$ADSD_RUNTIME" --target "$ADSD_TARGET" --draft "$ADSD_DRAFT" \
  --train "$ADSD_TRAIN" --test "$ADSD_TEST" \
  --attack configs/suffix.json --samples 10 --seed 2027 \
  --output "$ADSD_OUTPUT/adsd_10"
```

Omit `--attack` for benign inference. Each run saves generated text, token counters, timing, configuration and summary metrics. The full launcher writes `comparison.json` with paired metrics, validates the inputs and runtime, and records the GPU configuration.

## Suffix search

Use the same environment, data and prepared runtime as inference:

```bash
bash scripts/run_search.sh
python scripts/search/verify.py --result "$ADSD_OUTPUT/search/result.json"
```

The search settings are fixed in [configs/search.json](configs/search.json): text initialization, seed 0, and five sweeps over a five-token suffix (25 position updates). The completed search recovered the default suffix with identical text and token IDs; its saved result is in [results/search.json](results/search.json).

Each run writes `result.json`, `progress.json`, `stdout.log` and an inference-ready `suffix.json` under `$ADSD_OUTPUT/search`. Use a new output directory when restarting a search. To evaluate the searched suffix, pass `--attack "$ADSD_OUTPUT/search/suffix.json"` to the inference command above. The verification command checks equality with the bundled default suffix.

## Results

| Metric | Benign | ADSD |
| --- | ---: | ---: |
| Mean latency (s) | 26.38 | 42.77 |
| Block efficiency | 6.021 | 3.585 |
| Accuracy | 82.13% | 87.07% |

Mean latency increases by **62.11%** over the complete 263-question evaluation. See [RESULTS.md](RESULTS.md) for metric definitions and saved measurements.

## Tests

```bash
python -m unittest discover -s tests -v
```

Third-party source attribution and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use ADSD in your research, please cite [Adversarial Prompts for Acceptance Collapse in Speculative Decoding](https://arxiv.org/abs/2607.21804):

```bibtex
@misc{wang2026adversarialpromptsacceptancecollapse,
  title         = {Adversarial Prompts for Acceptance Collapse in Speculative Decoding},
  author        = {Run Wang and Chaoyi Zhou and Xi Liu and Yi Zhu and Amir Salarpour and Pedram MohajerAnsari and Zhi-Qi Cheng and Feng Luo and Siyu Huang and Mert D. Pes{\'e}},
  year          = {2026},
  eprint        = {2607.21804},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2607.21804}
}
```
