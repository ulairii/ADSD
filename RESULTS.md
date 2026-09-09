# Results

The fixed suffix in `configs/suffix.json` was evaluated on the first 263 GSM8K test questions using Qwen2.5-14B/0.5B Instruct-GPTQ-Int8 and NVIDIA A100 80GB PCIe GPUs.

| Metric | Benign | ADSD |
| --- | ---: | ---: |
| Questions | 263 | 263 |
| Mean latency (s) | 26.3845 | 42.7716 |
| Block efficiency | 6.0209 | 3.5845 |
| All-block efficiency | 5.9579 | 3.5568 |
| Accepted / drafted tokens | 50.43% | 26.07% |
| Accuracy (native parser) | 82.13% | 87.07% |
| Mean generated tokens | 205.17 | 204.77 |
| Throughput (tokens/s) | 7.7763 | 4.7875 |

The mean latency ratio is **1.6211**, corresponding to a **62.11%** increase.

The settings for these saved measurements are included in [results/gsm8k.json](results/gsm8k.json).

## Metrics

Mean latency averages per-request generation time. Block efficiency uses blocks with 10 drafted tokens, matching the benchmark convention. All-block efficiency divides total generated tokens by total target calls. Acceptance divides accepted draft tokens by drafted tokens. Accuracy uses the native GSM8K answer parser. Throughput divides total generated tokens by total generation time.

[Machine-readable results](results/gsm8k.json) preserve the exact metric values. [Per-question measurements](results/gsm8k_measurements.csv) contain the paired measurements. Exact model and dataset revisions are recorded in [the inference protocol](configs/protocol.json).
