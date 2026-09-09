#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ADSD_PYTHON:-python}"
: "${ADSD_TARGET:?}" "${ADSD_DRAFT:?}" "${ADSD_TRAIN:?}" "${ADSD_TEST:?}"
: "${ADSD_RUNTIME:?}" "${ADSD_OUTPUT:?}" "${ADSD_CACHE_ROOT:?}"
[[ $# -eq 0 ]] || { echo "Usage: bash scripts/run_inference.sh" >&2; exit 2; }
common=(--runtime "$ADSD_RUNTIME" --target "$ADSD_TARGET" --draft "$ADSD_DRAFT"
        --train "$ADSD_TRAIN" --test "$ADSD_TEST" --samples 263 --cap 512 --seed 2027)

"$python_bin" "$root/scripts/inference/run.py" evaluate "${common[@]}" \
    --output "$ADSD_OUTPUT/benign"
"$python_bin" "$root/scripts/inference/run.py" evaluate "${common[@]}" \
    --attack "$root/configs/suffix.json" --output "$ADSD_OUTPUT/adsd"
"$python_bin" "$root/scripts/inference/report.py" --run-root "$ADSD_OUTPUT" \
    --runtime-manifest "$ADSD_RUNTIME/manifest.json" --output "$ADSD_OUTPUT/comparison.json"
