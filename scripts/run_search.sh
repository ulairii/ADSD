#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ADSD_PYTHON:-python}"
: "${ADSD_TARGET:?}" "${ADSD_DRAFT:?}" "${ADSD_TRAIN:?}" "${ADSD_TEST:?}"
: "${ADSD_RUNTIME:?}" "${ADSD_OUTPUT:?}" "${ADSD_CACHE_ROOT:?}"
[[ $# -eq 0 ]] || { echo "Usage: bash scripts/run_search.sh" >&2; exit 2; }
"$python_bin" "$root/scripts/search/run.py" \
    --runtime "$ADSD_RUNTIME" --target "$ADSD_TARGET" --draft "$ADSD_DRAFT" \
    --train "$ADSD_TRAIN" --test "$ADSD_TEST" --output "$ADSD_OUTPUT/search"
