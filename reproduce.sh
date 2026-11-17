#!/usr/bin/env bash
# Rebuild every version (baseline, v1-v11), score them, and build the comparison site.
#
#   SOCCER_PY=/path/to/soccer/bin/python PATHCRF_PY=/path/to/pathcrf/bin/python ./reproduce.sh
#
# Defaults assume conda envs named `soccer` and `pathcrf` (see environments/). Needs the match data in
# ./data (or MTAH_DATA_DIR), ffmpeg on PATH, and `git submodule update --init` for PathCRF.
set -euo pipefail
cd "$(dirname "$0")"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
SOCCER_PY="${SOCCER_PY:-$CONDA_BASE/envs/soccer/bin/python}"
PATHCRF_PY="${PATHCRF_PY:-$CONDA_BASE/envs/pathcrf/bin/python}"
DEVICE="${PATHCRF_DEVICE:-cuda:0}"

step() { echo; echo "=== $*"; }

step "baseline";            "$SOCCER_PY" starter/baseline.py
step "v1 v2 v3";            for v in v1 v2 v3; do "$SOCCER_PY" solution/$v.py; done
step "PathCRF ($DEVICE)";   "$PATHCRF_PY" solution/pathcrf_adapter.py --device "$DEVICE"
step "v4a v4b";             "$SOCCER_PY" solution/v4.py
step "v5 v6";               for v in v5 v6; do "$SOCCER_PY" solution/$v.py; done
step "whistle detection";   "$SOCCER_PY" solution/whistle.py
step "v7 .. v11";           for v in v7 v8 v9 v10 v11; do "$SOCCER_PY" solution/$v.py; done
step "submission (v11)";    mkdir -p submission && cp evaluation/results/v11.csv submission/Soma_AI.csv
step "validate submission"; "$SOCCER_PY" evaluation/validate_submission.py submission/Soma_AI.csv
step "score (public 5 min)"; "$SOCCER_PY" evaluation/run_evaluation.py --no-detail
step "comparison site";     "$SOCCER_PY" tools/make_compare_site.py
echo
echo "Done. Serve the site with:"
echo "  $SOCCER_PY -m http.server 8790 --bind 127.0.0.1 --directory data/artifact/compare"
