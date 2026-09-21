#!/usr/bin/env bash
# Portable scene/object/grasp downloader.
#
# Streams public .tar.zst archives from Cloudflare R2 (no credentials needed)
# into a local cache dir, then symlinks them into molmospaces/assets/.
#
# Requirements:
#   - the `g1` conda env (or any env with the `molmospaces_resources` package)
#   - run from anywhere; paths are derived from this script's location
#
# Usage:
#   bash run_download.sh                          # moderate default set (~5k scenes)
#   COUNT=-1 bash run_download.sh                  # download ALL procthor+holodeck objaverse
#   CACHE_DIR=/data/cache bash run_download.sh     # store archives elsewhere
#   CONDA_ENV=myenv bash run_download.sh           # use a different conda env
set -euo pipefail

# --- repo root = two levels up from this script (molmospaces/scripts/) ---------
REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

# --- knobs (override any of these inline) -------------------------------------
CONDA_ENV="${CONDA_ENV:-g1}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/molmo-spaces-resources}"
COUNT="${COUNT:-1000}"   # per-source scene count; -1 = all, 0 = skip
export MLSPACES_EXTRACT_WORKERS="${MLSPACES_EXTRACT_WORKERS:-8}"
export MLSPACES_LINK_WORKERS="${MLSPACES_LINK_WORKERS:-64}"
export MLSPACES_SCAN_WORKERS="${MLSPACES_SCAN_WORKERS:-32}"
export PYTHONUNBUFFERED=1

# --- activate conda if available ----------------------------------------------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
python molmospaces/scripts/download_scenes.py \
  --project-root "${REPO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --procthor-objaverse-train "${COUNT}" \
  --holodeck-objaverse-train "${COUNT}" \
  --procthor-10k-train 0 --procthor-10k-val 0 --procthor-objaverse-val 0 \
  --holodeck-objaverse-val 0 --ithor 0 \
  2>&1 | tee "${REPO_ROOT}/download_scenes.log"
echo "EXIT_CODE=${PIPESTATUS[0]} ; download_scenes finished"
