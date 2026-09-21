#!/bin/bash
set -euo pipefail
# Usage: bash collect_parallel.sh [NUM_WORKERS] [EPISODES_PER_WORKER] [ENV_CONFIG] [debug]
#   ENV_CONFIG: a config name (bowl_mixed) or path (molmospaces/configs/bowl_mixed.py)


cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
ENV_CONFIG="${ENV_CONFIG:-molmospaces/configs/bowl_mixed.py}"
# Default to every GPU the machine actually has; override with GPUS=0,2,5.
DEFAULT_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - )"
DEFAULT_GPUS="${DEFAULT_GPUS:-0}"
BASE_REPO_ID="all"
# Each invocation writes into a unique timestamped dir so concurrent/repeated runs
# never clobber each other. Override with DATA_DIR=... or RUN_TAG=... if needed.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/data_${RUN_TAG}}"
SAVE_FAILURES="False"
FPS="10"
BASE_SEED="${BASE_SEED:-$(date +%s)}"

NUM_WORKERS="${1:-8}"
EPISODES="${2:-100}"
if [[ "$EPISODES" == "0" ]]; then
    EPISODES=1000000000
fi

# Remaining args: `debug` enables per-worker logs at data/_logs/<shard>.log
# (same as DEBUG=1); anything else selects the env config by name or path.
DEBUG="${DEBUG:-0}"
for arg in "${@:3}"; do
    if [[ "${arg}" == "debug" ]]; then
        DEBUG=1
    elif [[ -f "${arg}" ]]; then
        ENV_CONFIG="${arg}"
    elif [[ -f "molmospaces/configs/${arg%.py}.py" ]]; then
        ENV_CONFIG="molmospaces/configs/${arg%.py}.py"
    else
        echo "Unknown arg '${arg}' (not 'debug', not a config file, no molmospaces/configs/${arg%.py}.py)" >&2
        exit 1
    fi
done

GPUS="${GPUS:-${DEFAULT_GPUS}}"  # comma-separated GPU IDs to use, e.g. GPUS=2,3,5
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"

if [[ ! -f "${ENV_CONFIG}" ]]; then
    echo "ENV_CONFIG file not found: ${ENV_CONFIG}" >&2
    exit 1
fi
echo "Using env config: ${ENV_CONFIG}"
SCENE_GLOB="$("${PYTHON_BIN}" -c "import importlib.util as u; s=u.spec_from_file_location('c','${ENV_CONFIG}'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.get_config().scene)")"
SCENE_DIR="molmospaces/assets"
if [[ "${SCENE_GLOB}" == *.txt ]]; then
    SCENE_LIST_FILE="${SCENE_GLOB}"
    [[ -f "${SCENE_LIST_FILE}" ]] || SCENE_LIST_FILE="${SCENE_DIR}/${SCENE_GLOB}"
    SCENE_COUNT="$(grep -c '^[^#[:space:]]' "${SCENE_LIST_FILE}" 2>/dev/null || echo 0)"
else
    SCENE_COUNT="$(cd "${SCENE_DIR}" && ls ${SCENE_GLOB} 2>/dev/null | grep -vc '_ceiling\.xml$')"
fi
if [[ "${SCENE_COUNT}" -eq 0 ]]; then
    echo "No scenes matched ${SCENE_GLOB} under ${SCENE_DIR}/" >&2
    exit 1
fi
echo "Found ${SCENE_COUNT} scenes from ${SCENE_GLOB}; ${NUM_WORKERS} workers across GPUs [${GPU_LIST[*]}]."
echo "Writing this run to: ${DATA_DIR}/"

STATS_INTERVAL="${STATS_INTERVAL:-60}"
MAX_EPISODES="${MAX_EPISODES:-10000000 }"  # 0 = no cap; otherwise stop once total ≥ this

count_stats() {
    # Prints "<total_episodes> <total_frames>" summed across all shards.
    local result
    result=$( {
        find "${DATA_DIR}" -path '*/meta/info.json' 2>/dev/null \
            | xargs -I{} "${PYTHON_BIN}" -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get('total_episodes', 0), d.get('total_frames', 0))
except Exception:
    print('0 0')
" {} 2>/dev/null \
            | awk '{e+=$1; f+=$2} END {print e+0, f+0}'
    } 2>/dev/null ) || result="0 0"
    echo "${result:-0 0}"
}

PIDS=()
cleanup() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        kill -INT "${PIDS[@]}" 2>/dev/null || true
    fi
    if [[ -n "${STATS_PID:-}" ]]; then
        kill "${STATS_PID}" 2>/dev/null || true
    fi
}
trap 'echo "Stopping all workers (graceful)..."; cleanup; wait 2>/dev/null || true; exit 0' INT TERM
trap cleanup EXIT

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    seed=$((BASE_SEED * 1000 + i))
    gpu="${GPU_LIST[$((i % NUM_GPUS))]}"
    shard="$(printf "%s_run%02d" "${BASE_REPO_ID}" "${i}")"

    echo "Starting worker ${i} on GPU ${gpu} -> ${DATA_DIR}/${shard} (seed=${seed})"
    if [[ "${DEBUG}" == "1" ]]; then
        mkdir -p "${DATA_DIR}/_logs"
        log_path="${DATA_DIR}/_logs/${shard}.log"
        MUJOCO_LOG_FILE=/dev/null HF_HUB_OFFLINE=1 \
        "${PYTHON_BIN}" -u main.py \
            --gpu="${gpu}" \
            --seed="${seed}" \
            --episodes="${EPISODES}" \
            --record \
            --save_failures="${SAVE_FAILURES}" \
            --repo_id="${shard}" \
            --data_dir="${DATA_DIR}" \
            --fps="${FPS}" \
            --env="${ENV_CONFIG}" \
            >"${log_path}" 2>&1 &
    else
        MUJOCO_LOG_FILE=/dev/null HF_HUB_OFFLINE=1 \
        "${PYTHON_BIN}" -u main.py \
            --gpu="${gpu}" \
            --seed="${seed}" \
            --episodes="${EPISODES}" \
            --record \
            --save_failures="${SAVE_FAILURES}" \
            --repo_id="${shard}" \
            --data_dir="${DATA_DIR}" \
            --fps="${FPS}" \
            --env="${ENV_CONFIG}" \
            >/dev/null 2>&1 &
    fi
    PIDS+=($!)
done

if [[ "${DEBUG}" == "1" ]]; then
    echo "DEBUG=1: per-worker logs at ${DATA_DIR}/_logs/<shard>.log"
fi
echo "All ${NUM_WORKERS} workers started. Ctrl+C to stop."

START_TS=$(date +%s)
read START_EP START_FR <<< "$(count_stats)"
PREV_TS=${START_TS}
PREV_EP=${START_EP}
PREV_FR=${START_FR}
(
    while sleep "${STATS_INTERVAL}"; do
        read now now_fr <<< "$(count_stats)"
        now_ts=$(date +%s)
        elapsed=$(( now_ts - START_TS ))
        delta=$(( now - START_EP ))
        # interval deltas (since the previous stats print) -> instantaneous rate
        int_dt=$(( now_ts - PREV_TS ))
        int_de=$(( now - PREV_EP ))
        int_df=$(( now_fr - PREV_FR ))
        rates=$("${PYTHON_BIN}" -c "
elapsed=${elapsed}; de=${delta}; fps=${FPS}; tot_fr=${now_fr}
int_dt=${int_dt}; int_de=${int_de}; int_df=${int_df}
epm = int_de * 60 / max(int_dt, 1)          # instantaneous: last interval only
hpm = int_df / fps / 3600 * 60 / max(int_dt, 1)
cum_epm = de * 60 / max(elapsed, 1)         # cumulative since start (dragged by warmup)
total_h = tot_fr / fps / 3600
print(f'{hpm:.2f} {epm:.1f} {cum_epm:.1f} {total_h:.2f}')
")
        read hpm epm cum_epm total_h <<< "${rates}"
        echo "[$(date +%H:%M:%S)] total=${now} ep / ${total_h}h  +${int_de} in ${int_dt}s  ${hpm} hours/min  (${epm} ep/min now | ${cum_epm} avg)"
        PREV_TS=${now_ts}; PREV_EP=${now}; PREV_FR=${now_fr}
    done
) &
STATS_PID=$!

# Stay alive while at least one worker is still running. set -e + `wait`
# can return early when individual workers exit, so poll explicitly.
LAST_CAP_CHECK=$(date +%s)
while true; do
    any_alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then any_alive=1; break; fi
    done
    [[ "$any_alive" -eq 0 ]] && break
    if [[ "${MAX_EPISODES}" -gt 0 ]]; then
        now_ts=$(date +%s)
        if (( now_ts - LAST_CAP_CHECK >= STATS_INTERVAL )); then
            LAST_CAP_CHECK=$now_ts
            read current _ <<< "$(count_stats)"
            if (( current >= MAX_EPISODES )); then
                echo "[$(date +%H:%M:%S)] reached MAX_EPISODES=${MAX_EPISODES} (have ${current}); stopping workers..."
                kill -INT "${PIDS[@]}" 2>/dev/null || true
                break
            fi
        fi
    fi
    sleep 5
done

echo "Collection finished."
echo "Each shard under ${DATA_DIR}/ is a standalone LeRobot dataset. Summarize them with:"
echo "  python molmospaces/scripts/dataset_stats.py ${DATA_DIR}"
