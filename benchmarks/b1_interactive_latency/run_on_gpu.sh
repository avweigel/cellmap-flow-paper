#!/bin/bash
# Run B1 end-to-end on a GPU node. Designed to be invoked via bsub:
#
#     bsub -q gpu_h100 -gpu "num=1" -W 4:00 -n 4 \
#          -o results/lsf_<label>.log \
#          benchmarks/b1_interactive_latency/run_on_gpu.sh \
#              <data_path> <hf_repo> <label> <output_json>
#
# Boots cellmap_flow_server in the background, waits for it to respond,
# runs the B1 client, saves results JSON, kills server.

set -euo pipefail

DATA_PATH="${1:?data_path required}"
HF_REPO="${2:?hf_repo required}"
LABEL="${3:?label required}"
OUTPUT_JSON="${4:?output_json required}"
SCALE="${5:-1}"
CHUNK_GRID_Z="${6:-8}"
CHUNK_GRID_Y="${7:-8}"
CHUNK_GRID_X="${8:-8}"
N_WARMUP="${9:-20}"
N_MEASURE="${10:-200}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Activate env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellmap-flow-paper

# Pick a free port (random in 20000-30000 range, retry on conflict).
pick_port() {
    while :; do
        local p=$((RANDOM % 10000 + 20000))
        (echo > /dev/tcp/127.0.0.1/$p) >/dev/null 2>&1 || { echo "$p"; return; }
    done
}
PORT=$(pick_port)

echo "[run_on_gpu] hostname=$(hostname) port=$PORT data=$DATA_PATH repo=$HF_REPO"

# Start server in background.
SERVER_LOG="${OUTPUT_JSON%.json}.server.log"
mkdir -p "$(dirname "$SERVER_LOG")"
cellmap_flow_server huggingface \
    --repo "$HF_REPO" \
    --data-path "$DATA_PATH" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_on_gpu] killing server pid=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait up to 5 min for server to respond. The first chunk request may take
# longer than this if the HuggingFace download is slow, so we poll the API
# spec endpoint instead, which responds as soon as Flask is up.
echo "[run_on_gpu] waiting for server"
for i in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/api_spec.json" > /dev/null 2>&1; then
        echo "[run_on_gpu] server up after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_on_gpu] server died early. Tail of log:"
        tail -50 "$SERVER_LOG"
        exit 2
    fi
    sleep 1
done

# Run B1 client
echo "[run_on_gpu] running B1 client"
python -m benchmarks.b1_interactive_latency.run \
    --server "http://127.0.0.1:$PORT" \
    --dataset data \
    --scale "$SCALE" \
    --chunk-grid "$CHUNK_GRID_Z" "$CHUNK_GRID_Y" "$CHUNK_GRID_X" \
    --n-warmup "$N_WARMUP" \
    --n-measure "$N_MEASURE" \
    --output "$OUTPUT_JSON" \
    --label "$LABEL"

echo "[run_on_gpu] done -> $OUTPUT_JSON"
