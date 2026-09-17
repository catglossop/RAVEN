#!/usr/bin/env bash
# Start the RAVEN planning server detached, so it survives the shell that launched it.
# It replaces RAGNav's planning server: same protocol, same default port (54322), so
# OmniVLA's --ragnav-host/--ragnav-port need no change.
#
#   bash scripts/launch_inference_server.sh
#   PORT=54322 GPU=0 SCENE_DIR=/data/scenes/bww8 bash scripts/launch_inference_server.sh
#   ALLOW_REMOTE=1 HOST=0.0.0.0 bash scripts/launch_inference_server.sh   # robot on another host
#
# Environment:
#   HOST/PORT       bind address (default 127.0.0.1:54322)
#   GPU             CUDA device for the image embedder (default 0; needs ~17 GB)
#   VLM             planning and completion model (default gemini-3.8-flash)
#   SCENE_DIR       optional: embed this scene at startup instead of on the first create_plan
#   LANDMARKS_FILE  optional: defaults to SCENE_DIR/*_landmarks.json
#   ALLOW_REMOTE    1 to accept non-loopback clients (RAGNav rejects them)
#   LOG_DIR         logs and per-plan JSONL (default <repo>/output/inference)
#   CACHE_DIR       scene embedding cache (default <repo>/output/inference_cache)
#   EXTRA_ARGS      passed through to raven.inference.server
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# In a git worktree the uv environment and output/ live in the main checkout.
MAIN_ROOT="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null | sed 's:/\.git/*$::')"
[[ -d "${MAIN_ROOT}" ]] || MAIN_ROOT="${REPO_ROOT}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-54322}"
GPU="${GPU:-0}"
VLM="${VLM:-gemini-3.8-flash}"
LOG_DIR="${LOG_DIR:-${MAIN_ROOT}/output/inference}"
CACHE_DIR="${CACHE_DIR:-${MAIN_ROOT}/output/inference_cache}"
if [[ -z "${PYTHON:-}" ]]; then
  for candidate in "${REPO_ROOT}/.venv/bin/python" "${MAIN_ROOT}/.venv/bin/python"; do
    [[ -x "${candidate}" ]] && PYTHON="${candidate}" && break
  done
fi
PYTHON="${PYTHON:-$(command -v python3)}"
# Run from the repo root so `python -m` picks up this checkout's raven package.
(cd "${REPO_ROOT}" && "${PYTHON}" -c "import raven.inference.server" >/dev/null 2>&1) || {
  echo "${PYTHON} cannot import raven.inference.server (needs RAVEN's uv environment);" >&2
  echo "set PYTHON=/path/to/.venv/bin/python" >&2; exit 1; }

if [[ -z "${GOOGLE_API_KEY:-}" && "${VLM}" == gemini-* ]]; then
  echo "GOOGLE_API_KEY is not set; ${VLM} calls will fail." >&2
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "Port ${PORT} is already in use (RAGNav's server, or a previous RAVEN server)." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/plans" "${CACHE_DIR}"
args=(--host "${HOST}" --port "${PORT}" --vlm "${VLM}" --completion vlm
      --cache-dir "${CACHE_DIR}" --log-dir "${LOG_DIR}/plans")
[[ -n "${SCENE_DIR:-}" ]] && args+=(--scene-dir "${SCENE_DIR}")
[[ -n "${LANDMARKS_FILE:-}" ]] && args+=(--landmarks-file "${LANDMARKS_FILE}")
[[ "${ALLOW_REMOTE:-0}" != "0" ]] && args+=(--allow-remote-clients)
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=(${EXTRA_ARGS})

log="${LOG_DIR}/server.log"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') starting on ${HOST}:${PORT} (GPU ${GPU}, ${VLM}) ===" >> "${log}"
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 setsid nohup \
  "${PYTHON}" -m raven.inference.server "${args[@]}" >> "${log}" 2>&1 < /dev/null &
pid=$!

for _ in $(seq 1 120); do
  grep -q "^Length-prefixed JSON server listening" "${log}" && break
  kill -0 "${pid}" 2>/dev/null || { echo "server exited; see ${log}" >&2; tail -5 "${log}" >&2; exit 1; }
  sleep 5
done

cat <<EOF
RAVEN planning server: pid ${pid} on ${HOST}:${PORT}
  log          ${log}
  plans        ${LOG_DIR}/plans/plans.jsonl
  embeddings   ${CACHE_DIR}
  check it     ${PYTHON} raven/inference/client.py --host ${HOST} --port ${PORT} --ping
  stop it      kill ${pid}
EOF
