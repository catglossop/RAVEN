#!/usr/bin/env bash
#
# Set up the RAVEN environment with uv.
#
# Usage:
#   bash docs/raven_setup.sh            # default environment
#   bash docs/raven_setup.sh --gpu      # additionally install faiss-gpu
#
# The environment is created at ./.venv from pyproject.toml + uv.lock. Activate
# it with `source .venv/bin/activate`, or prefix commands with `uv run`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRAS=()
for arg in "$@"; do
    case "$arg" in
        --gpu) EXTRAS+=(--extra gpu) ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Install uv if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Creates ./.venv with the exact versions pinned in uv.lock, downloading
# CPython 3.10 if the host does not provide it.
uv sync "${EXTRAS[@]+"${EXTRAS[@]}"}"

cat <<'EOF'

Done. Activate the environment with:

    source .venv/bin/activate

or run commands directly without activating:

    uv run python raven_qa_run.py --dataset real_world --agent raven --embedder qqmm --vlm gp3

EOF
