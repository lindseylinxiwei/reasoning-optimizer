#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
node_bin="/Users/lindseywei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin"
cli="/private/tmp/jev-netlify-cli/node_modules/netlify-cli/bin/run.js"
python_bin="${JEV_PYTHON:-/private/tmp/moar-modern-venv/bin/python}"
for file in "$node_bin/node" "$cli" "$python_bin"; do
  if [ ! -e "$file" ]; then
    echo "Missing runtime: $file. Ask Codex to refresh the local tools." >&2
    exit 1
  fi
done
export PATH="$node_bin:$PATH"
export NODE_OPTIONS="${NODE_OPTIONS:-} --dns-result-order=ipv4first"
export JEV_FORCE_IPV4=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export DOCETL_HOME_DIR="${DOCETL_HOME_DIR:-/private/tmp/jev-local-terminal}"
cd "$project_dir"
script="$project_dir/smoke_local_gateway.py"
if [ "${1:-}" = "pipeline" ]; then
  script="$project_dir/run_sustainability_local.py"
fi
exec "$node_bin/node" "$cli" dev:exec "$python_bin" "$script"
