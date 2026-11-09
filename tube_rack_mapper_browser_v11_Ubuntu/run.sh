#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .installation_complete ] || [ ! -x .venv/bin/python ]; then
  echo "The mapper is not installed yet. Run ./install.sh first."
  exit 1
fi

PORT="${TUBE_MAPPER_PORT:-8501}"
BASE_URL_PATH="${TUBE_MAPPER_BASE_URL_PATH:-}"
URL="http://localhost:${PORT}"
if [[ -n "$BASE_URL_PATH" ]]; then
  BASE_URL_PATH="${BASE_URL_PATH#/}"
  URL+="/${BASE_URL_PATH}"
fi
echo "Starting 96-Tube Rack Mapper at $URL"

streamlit_args=(
  --server.address localhost
  --server.port "$PORT"
  --server.headless true
  --browser.gatherUsageStats false
)
if [[ -n "$BASE_URL_PATH" ]]; then
  # Tailscale Funnel mounts this application below /matrix. Streamlit must
  # know the prefix so its assets and WebSocket endpoint use that same path.
  streamlit_args+=(--server.baseUrlPath "$BASE_URL_PATH")
fi

.venv/bin/python -m streamlit run streamlit_app.py \
  "${streamlit_args[@]}" &
SERVER_PID=$!

cleanup() { kill "$SERVER_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

for attempt in {1..30}; do
  if curl -fsS "$URL/_stcore/health" >/dev/null 2>&1; then
    command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
    wait "$SERVER_PID"
    exit $?
  fi
  kill -0 "$SERVER_PID" >/dev/null 2>&1 || exit 1
  sleep 1
done

echo "The application did not become ready within 30 seconds."
exit 1
