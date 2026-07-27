#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .installation_complete ] || [ ! -x .venv/bin/python ]; then
  echo "The mapper is not installed yet. Run ./install.sh first."
  exit 1
fi

PORT=8501
URL="http://localhost:${PORT}"
echo "Starting 96-Tube Rack Mapper at $URL"

.venv/bin/python -m streamlit run streamlit_app.py \
  --server.address localhost \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false &
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
