#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".installation_complete" ] || [ ! -x ".venv/bin/python" ]; then
  osascript -e 'display dialog "The browser version is not installed yet. Run install.command first and wait for INSTALLATION COMPLETE." buttons {"OK"} default button "OK"'
  exit 1
fi

PORT=8501
URL="http://localhost:${PORT}"

echo "Starting 96-Tube Rack Mapper..."
echo "The browser will open at $URL"
echo "Keep this Terminal window open while using the application."
echo

.venv/bin/python -m streamlit run streamlit_app.py \
  --server.address localhost \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for attempt in {1..30}; do
  if curl -fsS "$URL/_stcore/health" >/dev/null 2>&1; then
    open "$URL"
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo
    echo "The server stopped before it became ready."
    exit 1
  fi
  sleep 1
done

echo
echo "The application did not become ready within 30 seconds."
kill "$SERVER_PID" >/dev/null 2>&1 || true
exit 1
