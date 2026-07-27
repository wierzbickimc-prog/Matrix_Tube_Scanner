#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_FILE="$PWD/install.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "96-Tube Rack Mapper installer"
echo "Folder: $PWD"
echo "macOS: $(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "Architecture: $(uname -m)"
echo

if ! command -v curl >/dev/null 2>&1; then
  osascript -e 'display dialog "The installer needs curl, but curl was not found on this Mac." buttons {"OK"} default button "OK"'
  exit 1
fi

UV_DIR="$PWD/.uv-bin"
UV="$UV_DIR/uv"

if [ ! -x "$UV" ]; then
  echo "Downloading the local Python installer..."
  mkdir -p "$UV_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
fi

echo
echo "Installing a private Python 3.12 runtime..."
"$UV" python install 3.12

echo
echo "Creating the private application environment..."
rm -rf .venv
"$UV" venv --python 3.12 .venv

echo
echo "Installing application packages..."
"$UV" pip install --python .venv/bin/python -r requirements.txt

echo
echo "Verifying installation..."
.venv/bin/python - <<'PY'
import sys
import cv2
import numpy
import pandas
import streamlit
import zxingcpp

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("Streamlit:", streamlit.__version__)
print("ZXing-C++: loaded")
PY

touch .installation_complete

echo
echo "INSTALLATION COMPLETE"
echo "Double-click run.command to start the mapper."
echo "A copy of this output is saved as install.log."

osascript -e 'display dialog "Installation complete. Double-click run.command. The mapper will open in your web browser." buttons {"OK"} default button "OK"'
