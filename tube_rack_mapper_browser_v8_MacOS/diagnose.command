#!/bin/bash
cd "$(dirname "$0")"

echo "96-Tube Rack Mapper diagnostics"
echo "Folder: $PWD"
echo "macOS: $(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "Architecture: $(uname -m)"
echo

if [ -x ".uv-bin/uv" ]; then
  echo "uv: $(.uv-bin/uv --version)"
else
  echo "uv: NOT INSTALLED"
fi

if [ -x ".venv/bin/python" ]; then
  echo "Private Python: $(.venv/bin/python --version)"
  echo
  .venv/bin/python - <<'PY'
modules = ["cv2", "numpy", "pandas", "streamlit", "zxingcpp"]
for name in modules:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "loaded")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: ERROR - {exc}")
PY
else
  echo "Private Python: NOT INSTALLED"
fi

echo
echo "Installation marker:"
if [ -f ".installation_complete" ]; then
  echo "present"
else
  echo "missing"
fi

echo
read -r -p "Press Return to close..."
