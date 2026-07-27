#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "96-Tube Rack Mapper diagnostics"
echo "System: $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "Architecture: $(uname -m)"
echo
[ -x .uv-bin/uv ] && echo "uv: $(.uv-bin/uv --version)" || echo "uv: NOT INSTALLED"
[ -x .venv/bin/python ] && echo "Private Python: $(.venv/bin/python --version)" || echo "Private Python: NOT INSTALLED"
[ -f .installation_complete ] && echo "Installation marker: present" || echo "Installation marker: missing"
