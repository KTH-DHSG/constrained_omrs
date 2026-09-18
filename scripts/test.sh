#!/usr/bin/env bash
set -euo pipefail

# ROS 2 installations expose pytest plugins through PYTHONPATH. Those plugins
# are unrelated to this pure-Python package and may pull ROS-only dependencies
# into pytest startup. Disable third-party pytest plugin auto-discovery.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

exec uv run python -m pytest "$@"
