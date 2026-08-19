#!/bin/bash
# Resolve Python venv. Sourced by other hooks.
#
# On success (venv present): VENV_BIN is set to the venv's bin directory and
# callers can run `"$VENV_BIN/ruff"`, `"$VENV_BIN/pytest"`, etc.
#
# If the project declares Python dependencies or packaging but has no venv:
# exit 2, which propagates through `source` and blocks the enclosing hook.
#
# If it declares none (e.g. a C++ project like this one, whose only Python is
# stdlib utility scripts under scripts/ and test/python/): set VENV_BIN="" and
# return. Callers already guard every Python
# tool invocation with `[ -x "$VENV_BIN/foo" ]`, so those steps skip naturally
# without the hook failing. This repo declares no Python dependencies, so
# VENV_BIN is always empty here and every Python step below skips.
#
# Usage:
#   source "$(dirname "$0")/resolve-venv.sh"

VENV_DIR=""
[ -d ".venv" ] && VENV_DIR=".venv"
[ -d "venv" ] && VENV_DIR="venv"

if [ -z "$VENV_DIR" ]; then
  NEEDS_VENV=0
  [ -f "setup.py" ] && NEEDS_VENV=1
  [ -f "setup.cfg" ] && NEEDS_VENV=1
  [ -f "Pipfile" ] && NEEDS_VENV=1
  [ -f "poetry.lock" ] && NEEDS_VENV=1
  [ -f "uv.lock" ] && NEEDS_VENV=1
  for req in requirements*.txt; do
    [ -e "$req" ] && NEEDS_VENV=1
  done
  # PEP 621 [project], PEP 517 [build-system], PEP 735 [dependency-groups], a
  # build backend's own table, or a bare `dependencies = [...]`. A pyproject.toml
  # holding nothing but [tool.*] config is not a Python project.
  if [ -f "pyproject.toml" ] && grep -qE \
    '^[[:space:]]*(\[(project|build-system|dependency-groups)\]|\[tool\.(poetry|pdm|hatch|flit|uv|pixi|setuptools)|dependencies[[:space:]]*=)' \
    pyproject.toml; then
    NEEDS_VENV=1
  fi

  if [ "$NEEDS_VENV" -eq 1 ]; then
    echo "FAILED: No virtualenv found. Create one first:"
    echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
    exit 2
  fi
  # No declared dependencies — treat staged .py files as standalone utilities
  # and let Python tool steps skip via the `-x "$VENV_BIN/..."` guards.
  VENV_BIN=""
  return 0 2>/dev/null || exit 0
fi

VENV_BIN="$VENV_DIR/bin"
