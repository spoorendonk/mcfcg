#!/bin/bash
# Git pre-commit hook.
#
# Flow:
#   1. Auto-format + auto-fix lint (staged files only)
#   2. Tests → hard block (no bypass)

set -e

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)

[ -z "$STAGED_FILES" ] && exit 0

CPP_FILES=$(echo "$STAGED_FILES" | grep -E '\.(cpp|cc|cxx|h|hpp|hxx)$' || true)
PY_FILES=$(echo "$STAGED_FILES" | grep -E '\.py$' || true)
SH_FILES=$(echo "$STAGED_FILES" | grep -E '\.sh$' || true)

# Python venv — required for all Python tooling
if [ -n "$PY_FILES" ]; then
  source "$(dirname "$0")/resolve-venv.sh"
fi

# ============================================================
# Step 1: Auto-format and auto-fix
# ============================================================

# C++ formatting
if [ -n "$CPP_FILES" ] && command -v clang-format &>/dev/null; then
  clang-format -i $CPP_FILES || true
fi

# Shell script formatting
if [ -n "$SH_FILES" ] && command -v shfmt &>/dev/null; then
  shfmt -w $SH_FILES || true
fi

# Python formatting + auto-fixable lint
if [ -n "$PY_FILES" ] && [ -x "$VENV_BIN/ruff" ]; then
  "$VENV_BIN/ruff" format --quiet $PY_FILES
  "$VENV_BIN/ruff" check --fix --quiet $PY_FILES 2>/dev/null || true
fi

# C++ auto-fixable lint (safe clang-tidy fixes only)
if [ -n "$CPP_FILES" ] && command -v clang-tidy &>/dev/null; then
  COMPILE_DB=""
  [ -f "build/compile_commands.json" ] && COMPILE_DB="-p build"
  [ -f "compile_commands.json" ] && COMPILE_DB="-p ."

  if [ -n "$COMPILE_DB" ]; then
    for f in $CPP_FILES; do
      clang-tidy $COMPILE_DB \
        --checks="-*,modernize-use-nullptr,modernize-use-override,modernize-use-using,readability-braces-around-statements" \
        --fix --quiet "$f" 2>/dev/null || true
    done
  fi
fi

# Re-stage files modified by formatting/lint fixes
while read -r f; do
  git add "$f"
done < <(git diff --name-only -- $STAGED_FILES 2>/dev/null)

# ============================================================
# Step 2: Tests (hard block)
# ============================================================

echo "=== Pre-commit: running tests ==="

TESTS_FAILED=0

# Only run C++ tests if C++ files were staged
if [ -n "$CPP_FILES" ] && [ -d "build" ] && command -v ctest &>/dev/null; then
  echo "Running C++ tests..."
  if ! GTEST_BRIEF=1 ctest --test-dir build --output-on-failure --progress -j"$(nproc 2>/dev/null || echo 4)"; then
    echo "FAILED: C++ tests failed."
    TESTS_FAILED=1
  fi
fi

# Only run Python tests if Python files were staged
if [ -n "$PY_FILES" ] && [ -x "$VENV_BIN/pytest" ]; then
  echo "Running Python tests..."
  rc=0; "$VENV_BIN/pytest" --tb=short -q || rc=$?
  # rc=0: passed, rc=5: no tests collected (all skipped) — both OK
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    echo "FAILED: Python tests failed."
    TESTS_FAILED=1
  fi
fi

if [ "$TESTS_FAILED" -ne 0 ]; then
  echo ""
  echo "Commit blocked: tests failed. Fix tests before committing."
  exit 1
fi

echo "Tests passed."
exit 0
