#!/usr/bin/env bash
set -euo pipefail

# Prepares intermodal MMCF instances from the tumBAIS repo.
# 1. Shallow-clones the repo (if not already present), skipping LFS
# 2. Checks if LFS data files are real; if not, attempts git lfs pull
#    If that fails, warns and exits — user must manually place the data files
# 3. Generates instances using generate_instances.py (SUBWAY, BUS, SBT)
# 4. Cleans instances with mcfcg_clean
# 5. Places cleaned instances in data/intermodal/
#
# Fully re-runnable: skips clone if repo exists, skips LFS if files are real,
# skips generation if output exists, skips cleaning if cleaned file exists.
#
# Requirements:
#   - Python 3 with: networkx, pandas, numpy, geopandas, shapely, lxml
#   - Build mcfcg_clean first: cmake --build build -j$(nproc)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data/intermodal"
REPO_DIR="$ROOT_DIR/data/intermodal-repo"
CLEAN_BIN="$ROOT_DIR/build/mcfcg_clean"
RAW_DIR="$DATA_DIR/raw"

UPSTREAM="https://github.com/tumBAIS/intermodalTransportationNetworksCG.git"

# LFS data files that must be real (not pointers)
LFS_FILES=(
    "$REPO_DIR/data/network_pt_road.xml"
    "$REPO_DIR/data/schedule.xml"
)

if [ ! -x "$CLEAN_BIN" ]; then
    echo "Error: mcfcg_clean not found at $CLEAN_BIN"
    echo "Build first: cmake --build build -j\$(nproc)"
    exit 1
fi

# --- Step 1: Clone repo (skip LFS) ---
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning intermodal repo (skipping LFS) ..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$UPSTREAM" "$REPO_DIR"
else
    echo "Repo already present at $REPO_DIR"
fi

# --- Step 2: Check LFS data files ---
lfs_ok=true
for f in "${LFS_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Missing: $f"
        lfs_ok=false
    elif head -1 "$f" | grep -q "^version https://git-lfs"; then
        echo "LFS pointer: $f"
        lfs_ok=false
    fi
done

if [ "$lfs_ok" = false ]; then
    echo ""
    echo "Some LFS data files are missing or are pointers."
    echo "Attempting git lfs pull ..."
    if (cd "$REPO_DIR" && git lfs pull 2>/dev/null); then
        echo "LFS pull succeeded."
    else
        echo ""
        echo "ERROR: Failed to fetch LFS data (upstream LFS quota may be exceeded)."
        echo ""
        echo "Manually copy the data files into $REPO_DIR/data/:"
        echo "  - network_pt_road.xml  (~113 MB)"
        echo "  - schedule.xml         (~42 MB)"
        echo ""
        echo "Then re-run this script."
        exit 1
    fi
fi

echo "LFS data files verified."

# --- Step 3: Generate instances (all modes, seed 0) ---
mkdir -p "$RAW_DIR" "$DATA_DIR"

echo "Generating instances ..."
python3 "$SCRIPT_DIR/generate_instances.py" \
    --repo "$REPO_DIR" \
    --output "$RAW_DIR" \
    --seeds 0 \
    --modes subway bus sbt

# --- Step 4: Clean and gzip instances ---
# Cleaned output is gzipped (.txt.gz) — that's what the test suite and
# mcfcg_cli load.  Skip if the .txt.gz already exists.
echo "Cleaning instances ..."
for raw in "$RAW_DIR"/*.txt; do
    [ -f "$raw" ] || continue
    name="$(basename "$raw")"
    cleaned="$DATA_DIR/$name"
    if [ -f "$cleaned.gz" ]; then
        echo "  $name.gz: already present, skipping"
        continue
    fi
    echo "  $name: cleaning ..."
    "$CLEAN_BIN" "$raw" --output "$cleaned"
    gzip -f "$cleaned"
done

echo ""
echo "Done. Cleaned instances in $DATA_DIR/"
ls -lh "$DATA_DIR"/*.txt 2>/dev/null | grep -v raw || true
