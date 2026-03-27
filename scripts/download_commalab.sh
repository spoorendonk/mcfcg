#!/usr/bin/env bash
set -euo pipefail

# Downloads CommaLab/UniPi MMCF benchmark instances (grid and planar families)
# into data/commalab/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data/commalab"

BASE_URL="https://commalab.di.unipi.it/files/Data/MMCF"

mkdir -p "$DATA_DIR"

for family in grid planar; do
    archive="$family.tgz"
    url="$BASE_URL/$archive"
    dest="$DATA_DIR/$family"

    if [ -d "$dest" ]; then
        echo "$family: already present at $dest, skipping"
        continue
    fi

    echo "$family: downloading $url ..."
    tmp="$DATA_DIR/$archive"
    curl -fSL "$url" -o "$tmp"

    echo "$family: extracting ..."
    mkdir -p "$dest"
    tar xzf "$tmp" -C "$dest"
    rm "$tmp"

    echo "$family: done ($(find "$dest" -type f | wc -l) files)"
done

echo "All CommaLab instances in $DATA_DIR"
