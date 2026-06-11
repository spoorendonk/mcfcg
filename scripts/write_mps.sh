#!/usr/bin/env bash
# Write <instance>.mps.gz (the compact source-based LP, paper section 2.2) next
# to each instance, via the mcfcg CLI's --write-mps. HiGHS-only — no GPU/COPT
# needed. Set MCFCG_CLI to override the binary (default build/mcfcg_cli).
#
# Usage:
#   scripts/write_mps.sh [family ...]
# families: grid planar transportation intermodal (default: all four).
#
# Note: the source LP has |S|*|E| variables. On intermodal (unique sources,
# |S| ~ |K|) and the largest transportation/planar instances the MPS can be
# very large; restrict with an explicit family list if that's a concern.

CLI=${MCFCG_CLI:-build/mcfcg_cli}
families=("$@")
[ ${#families[@]} -eq 0 ] && families=(grid planar transportation intermodal)

if [ ! -x "$CLI" ]; then
    echo "mcfcg CLI not found at '$CLI' (build it, or set MCFCG_CLI)." >&2
    exit 1
fi

write() {  # <instance-path> <output.mps.gz>
    if "$CLI" "$1" --write-mps "$2" >/dev/null 2>&1; then
        echo "  ok   $2"
    else
        echo "  FAIL $1" >&2
    fi
}

for fam in "${families[@]}"; do
    echo "[$fam]"
    case "$fam" in
        grid)    for f in data/commalab/grid/grid[0-9]*;   do [ -f "$f" ] && write "$f" "$f.mps.gz"; done ;;
        planar)  for f in data/commalab/planar/planar[0-9]*; do [ -f "$f" ] && write "$f" "$f.mps.gz"; done ;;
        transportation)
            for net in data/transportation/*_net.tntp.gz; do
                [ -f "$net" ] || continue
                write "$net" "${net%_net.tntp.gz}.mps.gz"
            done ;;
        intermodal)
            for inst in data/intermodal/*.txt.gz; do
                [ -f "$inst" ] || continue
                write "$inst" "${inst%.txt.gz}.mps.gz"
            done ;;
        *) echo "  unknown family '$fam'" >&2 ;;
    esac
done
