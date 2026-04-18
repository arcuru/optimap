#!/usr/bin/env bash
# Sweep benchmark comparing all three hash_tag implementations.
#
# Runs the sweep benchmark three times (once per feature variant), suffixing
# design names with the variant (e.g. "UFM/asm", "UFM/128", "UFM/default").
# Merges results into a single CSV for plotting.
#
# Usage:
#   ./scripts/sweep-hash-tag.sh                        # full run
#   ./scripts/sweep-hash-tag.sh --max-n 1000000        # cap N range
#   ./scripts/sweep-hash-tag.sh --design UFM           # one design only
#   ./scripts/sweep-hash-tag.sh --op lookup_hit        # one operation only
#   ./scripts/sweep-hash-tag.sh --plot-only            # re-plot latest CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench-results"

mkdir -p "$RESULTS_DIR"

# Parse flags
PLOT_ONLY=false
SWEEP_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--plot-only" ]]; then
        PLOT_ONLY=true
    else
        SWEEP_ARGS+=("$arg")
    fi
done

if [[ "$PLOT_ONLY" == false ]]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
    CSV="$RESULTS_DIR/sweep-hash-tag-${TIMESTAMP}.csv"
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT

    echo "=== Hash tag sweep: 3 variants × sweep ==="

    # Variant 1: reduced-hash-asm (default features)
    echo "[1/3] reduced-hash-asm (2 insn, 255 values)..."
    cargo bench --bench sweep -- "${SWEEP_ARGS[@]}" 2>/dev/null \
        | tail -n +2 \
        | sed 's/,\([^,]*\),/,\1\/asm,/' \
        > "$TMPDIR/asm.csv"

    # Variant 2: reduced-hash-128
    echo "[2/3] reduced-hash-128 (1 insn, 128 values)..."
    cargo bench --bench sweep --no-default-features --features reduced-hash-128 \
        -- "${SWEEP_ARGS[@]}" 2>/dev/null \
        | tail -n +2 \
        | sed 's/,\([^,]*\),/,\1\/128,/' \
        > "$TMPDIR/128.csv"

    # Variant 3: pure Rust (no features)
    echo "[3/3] pure Rust (3 insn, 255 values)..."
    cargo bench --bench sweep --no-default-features \
        -- "${SWEEP_ARGS[@]}" 2>/dev/null \
        | tail -n +2 \
        | sed 's/,\([^,]*\),/,\1\/pure,/' \
        > "$TMPDIR/pure.csv"

    # Merge into single CSV
    echo "operation,design,n,ns_per_op" > "$CSV"
    cat "$TMPDIR/asm.csv" "$TMPDIR/128.csv" "$TMPDIR/pure.csv" >> "$CSV"

    ln -sf "sweep-hash-tag-${TIMESTAMP}.csv" "$RESULTS_DIR/sweep-hash-tag-latest.csv"

    ROWS=$(( $(wc -l < "$CSV") - 1 ))
    echo "Saved $ROWS data rows to $CSV"
else
    CSV="$RESULTS_DIR/sweep-hash-tag-latest.csv"
    if [[ ! -f "$CSV" ]]; then
        echo "No sweep-hash-tag-latest.csv found. Run without --plot-only first."
        exit 1
    fi
    echo "Re-plotting from $CSV"
fi

# ── Generate plots ──────────────────────────────────────────────────────────
echo "Generating plots..."
nix shell nixpkgs#gnuplot -c gnuplot \
    -e "csv='$CSV'; outdir='$RESULTS_DIR'" \
    "$SCRIPT_DIR/sweep-hash-tag-plot.gp"

PNGS=$(ls "$RESULTS_DIR"/hash-tag-*.png 2>/dev/null | sort)
if [[ -n "$PNGS" ]]; then
    echo "Plots:"
    echo "$PNGS" | while read -r f; do echo "  $f"; done
else
    echo "Warning: no plots generated"
fi
