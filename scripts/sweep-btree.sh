#!/usr/bin/env bash
# Sorted-map sweep pipeline: run sweep_btree, save CSV, generate plots.
#
# Usage:
#   ./scripts/sweep-btree.sh                       # full run (new dated run dir)
#   ./scripts/sweep-btree.sh --max-n 1000000       # cap N range
#   ./scripts/sweep-btree.sh --op insert           # one operation only
#   ./scripts/sweep-btree.sh --plot-only           # re-plot from latest CSV
#
# Honours BENCH_RUN_DIR for bench-overnight.sh bundling. See sweep-bench.sh
# header for the storage layout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench-results"

mkdir -p "$RESULTS_DIR/runs"

PLOT_ONLY=false
SWEEP_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--plot-only" ]]; then
        PLOT_ONLY=true
    else
        SWEEP_ARGS+=("$arg")
    fi
done

if [[ -z "${BENCH_RUN_DIR:-}" ]]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
    SHA=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")
    RUN_DIR="$RESULTS_DIR/runs/${TIMESTAMP}-${SHA}-btree"
    mkdir -p "$RUN_DIR/plots"
else
    RUN_DIR="$BENCH_RUN_DIR"
    mkdir -p "$RUN_DIR/plots"
fi

CSV="$RUN_DIR/sweep-btree.csv"
LOG="$RUN_DIR/stdout.sweep-btree.log"

if [[ "$PLOT_ONLY" == false ]]; then
    echo "Run dir: $RUN_DIR"
    echo "Args:    ${SWEEP_ARGS[*]:-<none>}"
    echo "Running sweep_btree benchmark..."

    printf '%s\n' "${SWEEP_ARGS[@]:-}" >"$RUN_DIR/sweep-btree.args"
    ( cd "$PROJECT_DIR" && direnv exec . cargo bench --bench sweep_btree -- "${SWEEP_ARGS[@]}" ) 2>"$LOG" >"$CSV"

    ROWS=$(( $(wc -l < "$CSV") - 1 ))
    echo "Saved $ROWS data rows to $CSV"
else
    if [[ ! -f "$CSV" ]]; then
        echo "No sweep-btree.csv in $RUN_DIR. Run without --plot-only first."
        exit 1
    fi
    echo "Re-plotting from $CSV"
fi

echo "Generating plots..."
nix shell nixpkgs#gnuplot -c gnuplot \
    -e "csv='$CSV'; outdir='$RUN_DIR/plots'" \
    "$SCRIPT_DIR/sweep-btree-plot.gp"

PNGS=$(ls "$RUN_DIR/plots/"btree-*.png 2>/dev/null | sort)
if [[ -n "$PNGS" ]]; then
    echo "Plots:"
    echo "$PNGS" | while read -r f; do echo "  $f"; done
else
    echo "Warning: no plots generated"
fi

if [[ -z "${BENCH_RUN_DIR:-}" ]]; then
    ln -sfn "runs/$(basename "$RUN_DIR")" "$RESULTS_DIR/latest"
fi
