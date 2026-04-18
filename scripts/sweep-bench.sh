#!/usr/bin/env bash
# Sweep benchmark pipeline: run benchmarks, save CSV, generate plots.
#
# Usage:
#   ./scripts/sweep-bench.sh                      # full run
#   ./scripts/sweep-bench.sh --max-n 1000000      # cap N range
#   ./scripts/sweep-bench.sh --op insert           # one operation only
#   ./scripts/sweep-bench.sh --plot-only           # re-plot from latest CSV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench-results"

mkdir -p "$RESULTS_DIR"

# Check for --plot-only flag
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
    # ── Run benchmark ────────────────────────────────────────────────────
    TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
    CSV="$RESULTS_DIR/sweep-${TIMESTAMP}.csv"

    echo "Running sweep benchmark..."
    cargo bench --bench sweep -- "${SWEEP_ARGS[@]}" 2>/dev/null > "$CSV"

    # Update latest symlink
    ln -sf "sweep-${TIMESTAMP}.csv" "$RESULTS_DIR/sweep-latest.csv"

    ROWS=$(( $(wc -l < "$CSV") - 1 ))
    echo "Saved $ROWS data rows to $CSV"
else
    CSV="$RESULTS_DIR/sweep-latest.csv"
    if [[ ! -f "$CSV" ]]; then
        echo "No sweep-latest.csv found. Run without --plot-only first."
        exit 1
    fi
    echo "Re-plotting from $CSV"
fi

# ── Generate plots ───────────────────────────────────────────────────────
echo "Generating plots..."
nix shell nixpkgs#gnuplot -c gnuplot \
    -e "csv='$CSV'; outdir='$RESULTS_DIR'" \
    "$SCRIPT_DIR/sweep-plot.gp"

# List generated PNGs
PNGS=$(ls "$RESULTS_DIR"/*.png 2>/dev/null | sort)
if [[ -n "$PNGS" ]]; then
    echo "Plots:"
    echo "$PNGS" | while read -r f; do echo "  $f"; done
else
    echo "Warning: no plots generated"
fi
