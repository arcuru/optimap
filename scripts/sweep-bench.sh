#!/usr/bin/env bash
# Sweep benchmark pipeline: run benchmarks, save CSV, generate plots.
#
# Usage:
#   ./scripts/sweep-bench.sh                      # full run (new dated run dir)
#   ./scripts/sweep-bench.sh --max-n 1000000      # cap N range
#   ./scripts/sweep-bench.sh --op insert          # one operation only
#   ./scripts/sweep-bench.sh --design-set all     # full design matrix
#   ./scripts/sweep-bench.sh --plot-only          # re-plot from latest CSV
#
# Storage:
#   Each invocation writes into bench-results/runs/<ts>-<sha>/ unless
#   BENCH_RUN_DIR is exported (in which case it appends to that dir — used by
#   bench-overnight.sh to bundle multiple benches into one run).
#
#     bench-results/
#     ├── runs/2026-06-01-220000-c137aee/
#     │   ├── sweep.csv
#     │   ├── plots/{insert,lookup_hit,...}.png
#     │   ├── stdout.sweep.log
#     │   └── meta.json
#     ├── latest -> runs/<...>           (always points at most recent run)
#     └── (older flat files left as historical)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench-results"

mkdir -p "$RESULTS_DIR/runs"

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

# Resolve run dir: caller-provided BENCH_RUN_DIR or a fresh dated one
if [[ -z "${BENCH_RUN_DIR:-}" ]]; then
    TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
    SHA=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")
    RUN_DIR="$RESULTS_DIR/runs/${TIMESTAMP}-${SHA}"
    mkdir -p "$RUN_DIR/plots"
else
    RUN_DIR="$BENCH_RUN_DIR"
    mkdir -p "$RUN_DIR/plots"
fi

write_meta_once() {
    local meta="$RUN_DIR/meta.json"
    if [[ -f "$meta" ]]; then return; fi

    local sha dirty branch host kernel governor rustc avx512 avx2
    sha=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "nogit")
    dirty=$(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null | wc -l)
    branch=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    host=$(hostname)
    kernel=$(uname -r)
    governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    rustc=$( ( cd "$PROJECT_DIR" && direnv exec . rustc --version ) 2>/dev/null || echo "unknown")
    avx2=$(grep -c avx2 /proc/cpuinfo | head -1)
    avx512=$(grep -c avx512f /proc/cpuinfo | head -1)

    cat >"$meta" <<EOF
{
  "started":   "$(date -Iseconds)",
  "host":      "$host",
  "kernel":    "$kernel",
  "governor":  "$governor",
  "rustc":     "$rustc",
  "git_sha":   "$sha",
  "git_branch":"$branch",
  "git_dirty_files": $dirty,
  "cpu_avx2_cores":   $avx2,
  "cpu_avx512_cores": $avx512,
  "rustflags": "${RUSTFLAGS:-}"
}
EOF
}

write_meta_once

CSV="$RUN_DIR/sweep.csv"
LOG="$RUN_DIR/stdout.sweep.log"

if [[ "$PLOT_ONLY" == false ]]; then
    echo "Run dir: $RUN_DIR"
    echo "Args:    ${SWEEP_ARGS[*]:-<none>}"
    echo "Running sweep benchmark..."

    # Record what we ran
    printf '%s\n' "${SWEEP_ARGS[@]:-}" >"$RUN_DIR/sweep.args"

    # Cargo bench writes informational lines to stderr; the CSV stream goes to
    # stdout. Tee both — stdout into the CSV, stderr into a sibling log — so the
    # raw cargo output stays reproducible.
    ( cd "$PROJECT_DIR" && direnv exec . cargo bench --bench sweep -- "${SWEEP_ARGS[@]}" ) 2>"$LOG" >"$CSV"

    ROWS=$(( $(wc -l < "$CSV") - 1 ))
    echo "Saved $ROWS data rows to $CSV"
else
    if [[ ! -f "$CSV" ]]; then
        echo "No sweep.csv in $RUN_DIR. Run without --plot-only first."
        exit 1
    fi
    echo "Re-plotting from $CSV"
fi

# ── Generate plots ───────────────────────────────────────────────────────
echo "Generating plots..."
nix shell nixpkgs#gnuplot -c gnuplot \
    -e "csv='$CSV'; outdir='$RUN_DIR/plots'" \
    "$SCRIPT_DIR/sweep-plot.gp"

PNGS=$(ls "$RUN_DIR/plots/"*.png 2>/dev/null | sort)
if [[ -n "$PNGS" ]]; then
    echo "Plots:"
    echo "$PNGS" | while read -r f; do echo "  $f"; done
else
    echo "Warning: no plots generated"
fi

# Update latest pointer (only if we created the dir ourselves, to avoid the
# overnight wrapper repeatedly clobbering the symlink mid-run).
if [[ -z "${BENCH_RUN_DIR:-}" ]]; then
    ln -sfn "runs/$(basename "$RUN_DIR")" "$RESULTS_DIR/latest"
fi
