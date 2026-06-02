#!/usr/bin/env bash
# Overnight benchmark orchestrator.
#
# Runs the full set of bench artifacts needed to refresh the blog post and to
# archive a complete snapshot of `main`'s perf characteristics. Bundles every
# sub-bench's output into a single timestamped run dir, then kicks off a second
# `--design-set all` sweep into a separate run dir for the matrix archive.
#
# Usage:
#   ./scripts/bench-overnight.sh                 # full plan
#   ./scripts/bench-overnight.sh --smoke         # tiny --max-n=10000, no all-set, no criterion
#   ./scripts/bench-overnight.sh --skip-all      # skip the all-design archive run
#   ./scripts/bench-overnight.sh --skip-criterion # skip the criterion suite
#
# Pre-flight (recommended, requires sudo, not done by this script):
#   sudo cpupower frequency-set -g performance
#
# Outputs:
#   bench-results/runs/<ts>-<sha>-curated/        primary run (sweep curated + btree + perf_isolate + criterion)
#   bench-results/runs/<ts>-<sha>-all/            matrix archive run (sweep --design-set all)
#   bench-results/latest -> runs/<ts>-<sha>-curated   (symlink to primary)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench-results"

SMOKE=false
SKIP_ALL=false
SKIP_CRITERION=false
for arg in "$@"; do
    case "$arg" in
        --smoke)          SMOKE=true ;;
        --skip-all)       SKIP_ALL=true ;;
        --skip-criterion) SKIP_CRITERION=true ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
SHA=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")

PRIMARY_DIR="$RESULTS_DIR/runs/${TIMESTAMP}-${SHA}-curated"
ALL_DIR="$RESULTS_DIR/runs/${TIMESTAMP}-${SHA}-all"
mkdir -p "$PRIMARY_DIR/plots"

if [[ "$SMOKE" == true ]]; then
    MAX_N_ARGS=(--max-n 10000)
    SKIP_ALL=true
    SKIP_CRITERION=true
    echo "═══ SMOKE MODE — small N, no all-set, no criterion ═══"
else
    MAX_N_ARGS=()   # default (full sweep, max_n=10M)
fi

echo "═══ Overnight bench plan ═══"
echo "Primary dir: $PRIMARY_DIR"
[[ "$SKIP_ALL" == false ]] && echo "All-set dir: $ALL_DIR"
echo "Commit:      $SHA"
echo "Governor:    $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "Started:     $(date)"
echo

# ── 1. Curated sweep + btree + perf_isolate + criterion ──────────────────────
export BENCH_RUN_DIR="$PRIMARY_DIR"

echo "─── [1/5] curated sweep ($(date +%H:%M:%S)) ───"
"$SCRIPT_DIR/sweep-bench.sh" --design-set curated "${MAX_N_ARGS[@]}"

echo "─── [2/5] sorted-map sweep ($(date +%H:%M:%S)) ───"
"$SCRIPT_DIR/sweep-btree.sh" "${MAX_N_ARGS[@]}"

echo "─── [3/5] perf_isolate (Tomb / hashbrown / UFM / IPO / Splitsies / IPO64) ($(date +%H:%M:%S)) ───"
PERF_LOG="$PRIMARY_DIR/perf_isolate.log"
{
    echo "# perf_isolate runs — N=920000 (Tomb resize transient), 50K lookups × 5 passes × 3 runs"
    echo "# kind  N      lookups passes"
    ( cd "$PROJECT_DIR" && direnv exec . cargo build --release --example perf_isolate ) 2>&1 | tail -5
    for kind in tomb hb ufm ipo split ipo64; do
        for run in 1 2 3; do
            echo
            echo "=== $kind run $run ==="
            "$PROJECT_DIR/target/release/examples/perf_isolate" "$kind" 920000 50000 5 2>&1 || true
        done
    done
} >"$PERF_LOG" 2>&1
echo "  → $PERF_LOG"

if [[ "$SKIP_CRITERION" == false ]]; then
    echo "─── [4/5] criterion suite ($(date +%H:%M:%S)) ───"
    CRIT_LOG="$PRIMARY_DIR/criterion.log"
    # The criterion HTML reports go into target/criterion/<bench>/; we capture
    # the stdout summaries for archival next to the rest of the run.
    # Run benches individually so a failure in one doesn't take down the others.
    for bench in throughput load_factor construction dispatch raw_entry sets workloads tag_collision matrix btree distributions; do
        echo
        echo "=== cargo bench --bench $bench ==="
        ( cd "$PROJECT_DIR" && direnv exec . cargo bench --bench "$bench" ) 2>&1 || echo "(bench $bench failed — continuing)"
    done >"$CRIT_LOG" 2>&1
    # Mirror the criterion HTML root for self-containment
    if [[ -d "$PROJECT_DIR/target/criterion" ]]; then
        mkdir -p "$PRIMARY_DIR/criterion"
        cp -r "$PROJECT_DIR/target/criterion/." "$PRIMARY_DIR/criterion/" 2>/dev/null || true
    fi
    echo "  → $CRIT_LOG (+ $PRIMARY_DIR/criterion/)"
else
    echo "─── [4/5] criterion suite — SKIPPED ───"
fi

# Done with the primary run dir
unset BENCH_RUN_DIR
ln -sfn "runs/$(basename "$PRIMARY_DIR")" "$RESULTS_DIR/latest"

# ── 2. All-design archive sweep ──────────────────────────────────────────────
if [[ "$SKIP_ALL" == false ]]; then
    echo "─── [5/5] all-design archive sweep ($(date +%H:%M:%S)) ───"
    mkdir -p "$ALL_DIR/plots"
    export BENCH_RUN_DIR="$ALL_DIR"
    "$SCRIPT_DIR/sweep-bench.sh" --design-set all "${MAX_N_ARGS[@]}"
    unset BENCH_RUN_DIR
    # latest stays on the primary (curated) run — the all-design data is for
    # archive comparisons, not for the blog post's headline figures.
else
    echo "─── [5/5] all-design archive — SKIPPED ───"
fi

echo
echo "═══ Overnight bench complete: $(date) ═══"
echo "Primary:   $PRIMARY_DIR"
[[ "$SKIP_ALL" == false ]] && echo "Archive:   $ALL_DIR"
echo "Latest -> $(readlink "$RESULTS_DIR/latest")"
