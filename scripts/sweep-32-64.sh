#!/usr/bin/env bash
# Sweep benchmark: 32/64-slot variants vs 16-slot baselines
# Focuses on the best candidates from the matrix criterion bench.
# Saves to bench-results/sweep-32-64-YYYY-MM-DD-HHMMSS.csv

set -euo pipefail

MAX_N="${MAX_N:-2000000}"
TRIALS="${TRIALS:-3}"

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
OUTFILE="bench-results/sweep-32-64-${TIMESTAMP}.csv"
mkdir -p bench-results

# Map of labels to type names
declare -A DESIGNS=(
    # 16-slot baselines
    [hashbrown]="hashbrown::HashMap<u64,u64>"
    [Hi128_Tomb]="Hi128_TombMap<u64,u64>"
    [UFM]="UnorderedFlatMap<u64,u64>"
    [Gaps]="Gaps<u64,u64>"
    [Splitsies]="Splitsies<u64,u64>"
    [Top128_1bitAnd]="Top128_1bitAndMap<u64,u64>"
    # 16-slot embedded-overflow variants (non-baseline tags)
    [Hi8_EmbP2]="Hi8_EmbP2Map<u64,u64>"
    [Top128_EmbP2And]="Top128_EmbP2AndMap<u64,u64>"
    [Top255_EmbP2And]="Top255_EmbP2AndMap<u64,u64>"
    # 32-slot separate-overflow
    [Splitsies32]="Splitsies32Map<u64,u64>"
    [Top128_1bitAnd32]="Top128_1bitAnd32Map<u64,u64>"
    [Top255_1bitAnd32]="Top255_1bitAnd32Map<u64,u64>"
    [Lo128_8bit32]="Lo128_8bit32Map<u64,u64>"
    [Top128_8bitAnd32]="Top128_8bitAnd32Map<u64,u64>"
    # 32-slot embedded-overflow
    [Ufm32]="Ufm32Map<u64,u64>"
    [Gaps32]="Gaps32Map<u64,u64>"
    [Top128_EmbP2And32]="Top128_EmbP2And32Map<u64,u64>"
    [Top255_EmbP2And32]="Top255_EmbP2And32Map<u64,u64>"
    [Hi8_EmbP232]="Hi8_EmbP232Map<u64,u64>"
    # 64-slot separate-overflow
    [Splitsies64]="Splitsies64Map<u64,u64>"
    [Top255_1bitAnd64]="Top255_1bitAnd64Map<u64,u64>"
    [Lo128_8bit64]="Lo128_8bit64Map<u64,u64>"
    [Top128_8bitAnd64]="Top128_8bitAnd64Map<u64,u64>"
    # 64-slot embedded-overflow
    [Ufm64]="Ufm64Map<u64,u64>"
    [Gaps64]="Gaps64Map<u64,u64>"
    [Top128_EmbP2And64]="Top128_EmbP2And64Map<u64,u64>"
    [Top255_EmbP2And64]="Top255_EmbP2And64Map<u64,u64>"
)

echo "Sweep 32/64-slot: max_n=$MAX_N, trials=$TRIALS, ${#DESIGNS[@]} designs"
echo "Output: $OUTFILE"

# Header
echo "operation,design,n,ns_per_op" > "$OUTFILE"

for label in "${!DESIGNS[@]}"; do
    echo ""
    echo "=== $label ==="
    cargo bench --bench sweep -- \
        --max-n "$MAX_N" --trials "$TRIALS" --design "$label" 2>/dev/null \
        >> "$OUTFILE"
done

echo ""
echo "Done. Results in $OUTFILE"
echo "Rows: $(wc -l < "$OUTFILE")"
