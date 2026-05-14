# Sweep benchmark plotter — generates one PNG per operation from sweep CSV.
#
# Usage:
#   gnuplot -e "csv='bench-results/sweep-latest.csv'; outdir='bench-results'" scripts/sweep-plot.gp
#
# Variables (set via -e or defaults):
#   csv    — path to sweep CSV file
#   outdir — directory for output PNGs
#
# Designs plotted = the curated 12 from `benches/sweep.rs`. Display labels are
# cluster-representative names; underlying Rust types in src/ keep their brand
# names (UnorderedFlatMap, InPlaceOverflow, etc.).
#
#   Tomb cluster:    Tomb (Byte7_128, hashbrown-tag), TombWide (Byte7_254),
#                    Tomb64 (AVX-512), TombSoa (SoA flavor)
#   OvInline family: OvInline (UFM), OvInlineGaps (Gaps p2-stride),
#                    OvInline32 (AVX2 wide)
#   OvSplit:         16-slot separate-overflow (Splitsies)
#   Sorted:          FlatBTree, std::BTreeMap
#   Baseline:        hashbrown
#   Wrapper:         OptiMap (Auto policy)
#
# Designs not in the CSV are silently skipped by gnuplot (the awk filter
# yields no rows). Running a non-curated sweep is fine — only the matching
# lines render.

if (!exists("csv"))    csv    = "bench-results/sweep-latest.csv"
if (!exists("outdir")) outdir = "bench-results"

set datafile separator ","
set terminal pngcairo size 1400,900 enhanced font "sans,12"

set xlabel "N (elements)"
set ylabel "ns/op"
set logscale x 10
set logscale y 10
set format x "10^{%T}"
set grid xtics ytics lt 0 lw 0.5 lc rgb "#dddddd"
set key top left font ",9" spacing 1.1

# ── Color/style scheme ──────────────────────────────────────────────────────
# Cluster encoded by hue family; sub-variant by dash pattern:
#   Tomb family    → warm tones (red/purple/orange)
#   OvInline family → cool tones (blue/green)
#   OvSplit         → red
#   Sorted          → grey/black, long-dashed
#   Baselines       → bold black
#   Wrapper         → teal, bold

# Tomb family (warm tones)
set linetype  1 lc rgb "#b2182b" lw 2.0 dt solid     # Tomb         — dark red, bold (canonical)
set linetype  2 lc rgb "#d6604d" lw 1.5 dt 2         # TombWide     — lighter red, dashed
set linetype  3 lc rgb "#e08214" lw 1.5 dt solid     # Tomb64       — orange
set linetype  4 lc rgb "#7b3294" lw 1.5 dt 4         # TombSoa      — purple, long-dashed
# OvInline family (cool tones)
set linetype  5 lc rgb "#2166ac" lw 2.0 dt solid     # OvInline     — blue, bold (canonical)
set linetype  6 lc rgb "#1b7837" lw 1.5 dt 2         # OvInlineGaps — green, dashed
set linetype  7 lc rgb "#4393c3" lw 1.5 dt 3         # OvInline32   — sky blue, dash-dot
# OvSplit
set linetype  8 lc rgb "#d73027" lw 1.5 dt 5         # OvSplit      — red, dash-dot-dot
# Baselines + wrapper
set linetype  9 lc rgb "#1a1a1a" lw 2.5 dt solid     # hashbrown    — bold black (reference)
set linetype 10 lc rgb "#00838f" lw 2.0 dt solid     # OptiMap      — teal, bold
# Trees (slower curves, distinct dash)
set linetype 11 lc rgb "#525252" lw 1.5 dt 4         # FlatBTree    — grey, long-dashed
set linetype 12 lc rgb "#000000" lw 1.5 dt 5         # std::BTreeMap — black, dash-dot

# ── Per-operation plots ──────────────────────────────────────────────────────

operations = "insert lookup_hit lookup_miss remove iterate"

do for [i=1:words(operations)] {
    op = word(operations, i)

    if (op eq "insert")      { t = "Insert" }
    if (op eq "lookup_hit")  { t = "Lookup Hit" }
    if (op eq "lookup_miss") { t = "Lookup Miss" }
    if (op eq "remove")      { t = "Remove" }
    if (op eq "iterate")     { t = "Iterate" }

    set output sprintf("%s/%s.png", outdir, op)
    set title sprintf("%s — ns/op vs N", t)

    plot \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"Tomb\"'          '%s'", op, csv) using 3:4 with lines lt  1 title "Tomb", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"TombWide\"'      '%s'", op, csv) using 3:4 with lines lt  2 title "TombWide", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"Tomb64\"'        '%s'", op, csv) using 3:4 with lines lt  3 title "Tomb64", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"TombSoa\"'       '%s'", op, csv) using 3:4 with lines lt  4 title "TombSoa", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OvInline\"'      '%s'", op, csv) using 3:4 with lines lt  5 title "OvInline", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OvInlineGaps\"'  '%s'", op, csv) using 3:4 with lines lt  6 title "OvInlineGaps", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OvInline32\"'    '%s'", op, csv) using 3:4 with lines lt  7 title "OvInline32", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OvSplit\"'       '%s'", op, csv) using 3:4 with lines lt  8 title "OvSplit", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"hashbrown\"'     '%s'", op, csv) using 3:4 with lines lt  9 title "hashbrown", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OptiMap\"'       '%s'", op, csv) using 3:4 with lines lt 10 title "OptiMap (Auto)", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"FlatBTree\"'     '%s'", op, csv) using 3:4 with lines lt 11 title "FlatBTree", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"std::BTreeMap\"' '%s'", op, csv) using 3:4 with lines lt 12 title "std::BTreeMap"
}
