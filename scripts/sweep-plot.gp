# Sweep benchmark plotter — generates one PNG per operation from sweep CSV.
#
# Usage:
#   gnuplot -e "csv='bench-results/sweep-latest.csv'; outdir='bench-results'" scripts/sweep-plot.gp
#
# Variables (set via -e or defaults):
#   csv    — path to sweep CSV file
#   outdir — directory for output PNGs

if (!exists("csv"))    csv    = "bench-results/sweep-latest.csv"
if (!exists("outdir")) outdir = "bench-results"

set datafile separator ","
set terminal pngcairo size 1200,800 enhanced font "sans,12"

set xlabel "N (elements)"
set ylabel "ns/op"
set logscale x 10
set logscale y 10
set format x "10^{%T}"
set grid xtics ytics lt 0 lw 0.5 lc rgb "#dddddd"
set key top left font ",10" spacing 1.2

# Design colors — consistent across all plots
# UFM=blue, Gaps=forest, Splitsies=red, IPO=purple, IPO64=orange, hashbrown=black, OptiMap=teal
set linetype 1 lc rgb "#2166ac" lw 1.5 dt solid   # UFM
set linetype 2 lc rgb "#1b7837" lw 1.5 dt solid   # Gaps
set linetype 3 lc rgb "#d6604d" lw 1.5 dt solid   # Splitsies
set linetype 4 lc rgb "#7b3294" lw 1.5 dt solid   # IPO
set linetype 5 lc rgb "#e08214" lw 1.5 dt solid   # IPO64
set linetype 6 lc rgb "#1a1a1a" lw 2.0 dt solid   # hashbrown (bold, reference)
set linetype 7 lc rgb "#00838f" lw 1.5 dt solid   # OptiMap

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
        sprintf("< awk -F, '$1==\"%s\" && $2==\"UFM\"'        '%s'", op, csv) using 3:4 with lines lt 1 title "UFM", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"Gaps\"'       '%s'", op, csv) using 3:4 with lines lt 2 title "Gaps", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"Splitsies\"'  '%s'", op, csv) using 3:4 with lines lt 3 title "Splitsies", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"IPO\"'        '%s'", op, csv) using 3:4 with lines lt 4 title "IPO", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"IPO64\"'      '%s'", op, csv) using 3:4 with lines lt 5 title "IPO64", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"hashbrown\"'  '%s'", op, csv) using 3:4 with lines lt 6 title "hashbrown", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"OptiMap\"'    '%s'", op, csv) using 3:4 with lines lt 7 title "OptiMap"
}
