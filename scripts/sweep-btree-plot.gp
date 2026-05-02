# Sorted-map sweep plotter — one PNG per operation from sweep-btree CSV.
#
# Usage:
#   gnuplot -e "csv='bench-results/sweep-btree-latest.csv'; outdir='bench-results'" \
#           scripts/sweep-btree-plot.gp
#
# Variables (set via -e or defaults):
#   csv    — path to sweep CSV file
#   outdir — directory for output PNGs

if (!exists("csv"))    csv    = "bench-results/sweep-btree-latest.csv"
if (!exists("outdir")) outdir = "bench-results"

set datafile separator ","
set terminal pngcairo size 1400,900 enhanced font "sans,12"

set xlabel "N (elements)"
set ylabel "ns/op"
set logscale x 10
set logscale y 10
set format x "10^{%T}"
set grid xtics ytics lt 0 lw 0.5 lc rgb "#dddddd"
set key top left font ",10" spacing 1.2

# Two designs only; keep colors high-contrast and consistent across plots.
set linetype 1 lc rgb "#1b7837" lw 2.0 dt solid   # FlatBTree (green, solid)
set linetype 2 lc rgb "#1a1a1a" lw 2.0 dt 2       # BTreeMap (black, dashed — reference)

# ── Per-operation plots ─────────────────────────────────────────────────────

operations = "insert lookup_hit lookup_miss remove iterate"

do for [i=1:words(operations)] {
    op = word(operations, i)

    if (op eq "insert")      { t = "Insert" }
    if (op eq "lookup_hit")  { t = "Lookup Hit" }
    if (op eq "lookup_miss") { t = "Lookup Miss" }
    if (op eq "remove")      { t = "Remove" }
    if (op eq "iterate")     { t = "Iterate" }

    set output sprintf("%s/btree-%s.png", outdir, op)
    set title sprintf("FlatBTree vs std::BTreeMap — %s — ns/op vs N", t)

    plot \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"FlatBTree\"' '%s'", op, csv) \
            using 3:4 with lines lt 1 title "FlatBTree", \
        sprintf("< awk -F, '$1==\"%s\" && $2==\"BTreeMap\"'  '%s'", op, csv) \
            using 3:4 with lines lt 2 title "std::BTreeMap"
}

# ── Combined ratio plot ──────────────────────────────────────────────────────
# FlatBTree / BTreeMap ratio per op — < 1.0 means FlatBTree wins.

set output sprintf("%s/btree-ratio.png", outdir)
set title "FlatBTree / std::BTreeMap ratio (ns/op) — < 1.0 = FlatBTree wins"
set ylabel "FlatBTree / BTreeMap ratio"
unset logscale y
set yrange [0:2]
set arrow from graph 0,first 1 to graph 1,first 1 nohead lc rgb "#888888" dt 3

# Op colors, all solid lines.
set linetype 1 lc rgb "#1b7837" lw 1.8 dt solid   # insert (green)
set linetype 2 lc rgb "#2166ac" lw 1.8 dt solid   # lookup_hit (blue)
set linetype 3 lc rgb "#7b3294" lw 1.8 dt solid   # lookup_miss (purple)
set linetype 4 lc rgb "#d6604d" lw 1.8 dt solid   # remove (red)
set linetype 5 lc rgb "#e08214" lw 1.8 dt solid   # iterate (orange)

plot \
    sprintf("< awk -F, 'NR>1 && $1==\"insert\"      {if ($2==\"FlatBTree\") f[$3]=$4; else if ($2==\"BTreeMap\" && f[$3]>0) print $3\",\"f[$3]/$4}' '%s'", csv) \
        using 1:2 with lines lt 1 title "insert", \
    sprintf("< awk -F, 'NR>1 && $1==\"lookup_hit\"  {if ($2==\"FlatBTree\") f[$3]=$4; else if ($2==\"BTreeMap\" && f[$3]>0) print $3\",\"f[$3]/$4}' '%s'", csv) \
        using 1:2 with lines lt 2 title "lookup_hit", \
    sprintf("< awk -F, 'NR>1 && $1==\"lookup_miss\" {if ($2==\"FlatBTree\") f[$3]=$4; else if ($2==\"BTreeMap\" && f[$3]>0) print $3\",\"f[$3]/$4}' '%s'", csv) \
        using 1:2 with lines lt 3 title "lookup_miss", \
    sprintf("< awk -F, 'NR>1 && $1==\"remove\"      {if ($2==\"FlatBTree\") f[$3]=$4; else if ($2==\"BTreeMap\" && f[$3]>0) print $3\",\"f[$3]/$4}' '%s'", csv) \
        using 1:2 with lines lt 4 title "remove", \
    sprintf("< awk -F, 'NR>1 && $1==\"iterate\"     {if ($2==\"FlatBTree\") f[$3]=$4; else if ($2==\"BTreeMap\" && f[$3]>0) print $3\",\"f[$3]/$4}' '%s'", csv) \
        using 1:2 with lines lt 5 title "iterate"
