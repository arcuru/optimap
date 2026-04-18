# Hash tag variant comparison plotter.
# Generates one PNG per (operation × base design) from sweep-hash-tag CSV.
#
# Usage:
#   gnuplot -e "csv='bench-results/sweep-hash-tag-latest.csv'; outdir='bench-results'" \
#       scripts/sweep-hash-tag-plot.gp
#
# Each plot shows 3 lines for the same base design: /asm, /128, /pure.

if (!exists("csv"))    csv    = "bench-results/sweep-hash-tag-latest.csv"
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

# Variant colors: asm=blue (default), 128=red, pure=gray
set linetype 1 lc rgb "#2166ac" lw 2.0 dt solid   # asm (default)
set linetype 2 lc rgb "#d6604d" lw 1.5 dt dash    # 128-value
set linetype 3 lc rgb "#888888" lw 1.5 dt "."     # pure Rust

operations = "insert lookup_hit lookup_miss remove iterate"
designs = "UFM Gaps Splitsies IPO IPO64 hashbrown OptiMap"

do for [i=1:words(operations)] {
    op = word(operations, i)

    if (op eq "insert")      { t = "Insert" }
    if (op eq "lookup_hit")  { t = "Lookup Hit" }
    if (op eq "lookup_miss") { t = "Lookup Miss" }
    if (op eq "remove")      { t = "Remove" }
    if (op eq "iterate")     { t = "Iterate" }

    do for [j=1:words(designs)] {
        d = word(designs, j)

        # Check if this design has data (skip silently if not)
        set output sprintf("%s/hash-tag-%s-%s.png", outdir, op, d)
        set title sprintf("%s — %s — hash\\_tag variant comparison", t, d)

        plot \
            sprintf("< awk -F, '$1==\"%s\" && $2==\"%s/asm\"'  '%s'", op, d, csv) using 3:4 with lines lt 1 title sprintf("%s/asm (2i, 255v)", d), \
            sprintf("< awk -F, '$1==\"%s\" && $2==\"%s/128\"'  '%s'", op, d, csv) using 3:4 with lines lt 2 title sprintf("%s/128 (1i, 128v)", d), \
            sprintf("< awk -F, '$1==\"%s\" && $2==\"%s/pure\"' '%s'", op, d, csv) using 3:4 with lines lt 3 title sprintf("%s/pure (3i, 255v)", d)
    }
}
