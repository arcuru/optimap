#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 -p gnuplot
"""Generate gnuplot PNG plots for 32/64-slot sweep data."""
import subprocess, glob, os, sys

# Find newest sweep-32-64 file
files = sorted(glob.glob("bench-results/sweep-32-64-*.csv"))
if not files:
    print("No sweep-32-64 files found", file=sys.stderr)
    sys.exit(1)
fname = files[-1]
print(f"Plotting: {fname}", file=sys.stderr)

ops = ['insert', 'lookup_hit', 'lookup_miss', 'remove']

for op in ops:
    gp_script = f'''
set terminal pngcairo size 1400,900 enhanced font 'DejaVu Sans,11'
set output 'bench-results/32-64-{op}.png'
set title '{op} — 32/64-slot vs 16-slot sweep'
set xlabel 'N (entries)'
set ylabel 'ns/op'
set logscale xy
set grid
set key outside right top

# Filter data for this operation
datafile = '{fname}'

# Baseline 16-slot designs
plot \\
    '< grep "^{op}," {fname} | grep ",hashbrown,"'  using 3:4 with lines lw 2.5 lc rgb '#000000' title 'hashbrown', \\
    '< grep "^{op}," {fname} | grep ",Hi128_Tomb,"'   using 3:4 with lines lw 2 lc rgb '#444444' title 'Hi128_Tomb', \\
    '< grep "^{op}," {fname} | grep ",UFM,"'          using 3:4 with lines lw 2 lc rgb '#e41a1c' title 'UFM (16)', \\
    '< grep "^{op}," {fname} | grep ",Gaps,"'         using 3:4 with lines lw 2 lc rgb '#377eb8' title 'Gaps (16)', \\
    '< grep "^{op}," {fname} | grep ",Splitsies,"'    using 3:4 with lines lw 2 lc rgb '#4daf4a' title 'Splitsies (16)', \\
    '< grep "^{op}," {fname} | grep ",Top128_1bitAnd,"' using 3:4 with lines lw 1.5 lc rgb '#984ea3' title 'Top128_1bitAnd (16)', \\
    '< grep "^{op}," {fname} | grep ",Top255_EmbP2And,"' using 3:4 with lines lw 1.5 lc rgb '#ff7f00' title 'Top255_EmbP2And (16)', \\
    '< grep "^{op}," {fname} | grep ",Ufm32,"'        using 3:4 with lines lw 2.5 dt 2 lc rgb '#e41a1c' title 'Ufm32', \\
    '< grep "^{op}," {fname} | grep ",Splitsies32,"'  using 3:4 with lines lw 2 dt 2 lc rgb '#4daf4a' title 'Splitsies32', \\
    '< grep "^{op}," {fname} | grep ",Top128_1bitAnd32,"' using 3:4 with lines lw 1.5 dt 2 lc rgb '#984ea3' title 'Top128_1bitAnd32', \\
    '< grep "^{op}," {fname} | grep ",Top255_EmbP2And32,"' using 3:4 with lines lw 1.5 dt 2 lc rgb '#ff7f00' title 'Top255_EmbP2And32', \\
    '< grep "^{op}," {fname} | grep ",Gaps32,"'       using 3:4 with lines lw 1.5 dt 2 lc rgb '#377eb8' title 'Gaps32', \\
    '< grep "^{op}," {fname} | grep ",Splitsies64,"'  using 3:4 with lines lw 1.5 dt 3 lc rgb '#4daf4a' title 'Splitsies64', \\
    '< grep "^{op}," {fname} | grep ",Gaps64,"'       using 3:4 with lines lw 1.5 dt 3 lc rgb '#377eb8' title 'Gaps64', \\
    '< grep "^{op}," {fname} | grep ",Top255_1bitAnd64,"' using 3:4 with lines lw 1.5 dt 3 lc rgb '#a65628' title 'Top255_1bitAnd64'
'''

    try:
        subprocess.run(['gnuplot', '-e', gp_script], check=True, capture_output=True, text=True)
        print(f"  Generated bench-results/32-64-{op}.png", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  Error generating {op}: {e.stderr}", file=sys.stderr)

print("Done.", file=sys.stderr)
