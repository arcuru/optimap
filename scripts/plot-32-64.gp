set datafile separator ','
set terminal pngcairo size 1400,900
set logscale xy
set grid
set key outside right top
set xlabel 'N (entries)'
set ylabel 'ns/op'

# Insert
set output 'bench-results/32-64-insert.png'
set title 'insert — 32/64 vs 16-slot (N=100..2M)'
plot '/tmp/sweep32_insert_hashbrown.csv' u 1:2 w l lw 2.5 lc '#000000' t 'hashbrown', \
     '/tmp/sweep32_insert_Hi128_Tomb.csv' u 1:2 w l lw 2 lc '#666666' t 'Hi128_Tomb', \
     '/tmp/sweep32_insert_UFM.csv' u 1:2 w l lw 2 lc '#e41a1c' t 'UFM(16)', \
     '/tmp/sweep32_insert_Gaps.csv' u 1:2 w l lw 2 lc '#377eb8' t 'Gaps(16)', \
     '/tmp/sweep32_insert_Ufm32.csv' u 1:2 w l lw 2 dt 2 lc '#e41a1c' t 'Ufm32', \
     '/tmp/sweep32_insert_Gaps32.csv' u 1:2 w l lw 2 dt 2 lc '#377eb8' t 'Gaps32', \
     '/tmp/sweep32_insert_Splitsies32.csv' u 1:2 w l lw 1.5 dt 2 lc '#a65628' t 'Splitsies32', \
     '/tmp/sweep32_insert_Top128_1bitAnd32.csv' u 1:2 w l lw 1.5 dt 2 lc '#984ea3' t 'Top128_1bitAnd32', \
     '/tmp/sweep32_insert_Splitsies64.csv' u 1:2 w l lw 1.5 dt 3 lc '#a65628' t 'Splitsies64', \
     '/tmp/sweep32_insert_Gaps64.csv' u 1:2 w l lw 1.5 dt 3 lc '#e41a1c' t 'Gaps64', \
     '/tmp/sweep32_insert_Top255_1bitAnd64.csv' u 1:2 w l lw 1.5 dt 3 lc '#ff7f00' t 'Top255_1bitAnd64'

# Lookup hit
set output 'bench-results/32-64-lookup_hit.png'
set title 'lookup_hit — 32/64 vs 16-slot (N=100..2M)'
plot '/tmp/sweep32_lookup_hit_hashbrown.csv' u 1:2 w l lw 2.5 lc '#000000' t 'hashbrown', \
     '/tmp/sweep32_lookup_hit_Hi128_Tomb.csv' u 1:2 w l lw 2 lc '#666666' t 'Hi128_Tomb', \
     '/tmp/sweep32_lookup_hit_UFM.csv' u 1:2 w l lw 2 lc '#e41a1c' t 'UFM(16)', \
     '/tmp/sweep32_lookup_hit_Gaps.csv' u 1:2 w l lw 2 lc '#377eb8' t 'Gaps(16)', \
     '/tmp/sweep32_lookup_hit_Ufm32.csv' u 1:2 w l lw 2 dt 2 lc '#e41a1c' t 'Ufm32', \
     '/tmp/sweep32_lookup_hit_Gaps32.csv' u 1:2 w l lw 2 dt 2 lc '#377eb8' t 'Gaps32', \
     '/tmp/sweep32_lookup_hit_Splitsies32.csv' u 1:2 w l lw 1.5 dt 2 lc '#a65628' t 'Splitsies32', \
     '/tmp/sweep32_lookup_hit_Top128_1bitAnd32.csv' u 1:2 w l lw 1.5 dt 2 lc '#984ea3' t 'Top128_1bitAnd32', \
     '/tmp/sweep32_lookup_hit_Splitsies64.csv' u 1:2 w l lw 1.5 dt 3 lc '#a65628' t 'Splitsies64', \
     '/tmp/sweep32_lookup_hit_Gaps64.csv' u 1:2 w l lw 1.5 dt 3 lc '#e41a1c' t 'Gaps64', \
     '/tmp/sweep32_lookup_hit_Top255_1bitAnd64.csv' u 1:2 w l lw 1.5 dt 3 lc '#ff7f00' t 'Top255_1bitAnd64'

# Lookup miss
set output 'bench-results/32-64-lookup_miss.png'
set title 'lookup_miss — 32/64 vs 16-slot (N=100..2M)'
plot '/tmp/sweep32_lookup_miss_hashbrown.csv' u 1:2 w l lw 2.5 lc '#000000' t 'hashbrown', \
     '/tmp/sweep32_lookup_miss_Hi128_Tomb.csv' u 1:2 w l lw 2 lc '#666666' t 'Hi128_Tomb', \
     '/tmp/sweep32_lookup_miss_UFM.csv' u 1:2 w l lw 2 lc '#e41a1c' t 'UFM(16)', \
     '/tmp/sweep32_lookup_miss_Gaps.csv' u 1:2 w l lw 2 lc '#377eb8' t 'Gaps(16)', \
     '/tmp/sweep32_lookup_miss_Ufm32.csv' u 1:2 w l lw 2 dt 2 lc '#e41a1c' t 'Ufm32', \
     '/tmp/sweep32_lookup_miss_Gaps32.csv' u 1:2 w l lw 2 dt 2 lc '#377eb8' t 'Gaps32', \
     '/tmp/sweep32_lookup_miss_Splitsies32.csv' u 1:2 w l lw 1.5 dt 2 lc '#a65628' t 'Splitsies32', \
     '/tmp/sweep32_lookup_miss_Top128_1bitAnd32.csv' u 1:2 w l lw 1.5 dt 2 lc '#984ea3' t 'Top128_1bitAnd32', \
     '/tmp/sweep32_lookup_miss_Splitsies64.csv' u 1:2 w l lw 1.5 dt 3 lc '#a65628' t 'Splitsies64', \
     '/tmp/sweep32_lookup_miss_Gaps64.csv' u 1:2 w l lw 1.5 dt 3 lc '#e41a1c' t 'Gaps64', \
     '/tmp/sweep32_lookup_miss_Top255_1bitAnd64.csv' u 1:2 w l lw 1.5 dt 3 lc '#ff7f00' t 'Top255_1bitAnd64'

# Remove
set output 'bench-results/32-64-remove.png'
set title 'remove — 32/64 vs 16-slot (N=100..2M)'
plot '/tmp/sweep32_remove_hashbrown.csv' u 1:2 w l lw 2.5 lc '#000000' t 'hashbrown', \
     '/tmp/sweep32_remove_Hi128_Tomb.csv' u 1:2 w l lw 2 lc '#666666' t 'Hi128_Tomb', \
     '/tmp/sweep32_remove_UFM.csv' u 1:2 w l lw 2 lc '#e41a1c' t 'UFM(16)', \
     '/tmp/sweep32_remove_Gaps.csv' u 1:2 w l lw 2 lc '#377eb8' t 'Gaps(16)', \
     '/tmp/sweep32_remove_Ufm32.csv' u 1:2 w l lw 2 dt 2 lc '#e41a1c' t 'Ufm32', \
     '/tmp/sweep32_remove_Gaps32.csv' u 1:2 w l lw 2 dt 2 lc '#377eb8' t 'Gaps32', \
     '/tmp/sweep32_remove_Splitsies32.csv' u 1:2 w l lw 1.5 dt 2 lc '#a65628' t 'Splitsies32', \
     '/tmp/sweep32_remove_Top128_1bitAnd32.csv' u 1:2 w l lw 1.5 dt 2 lc '#984ea3' t 'Top128_1bitAnd32', \
     '/tmp/sweep32_remove_Splitsies64.csv' u 1:2 w l lw 1.5 dt 3 lc '#a65628' t 'Splitsies64', \
     '/tmp/sweep32_remove_Gaps64.csv' u 1:2 w l lw 1.5 dt 3 lc '#e41a1c' t 'Gaps64', \
     '/tmp/sweep32_remove_Top255_1bitAnd64.csv' u 1:2 w l lw 1.5 dt 3 lc '#ff7f00' t 'Top255_1bitAnd64'
