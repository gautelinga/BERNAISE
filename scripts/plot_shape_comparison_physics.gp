set terminal epslatex color standalone size 10cm,10cm

set style line 1 lt 1 lw 3 lc rgb '#F7FBFF' # very light blue
set style line 2 lt 1 lw 3 lc rgb '#DEEBF7' # 
set style line 3 lt 1 lw 3 lc rgb '#C6DBEF' # 
set style line 4 lt 1 lw 3 lc rgb '#9ECAE1' # light blue
set style line 5 lt 1 lw 3 lc rgb '#6BAED6' # 
set style line 6 lt 1 lw 3 lc rgb '#4292C6' # medium blue
set style line 7 lt 1 lw 3 lc rgb '#2171B5' #
set style line 8 lt 1 lw 3 lc rgb '#084594' # dark blue

set key samplen 2 maxrows 4 top left

set size ratio -1

set xlabel '$x$'
set ylabel '$y$'

set output "shape_comparison_physics.tex"
prefix8 = "shape_comparison_dx_time8/contour_dx"
prefix4 = "shape_comparison_dx_time4/contour_dx"

set xrange [0.5:2]
set yrange [0:1]

set xtics 0.5
set ytics 0.5

set mxtics 5
set mytics 5

set label at 0.95,0.5 "$t=4$"
set label at 1.68,0.5 "$t=8$"

pl prefix8.'0.02.dat' u 1:2 w l t '0.02' ls 2, \
prefix8.'0.01.dat' u 1:2 w l t '0.01' ls 4, \
prefix8.'0.005.dat' u 1:2 w l t '0.005' ls 6, \
prefix8.'0.0025.dat' u 1:2 w l t '0.0025' ls 8,\
prefix4.'0.02.dat' u 1:2 w l t '' ls 2, \
prefix4.'0.01.dat' u 1:2 w l t '' ls 4, \
prefix4.'0.005.dat' u 1:2 w l t '' ls 6, \
prefix4.'0.0025.dat' u 1:2 w l t '' ls 8
