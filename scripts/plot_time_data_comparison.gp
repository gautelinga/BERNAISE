set terminal epslatex color standalone size 12cm,6cm

set style line 1 lt 3 lw 3 lc rgb '#F7FBFF' # very light blue
set style line 2 lt 1 lw 3 lc rgb '#DEEBF7' # 
set style line 3 lt 4 lw 3 lc rgb '#C6DBEF' # 
set style line 4 lt 1 lw 3 lc rgb '#9ECAE1' # light blue
set style line 5 lt 5 lw 3 lc rgb '#6BAED6' # 
set style line 6 lt 1 lw 3 lc rgb '#4292C6' # medium blue
set style line 7 lt 6 lw 3 lc rgb '#2171B5' #
set style line 8 lt 1 lw 3 lc rgb '#084594' # dark blue

set key samplen 2 top left font ",18"

set ylabel 'Circumference $\ell$ (solid)'
set xlabel 'Time $t$'
set y2label 'Center of mass $x$ (dashed)'

set output "time_data_comparison_physics.tex"
prefix = "time_data_comparison_dx/time_data_dx"

#set xrange [0.5:2]
#set yrange [0:1]

#set xtics 0.5
#set ytics 0.5

#set mxtics 5
#set mytics 5

#set label at 0.95,0.5 "$t=4$"
#set label at 1.68,0.5 "$t=8$"

set ytics nomirror
set y2tics

pl prefix.'0.02.dat' u 2:3 w l t '$\Delta x = 0.02$' ls 2, \
prefix.'0.01.dat' u 2:3 w l t '0.01' ls 4, \
prefix.'0.005.dat' u 2:3 w l t '0.005' ls 6, \
prefix.'0.0025.dat' u 2:3 w l t '0.0025' ls 8,\
prefix.'0.02.dat' u 2:5 axes x1y2 w l t '' ls 2 dt 2, \
prefix.'0.01.dat' u 2:5 axes x1y2 w l t '' ls 4 dt 2, \
prefix.'0.005.dat' u 2:5 axes x1y2 w l t '' ls 6 dt 2, \
prefix.'0.0025.dat' u 2:5 axes x1y2 w l t '' ls 8 dt 2
