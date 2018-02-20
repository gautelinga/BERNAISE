set terminal epslatex color standalone size 12cm,6cm

set style line 1 pt 1 lw 2 lc rgb '#084594' # very light blue
set style line 2 pt 2 lw 2 lc rgb '#084594' # 
set style line 3 pt 3 lw 2 lc rgb '#084594' # 
set style line 4 pt 4 lw 2 lc rgb '#084594' # light blue
set style line 5 pt 1 lw 2 lc rgb '#084594' # 
set style line 6 pt 6 lw 2 lc rgb '#084594' # medium blue
set style line 7 pt 10 lw 2 lc rgb '#084594' #
set style line 8 pt 8 lw 2 lc rgb '#084594' # dark blue
set style line 9 lt 1 lw 2 lc rgb '#000000' # black


set key samplen 2 top left font ",18"

set ylabel 'Phasefield $\phi$'
set xlabel 'Space $x$'

set output "space_data_comparison_intrusion_bulk.tex"
prefix = "intbulk/prope"

set xrange [0.7:1.3]
#set yrange [0:1]

#set xtics 0.5
#set ytics 0.5

#set mxtics 5
#set mytics 5

#set label at 0.95,0.5 "$t=4$"
#set label at 1.68,0.5 "$t=8$"

set ytics nomirror
set y2tics

pl prefix.'1.dat' every 2 u 1:4 w p t '$\Delta x = 0.125$' ls 2, \
prefix.'2.dat' every 2 u 1:4 w p t '0.0625' ls 3, \
prefix.'3.dat' every 2 u 1:4 w p t '0.03125' ls 4, \
prefix.'4.dat' every 2 u 1:4 w p t '0.015625' ls 5,\
prefix.'5.dat' every 2 u 1:4 w p t '0.0078125' ls 6,\
prefix.'6.dat' every 2 u 1:4 w p t '0.00390625' ls 7,\
 tanh((x**1 -1.004)/(sqrt(2)*0.03)) t "Analytical" ls 9


#set output "time_data_comparison_intrusion_bulk.tex"

#pl prefix.'14.dat' every 3 u 1:4 w p t '$\Delta t = 0.16$' ls 2, \
#prefix.'9.dat' every 3 u 1:4 w p t '0.08' ls 3, \
#prefix.'8.dat' every 3 u 1:4 w p t '0.04' ls 4, \
#prefix.'7.dat' every 3 u 1:4 w p t '0.02' ls 5, \
#prefix.'6.dat' every 3 u 1:4 w p t '0.01' ls 6, \
#prefix.'5.dat' every 3 u 1:4 w p t '0.005' ls 7, \
#prefix.'3.dat' every 3 u 1:4 w p t '0.0025' ls 8 , tanh((x**1 -3)/(sqrt(2)*0.03)) t "Analytical"  ls 9 