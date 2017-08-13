set terminal epslatex color standalone size 10cm,8cm 
set output "convergens_in_time_intrusion_bulk.tex"
set log
set format x "$10^{%T}$"
set format y "$10^{%T}$"
set xlabel "$\\Delta t$"
set ylabel "Error norm in $L_2$"
set key top left  
set style line 1 lc rgb '#000000' lt 2 lw 4 pt 7 ps 1.5 
set style line 2 lc rgb '#0060ad' lt 2 lw 4 pt 7 ps 1.5 

pl 'errosconvergensintrusionbulk.dat' index 1 u 5:2 with linespoints ls 2  t "data", x**1 ls 1 t "$\\sim \\Delta t^1$"


set output "convergens_in_space_intrusion_bulk.tex"
set xlabel "$\\Delta x$"
set key top left  

pl 'errosconvergensintrusionbulk.dat' index 0 u 4:2 w lp ls 2 t  "data", x**2 ls 1 t "$\\sim \\Delta x^2$"