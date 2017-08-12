set terminal epslatex color standalone size 10cm,5cm

unset key

load 'blues.pal'

set size ratio -1

print "Dataset:"
folder = system('read hei; echo $hei')
print "Stop time:"
T = system('read hei; echo $hei')

file = folder."/Plots/shape_evolution.tex"
set output file

files = folder.'/Analysis/contour/contour_'

N = system('ls '.files.'*.dat | wc -l')-1

set xrange [0:2]
set yrange [0:1]
set cbrange[0:T]

set xlabel "$x$"
set ylabel "$y$"
set cblabel "Time $t$"

set xtics 1.0
set ytics 1.0
set cbtics format "%1.1f"

set mxtics 10
set mytics 10

plot for [i=0:N:N/5] sprintf(files.'%06d.dat', i) u 1:2 w l lt palette frac i*1./N lw 3

