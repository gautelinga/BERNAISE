cd ..
pwd 
coresavialbol=$(nproc)
echo "Number of computer cores [Max with HyperThreading(TM): $coresavialbol]: "
read cores 

 
interface_thickness=0.030
pf_mobility_coeff=0.000020
invers_grid_spacing=8

dt=0.000078125  
DT = 0.04
## Space convergens

## Spacing 1/8, Time 0.000078125
grid_spacing=$(bc <<< "scale = 16; (1. / $invers_grid_spacing)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 


## New Spacing 1/16, Time 0.000078125
grid_spacing=$(bc <<< "scale = 16; ($grid_spacing / 2.0)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## New Spacing 1/32, Time 0.000078125
grid_spacing=$(bc <<< "scale = 8; ($grid_spacing / 2.0)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## New Spacing 1/64, Time 0.000078125
grid_spacing=$(bc <<< "scale = 8; ($grid_spacing / 2.0)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## New Spacing 1/128, Time 0.000078125
grid_spacing=$(bc <<< "scale = 8; ($grid_spacing / 2.0)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## New Spacing 1/256, Time 0.000078125
grid_spacing=$(bc <<< "scale = 8; ($grid_spacing / 2.0)") 
grid_spacing=0$grid_spacing

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=$DT dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 


## Spacing 1/128, New Time 0.00015625 

grid_spacing=$(bc <<< "scale = 8; ($grid_spacing * 2.0)") 
grid_spacing=0$grid_spacing
dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.0003125 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.000625 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.00125 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.0025 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.005 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.01 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.02 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 


## Spacing 1/128, New Time 0.04 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 


## Spacing 1/128, New Time 0.08 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 

## Spacing 1/128, New Time 0.16 

dt=$(bc <<< "scale = 8; ($dt * 2.0)") 
dt=0$dt

echo Solving the intrusion_bulk problem. 
echo Time step: $dt 
echo Grid_spacing: $grid_spacing

mpiexec -n $cores python sauce.py problem=intrusion_bulk T=0.64 dt=$dt grid_spacing=$grid_spacing interface_thickness=$interface_thickness pf_mobility_coeff=$pf_mobility_coeff viscosity=[100.0,1.0] initial_interface="flat" 