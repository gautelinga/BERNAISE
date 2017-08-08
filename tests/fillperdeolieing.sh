cd ..
pwd 
coresavialbol=$(nproc)
echo "Number of computer cores [Max with HyperThreading(TM): $coresavialbol]: "
read cores 

pressure_left=0

echo Hourglass. Pressure Left $pressure_left both charge and none charge
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=-10 T=50
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=0 T=50

pressure_left=5

echo Hourglass. Pressure Left $pressure_left both charge and none charge
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=-10 T=50
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=0 T=50

pressure_left=20

echo Hourglass. Pressure Left $pressure_left both charge and none charge
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=-10 T=50 dt=0.01
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=0 T=50 dt=0.01

pressure_left=50

echo Hourglass. Pressure Left $pressure_left both charge and none charge
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=-10 T=50 dt=0.005 
mpiexec -n $cores python sauce.py problem=dolphin  pressure_left=$pressure_left surface_charge=0 T=50 dt=0.005
