__author__ = "Asger J. S Bolet <asgerbolet@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"

"""
Prototype of the timedependet stokes solver and a basic problem.
"""

import dolfin as df
import mshr as mshr
from mpi4py import MPI
from utilities/meshload import LoadMesh 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def problem(meshname='defulat'): 
    '''
    Defination of the Finit Element Method (FEM) version 
    of the time dependen Stokes equation.  
    '''
    if filename == defulat:
        mesh = LoadMesh()
    else:
        mesh = LoadMesh(meshname)
    
    tetha = 0.5 # time-stepping family  
    dt = 0.1
    eta = 1

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    P = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, V*P)


def main():
     #mesh = LoadMesh() 
     #df.plot(mesh)
     #df.interactive()

if __name__ == "__main__":
    main()