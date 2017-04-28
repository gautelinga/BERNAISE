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

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    P = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, V*P)


def main():
     #mesh = LoadMesh() 
     #df.plot(mesh)
     #df.interactive()

if __name__ == "__main__":
    main()