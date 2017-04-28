__author__ = "Asger J. S Bolet <asgerbolet@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"

''' 
The "mesh=LoadMesh(filename)" Fuction
'''

import dolfin as df
import mshr as mshr
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def LoadMesh(filename = "meshes/StraightCapilarDolfin_h1_l5_res10.h5"):
    ''' 
    Function that loads in the mesh specified in the "filename".
    The defulat mesh loaded is the defulat generated mesh from 
    "meshgeneration.StraightCapilar()".
    Note: The "LoadMesh" should be done at the folder "BERNAISE/". 
    '''
    if rank == 0:
        print "load mesh from" + filename 
    mesh = df.Mesh()
    hdf5 = df.HDF5File(mesh.mpi_comm(), filename, "r")
    hdf5.read(mesh,"mesh",False)
    hdf5.close()
    if rank == 0:
        print "Done"
    return mesh 

def main():
     mesh = LoadMesh() 
     df.plot(mesh)
     df.interactive()

if __name__ == "__main__":
    main()
