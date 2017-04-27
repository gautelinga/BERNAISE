import dolfin as df
import mshr as mshr
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def StoreMeshHDF5(mesh,meshpath):
    meshpathhdf5 = meshpath + ".h5"    
    hdf5 = df.HDF5File(mesh.mpi_comm(), meshpathhdf5, "w")
    if rank == 0:
        print 'Storing the mesh in /meshes' 
    hdf5.write(mesh,"mesh")
    hdf5.close()
    meshpathxdmf = meshpath + "_xdmf.xdmf"
    xdmff1 = df.XDMFFile(mesh.mpi_comm(), meshpathxdmf)
    xdmff1.write(mesh)
    if rank == 0:
        print 'Done.'


def StraightCapilar(res=10,height=1,length=5,usemshr=False): 
    if usemshr: # use mshr for the generation 
        if rank == 0:
            print "Genrating mesh using the mshr-tool" 
        # Define coners of Rectangle
        a = df.Point(0,0)
        b = df.Point(height,length)
        domain = mshr.Rectangle(a,b)
        mesh = mshr.generate_mesh(domain,res)
        meshpath ="../meshes/StraightCapilarMshr_h" + str(height) + "_l" + str(length) + "_res" + str(res)
        if rank == 0:
            print "Done."
    else: # use the dolfin build in function
        if rank == 0:
            print "Genrating mesh using the dolfin build in function"
        # Define coners of rectangle/capilar
        a = df.Point(0,0)
        b = df.Point(height,length)
        # Setting the reselution
        if height <= length:
            numberofpointsheight = res
            numberofpointslength = res*int(length/height)
        else:
            numberofpointsheight = res*int(height/length)
            numberofpointslength = res 
        mesh = df.RectangleMesh(a,b,numberofpointsheight,numberofpointslength)
        meshpath ="../meshes/StraightCapilarDolfin_h" + str(height) + "_l" + str(length) + "_res" + str(res)
        if rank == 0:
            print "Done."
    StoreMeshHDF5(mesh,meshpath)

def main():
     StraightCapilar() 

if __name__ == "__main__":
    main()

 