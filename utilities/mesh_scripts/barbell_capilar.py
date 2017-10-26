""" barbell_capilar script. """
from common import info
import dolfin as df
import mshr
import os
from generate_mesh import MESHES_DIR, store_mesh_HDF5


def description(**kwargs):
    info("")


def method(res=50, diameter=1., length=5., **kwargs):
    '''
    Function That Generates a mesh for a barbell capilar,
    Meshing method is mshr.

    Note: The generarted mesh is stored in "BERNAISE/meshes/".
    '''
    info("Generating mesh using the mshr tool.")

    inletdiameter = diameter*5.
    inletlength = diameter*4.

    # Define coners of "capilar"
    a = df.Point(-diameter/2., -length/2-inletlength/2.)
    b = df.Point(diameter/2., length/2+inletlength/2.)
    capilar = mshr.Rectangle(a, b)
    # Define coners of "leftbell
    c = df.Point(-inletdiameter/2., -length/2-inletlength)
    d = df.Point(inletdiameter/2., -length/2)
    leftbell = mshr.Rectangle(c, d)
    # Define coners of "rightbell"
    e = df.Point(-inletdiameter/2., length/2)
    f = df.Point(inletdiameter/2., length/2+inletlength)
    rightbell = mshr.Rectangle(e, f)

    domain = capilar + leftbell + rightbell
    mesh = mshr.generate_mesh(domain, res)
    meshpath = os.path.join(MESHES_DIR,
                            "BarbellCapilarDolfin_d" + str(diameter) + "_l" +
                            str(length) + "_res" + str(res))
    store_mesh_HDF5(mesh, meshpath)
