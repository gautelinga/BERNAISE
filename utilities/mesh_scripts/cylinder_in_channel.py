""" Cylinder in channel script. """
from common import info
import dolfin as df
import mshr
import os
from generate_mesh import MESHES_DIR, store_mesh_HDF5
import matplotlib.pyplot as plt


def description(**kwargs):
    info("Generates mesh for a cylinder in 2D channel.")


def method(res=96, H=0.41, L=2.2, x=0.2, y=0.2, r=0.05,
           segments=100, show=False, **kwargs):
    '''
    Generates mesh for the 'Flow around cylinder' benchmark:

    http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow.html

    Note: The generated mesh is stored in "BERNAISE/meshes/".
    '''
    info("Generating mesh using the mshr tool.")

    rect = mshr.Rectangle(df.Point(0, 0), df.Point(L, H))
    cyl = mshr.Circle(df.Point(x, y), r, segments=segments)
    domain = rect - cyl
    mesh = mshr.generate_mesh(domain, res)
    meshpath = os.path.join(
        MESHES_DIR,
        "cylinderinchannel_H{}_L{}_x{}_y{}_r{}_res{}".format(
            H, L, x, y, r, res))
    store_mesh_HDF5(mesh, meshpath)

    if show:
        df.plot(mesh)
        plt.show()
