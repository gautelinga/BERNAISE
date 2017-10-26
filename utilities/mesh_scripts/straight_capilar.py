""" straight_capilar script. """
from common import info
import dolfin as df
import os
from generate_mesh import MESHES_DIR, store_mesh_HDF5
import mshr


def description(**kwargs):
    info("""
    Function that generates a mesh for a straight capilar, default
    meshing method is dolfin's "RectangleMesh" but has an option for
    mshr.

    Note: The generated mesh is stored in "BERNAISE/meshes/".
    """)


def method(res=10, height=1, length=5, use_mshr=False, **kwargs):
    '''
    Function that generates a mesh for a straight capilar, default
    meshing method is dolfin's "RectangleMesh" but has an option for
    mshr.

    Note: The generated mesh is stored in "BERNAISE/meshes/".

    '''
    if use_mshr:  # use mshr for the generation
        info("Generating mesh using the mshr tool")
        # Define coners of Rectangle
        a = df.Point(0, 0)
        b = df.Point(height, length)
        domain = mshr.Rectangle(a, b)
        mesh = mshr.generate_mesh(domain, res)
        meshpath = os.path.join(MESHES_DIR,
                                "StraightCapilarMshr_h" + str(height) + "_l" +
                                str(length) + "_res" + str(res))
        info("Done.")
    else:  # use the Dolfin built-in function
        info("Generating mesh using the Dolfin built-in function.")
        # Define coners of rectangle/capilar
        a = df.Point(0, 0)
        b = df.Point(height, length)
        # Setting the reselution
        if height <= length:
            num_points_height = res
            num_points_length = res*int(length/height)
        else:
            num_points_height = res*int(height/length)
            num_points_length = res
        mesh = df.RectangleMesh(a, b, num_points_height, num_points_length)
        meshpath = os.path.join(MESHES_DIR,
                                "StraightCapilarDolfin_h" +
                                str(height) + "_l" +
                                str(length) + "_res" + str(res))
    store_mesh_HDF5(mesh, meshpath)
