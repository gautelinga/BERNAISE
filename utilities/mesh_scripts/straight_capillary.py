""" straight_capilar script. """
import os
import dolfin as df
from generate_mesh import MESHES_DIR, store_mesh_HDF5
import mshr
import matplotlib.pyplot as plt
from common import info


def description(**kwargs):
    info("""
    Function that generates a mesh for a straight capilar, default
    meshing method is dolfin's "RectangleMesh" but has an option for
    mshr.

    Note: The generated mesh is stored in "BERNAISE/meshes/".
    """)


def method(res=10, height=1, length=5, use_mshr=False, show=False, **kwargs):
    '''
    Function that generates a mesh for a straight capillary, default
    meshing method is Dolfin's RectangleMesh, but there is an option for
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
        meshpath = os.path.join(
            MESHES_DIR,
            "straight_capillary_mshr_h{}_l{}_res{}".format(
                height, length, res))
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
        meshpath = os.path.join(
            MESHES_DIR,
            "straight_capillary_dolfin_h{}_l{}_res{}".format(
                height, length, res))
    store_mesh_HDF5(mesh, meshpath)

    if show:
        df.plot(mesh)
        plt.show()
