""" snoevsen script. """
from common import info
import mshr
import dolfin as df
import numpy as np
import os
from generate_mesh import MESHES_DIR, store_mesh_HDF5
import matplotlib.pyplot as plt


def description(**kwargs):
    info("Snoevsen mesh.")


def method(L=3., H=1., R=0.3, n_segments=40, res=60, show=False, **kwargs):
    """
    Generates mesh of Snausen/Snoevsen.
    """
    info("Generating mesh of Snoevsen.")

    # Define points:
    pt_A = df.Point(0., 0.)
    pt_B = df.Point(L, H)
    pt_C = df.Point(L/2-2*R, 0.)
    pt_D = df.Point(L/2+2*R, 0.)
    pt_E = df.Point(L/2-2*R, -R)
    pt_F = df.Point(L/2+2*R, -R)
    pt_G = df.Point(L/2, -R)

    tube = mshr.Rectangle(pt_A, pt_B)
    add_rect = mshr.Rectangle(pt_E, pt_D)
    neg_circ_L = mshr.Circle(pt_E, R, segments=n_segments)
    neg_circ_R = mshr.Circle(pt_F, R, segments=n_segments)
    pos_circ = mshr.Circle(pt_G, R, segments=n_segments)

    domain = tube + add_rect - neg_circ_L - neg_circ_R + pos_circ

    mesh = mshr.generate_mesh(domain, res)

    # check that mesh is periodic along x
    boun = df.BoundaryMesh(mesh, "exterior")
    y = boun.coordinates()[:, 1]
    y = y[y < 1.]
    y = y[y > 0.]
    n_y = len(y)
    n_y_unique = len(np.unique(y))

    assert(n_y == 2*n_y_unique)

    mesh_path = os.path.join(MESHES_DIR,
                             "snoevsen_res" + str(res))
    store_mesh_HDF5(mesh, mesh_path)

    if show:
        df.plot(mesh, "Mesh")
        plt.show()
