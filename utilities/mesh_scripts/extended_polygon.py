""" extended_polygon script. """
from generate_mesh import MESHES_DIR, make_polygon, store_mesh_HDF5, \
    numpy_to_dolfin
import numpy as np
import os
from meshpy import triangle as tri
from utilities.plot import plot_edges, plot_faces
import dolfin as df
from common import info


def description(**kwargs):
    info("")


def method(Lx=1., Ly=1., scale=0.75, dx=0.02, do_plot=True,
           polygon="dolphin", center=(0.5, 0.5), **kwargs):
    edges = np.loadtxt(os.path.join(MESHES_DIR, polygon + ".edges"),
                       dtype=int).tolist()
    nodes = np.loadtxt(os.path.join(MESHES_DIR, polygon + ".nodes"))

    nodes[:, 0] -= 0.5*np.max(nodes[:, 0])
    nodes[:, 1] -= 0.5*np.max(nodes[:, 1])
    nodes[:, :] *= scale
    nodes[:, 0] += center[0]*Lx
    nodes[:, 1] += center[1]*Ly

    nodes = nodes.tolist()

    x_min, x_max = 0., Lx
    y_min, y_max = 0., Ly

    corner_pts = [(x_min, y_min),
                  (x_max, y_min),
                  (x_max, y_max),
                  (x_min, y_max)]

    outer_nodes, outer_edges = make_polygon(corner_pts, dx, len(nodes))
    nodes.extend(outer_nodes)
    edges.extend(outer_edges)

    plot_edges(nodes, edges)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)
    mi.set_holes([(center[0]*Lx, center[1]*Ly)])

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    if do_plot:
        plot_faces(coords, faces)

    mesh = numpy_to_dolfin(coords, faces)

    if do_plot:
        df.plot(mesh)
        df.interactive()

    mesh_path = os.path.join(MESHES_DIR,
                             polygon + "_dx" + str(dx))
    store_mesh_HDF5(mesh, mesh_path)
