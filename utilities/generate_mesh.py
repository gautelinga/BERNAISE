__author__ = "Asger J. S Bolet <asgerbolet@gmail.com>, Gaute Linga <gaute.linga@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"
"""
Mesh generating functions in BERNAISE. 

Usage:
python generate_mesh.py mesh={mesh generating function} [+optional arguments]

"""
import mshr  # must be imported before dolfin!
import dolfin as df
import numpy as np
import os
import sys
# Find path to the BERNAISE root folder
bernaise_path = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-2])
# ...and append it to sys.path to get functionality from BERNAISE
sys.path.append(bernaise_path)
from common import parse_command_line, info, info_on_red, \
    remove_safe
from mpi4py.MPI import COMM_WORLD
import meshpy.triangle as tri
from plot import plot_edges, plot_faces
import h5py
from utilities import get_methods, get_help


# Directory to store meshes in
MESHES_DIR = os.path.join(bernaise_path, "meshes/")

__all__ = ["store_mesh_HDF5", "numpy_to_dolfin", "plot_faces",
           "plot_edges"]


comm = COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def store_mesh_HDF5(mesh, meshpath):
    '''
    Function that stores generated mesh in both "HDMF5"
    (.h5) format and in "XDMF" (.XMDF) format.
    '''
    meshpath_hdf5 = meshpath + ".h5"
    with df.HDF5File(mesh.mpi_comm(), meshpath_hdf5, "w") as hdf5:
        info("Storing the mesh in " + MESHES_DIR)
        hdf5.write(mesh, "mesh")
    meshpath_xdmf = meshpath + "_xdmf.xdmf"
    xdmff = df.XDMFFile(mesh.mpi_comm(), meshpath_xdmf)
    xdmff.write(mesh)
    info("Done.")


def add_vertical_boundary_vertices(l, x, y, n, dr):
    """A function to make a 'polygon'-straightline in order to make a
    periodic domain.  l is eges in the polygon, x is the x coordinat
    for the line, y is the lengt of the line, n is the number for
    verticies and dr
    """
    for i in range(1, n):
        if dr == -1:  # Values go from (near) y to 0 with n points
            val = y * float(n - i)/float(n)
        elif dr == 1:  # values go from (near) 0 to y with n points
            val = y * float(i)/float(n)
        l.append(df.Point(x, val))


def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]


def rad_points(x_c, rad, dx, theta_start=0., theta_stop=2*np.pi):
    if theta_stop > theta_start:
        arc_length = (theta_stop-theta_start)*rad
    else:
        arc_length = (-theta_stop+theta_start)*rad
    return [(rad * np.cos(theta) + x_c[0], rad * np.sin(theta) + x_c[1])
            for theta in np.linspace(theta_start, theta_stop,
                                     int(np.ceil(arc_length/dx)+1),
                                     endpoint=True)]


def line_points(x_left, x_right, dx):
    N = int(np.ceil(np.sqrt((x_right[0]-x_left[0])**2 +
                            (x_right[1]-x_left[1])**2)/dx))
    return zip(np.linspace(x_left[0], x_right[0], N+1,
                           endpoint=True).flatten(),
               np.linspace(x_left[1], x_right[1], N+1,
                           endpoint=True).flatten())


def make_polygon(corner_pts, dx, start=0):
    segs = zip(corner_pts[:], corner_pts[1:] + [corner_pts[0]])
    nodes = []
    for x, y in segs:
        nodes.extend(line_points(x, y, dx)[:-1])
    edges = round_trip_connect(start, start+len(nodes)-1)
    return nodes, edges


def numpy_to_dolfin_old(nodes, elements):
    """ Deprecated version of numpy_to_dolfin. To be removed? """
    tmpfile = "tmp.xml"

    if rank == 0:
        with open(tmpfile, "w") as f:
            f.write("""
<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
    <mesh celltype="triangle" dim="2">
        <vertices size="%d">""" % len(nodes))

            for i, pt in enumerate(nodes):
                f.write('<vertex index="%d" x="%g" y="%g"/>' % (
                    i, pt[0], pt[1]))

            f.write("""
        </vertices>
        <cells size="%d">""" % len(elements))

            for i, element in enumerate(elements):
                f.write('<triangle index="%d" v0="%d" v1="%d" v2="%d"/>' % (
                    i, element[0], element[1], element[2]))

            f.write("""
            </cells>
          </mesh>
        </dolfin>""")

    comm.Barrier()

    mesh = df.Mesh(tmpfile)

    comm.Barrier()

    if rank == 0 and os.path.exists(tmpfile):
        os.remove(tmpfile)
    return mesh


def numpy_to_dolfin(nodes, elements):
    """ Convert nodes and elements to a dolfin mesh object. """
    tmpfile = "tmp.h5"

    if rank == 0:
        with h5py.File(tmpfile, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_("triangle")
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    comm.Barrier()

    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), tmpfile, "r")
    h5f.read(mesh, "mesh", False)
    h5f.close()

    comm.Barrier()

    remove_safe(tmpfile)
    return mesh


def call_method(method, methods, scripts_folder, cmd_kwargs):
    # Call the specified method
    if method[-1] == "?" and method[:-1] in methods:
        m = __import__("{}.{}".format(scripts_folder,
                                      method[:-1])).__dict__[method[:-1]]
        m.description(**cmd_kwargs)
    elif method in methods:
        m = __import__("{}.{}".format(scripts_folder, method)).__dict__[method]
        m.method(**cmd_kwargs)
    else:
        info_on_red("The specified mesh generation method doesn't exist.")


def main():
    cmd_kwargs = parse_command_line()

    method = cmd_kwargs.get("mesh", "straight_capilar")

    scripts_folder = "mesh_scripts"

    methods = get_methods(scripts_folder)

    # Get help if it was called for.
    if cmd_kwargs.get("help", False):
        get_help(methods, scripts_folder, __file__)

    call_method(method, methods, scripts_folder, cmd_kwargs)


if __name__ == "__main__":
    main()
