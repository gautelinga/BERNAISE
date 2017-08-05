import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.tri import tricontour
from matplotlib.tri import TriContourSet
from mpi4py.MPI import COMM_WORLD as comm
from common.io import remove_safe

rank = comm.Get_rank()
size = comm.Get_size()


def plot_edges(pts, edges):
    nppts = np.array(pts)
    npedges = np.array(edges)
    lc = LineCollection(nppts[npedges])

    fig = plt.figure()
    plt.gca().add_collection(lc)
    plt.xlim(nppts[:, 0].min(), nppts[:, 0].max())
    plt.ylim(nppts[:, 1].min(), nppts[:, 1].max())
    plt.plot(nppts[:, 0], nppts[:, 1], 'ro')
    plt.show()


def plot_faces(coords, faces):
    fig = plt.figure()
    colors = np.arange(len(faces))
    plt.tripcolor(coords[:, 0], coords[:, 1], faces,
                  facecolors=colors, edgecolors='k')
    plt.gca().set_aspect('equal')
    plt.show()


def plot_contour(nodes, elems, vals):
    fig = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(nodes[:, 0], nodes[:, 1], elems, vals)
    plt.colorbar()
    plt.show()


def plot_quiver(nodes, elems, vals):
    """ Plots quivers """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    vals_norm = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2) + 1e-10
    # vals_norm_max = np.max(vals_norm)
    ax.tricontourf(nodes[:, 0], nodes[:, 1], elems,
                   vals_norm)
    ax.quiver(nodes[:, 0], nodes[:, 1],
              vals[:, 0]/vals_norm,
              vals[:, 1]/vals_norm)
    plt.show()


def zero_level_set(nodes, elems, vals, show=False, save_file=False):
    """ Returns the zero level set of the phase field, i.e. the phase boundary.
    GL: Possibly not a plot function?
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cs = ax.tricontour(nodes[:, 0], nodes[:, 1], elems, vals, levels=[0.])
    paths = cs.collections[0].get_paths()
    x, y = zip(*paths[0].vertices)
    plt.close(fig)

    if rank == 0 and show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.plot(x, y, '*-')
        plt.show()

    if rank == 0 and save_file:
        remove_safe(save_file)
        with open(save_file, "ab+") as f:
            for path in paths:
                np.savetxt(f, path.vertices)
                f.write("\n")

    paths_out = []
    for path in paths:
        paths_out.append(path.vertices)
    return paths_out
