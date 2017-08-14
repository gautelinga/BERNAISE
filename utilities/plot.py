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


__all__ = ["plot_edges", "plot_faces", "plot_contour", "plot_quiver",
           "zero_level_set", "plot_fancy", "plot_any_field"]


class Figure:
    def __init__(self, title=None, show=True, aspect_equal=True,
                 save=None, base_fig=None, xlabel="x", ylabel="y",
                 colorbar=True, clabel=None, subplots=False,
                 tight_layout=False, ticks=True):
        self.title = title
        self.show = show
        self.aspect_equal = aspect_equal
        self.save = save
        self.base_fig = base_fig
        self.colorbar = colorbar
        self.subplots = subplots
        self.colorbar_ax = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tight_layout = tight_layout
        self.ticks = ticks

        if self.base_fig is not None:
            self.fig = self.base_fig
        else:
            self.fig, self.ax = plt.subplots()

        if self.aspect_equal:
            plt.gca().set_aspect('equal')

        if self.title is not None:
            self.fig.canvas.set_window_title(self.title)

        if isinstance(self.xlabel, str):
            plt.xlabel(self.xlabel)

        if isinstance(self.ylabel, str):
            plt.ylabel(self.ylabel)

        if isinstance(clabel, str):
            self.clabel = clabel
        else:
            self.clabel = ""

        if not self.ticks:
            self.ax.set_axis_off()

    def __del__(self):
        """ This is called upon the last reference to this object. """
        if self.colorbar and self.colorbar_ax is not None:
            self.fig.colorbar(self.colorbar_ax, label=self.clabel)
        elif self.colorbar:
            plt.colorbar(label=self.clabel)

        if self.tight_layout:
            plt.tight_layout()

        if self.save is not None:
            plt.savefig(self.save, bbox_inches="tight", pad_inches=0.)

        if self.show:
            plt.show()
        else:
            plt.close()


def plot_edges(pts, edges, title=None, clabel=None,
               save=None, show=True):
    nppts = np.array(pts)
    npedges = np.array(edges)
    lc = LineCollection(nppts[npedges])

    fig = Figure(title=title, clabel=clabel, save=save, show=show)
    plt.xlim(nppts[:, 0].min(), nppts[:, 0].max())
    plt.ylim(nppts[:, 1].min(), nppts[:, 1].max())
    plt.plot(nppts[:, 0], nppts[:, 1], 'ro')
    return fig


def plot_faces(coords, faces, face_values=None, title=None,
               clabel=None, colorbar=True, save=None, show=True):
    """ Plot a mesh with values given at faces. """
    if face_values is None:
        colors = np.arange(len(faces))
        clabel = "Face number"
    else:
        colors = face_values
    fig = Figure(title=title, clabel=clabel, colorbar=colorbar,
                 save=save, show=show)
    plt.tripcolor(coords[:, 0], coords[:, 1], faces,
                  facecolors=colors, edgecolors='k')
    return fig


def plot_contour(nodes, elems, vals,
                 title=None, clabel=None,
                 save=None, show=True):
    """ Contour plot; values given at nodes. """
    fig = Figure(title=title, clabel=clabel, save=save, show=show)
    plt.tricontourf(nodes[:, 0], nodes[:, 1], elems, vals)
    return fig


def plot_probes(nodes, elems, probes, title=None, clabel=None,
                colorbar=False, save=None, show=True):
    fig = plot_faces(nodes, elems, face_values=np.ones(len(elems)),
                     colorbar=colorbar, title=title,
                     save=save, show=show)
    plt.plot(probes[:, 0], probes[:, 1], "ro-")
    return fig


def plot_quiver(nodes, elems, vals, title=None, clabel=None,
                save=None, show=True):
    """ Plots quivers with contour in the background.
    Values given at nodes. """
    fig = Figure(title=title, subplots=True, clabel=clabel,
                 save=save, show=show)

    vals_norm = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2) + 1e-10
    # vals_norm_max = np.max(vals_norm)
    fig.colorbar_ax = fig.ax.tricontourf(nodes[:, 0], nodes[:, 1], elems,
                                         vals_norm)
    fig.ax.quiver(nodes[:, 0], nodes[:, 1],
                  vals[:, 0]/vals_norm, vals[:, 1]/vals_norm)


def plot_fancy(nodes, elems, phi, charge, u=None, charge_max=None,
               show=False, save=None):
    """ Plots fancily. """
    fig = Figure(colorbar=False, tight_layout=True, show=show,
                 xlabel="", ylabel="", save=save, ticks=False)

    if charge_max is None:
        charge_max = max(np.max(np.abs(charge)), 1e-10)

    cmap = plt.cm.get_cmap('Greys')
    cmap._init()
    cmap._lut[:, :] = 0.
    length = len(cmap._lut[:, -1])
    # cmap._lut[:, -1] = np.linspace(0., 1.0, length)
    cmap._lut[:length/2, -1] = 0.
    cmap._lut[length/2:, -1] = 1.

    phi[phi > 1.] = 1.
    phi[phi < -1.] = -1.

    plt.tripcolor(nodes[:, 0], nodes[:, 1], elems, charge,
                  cmap=plt.get_cmap("coolwarm"), shading="gouraud",
                  vmin=-charge_max, vmax=charge_max)
    plt.tricontourf(nodes[:, 0], nodes[:, 1], elems, phi,
                    cmap=cmap, levels=[-2.0, 0., 2.0], antialiased=True)

    if u is not None:
        u_norm = np.sqrt(u[:, 0]**2 + u[:, 1]**2) + 1e-10
        colors = phi
        norm = plt.Normalize()
        norm.autoscale(colors)
        colormap = cmap  # plt.cm.get_cmap('inferno')
        cmap._lut[:, -1] = 0.5
        cmap._lut[length/2:, :-1] = 1.

        fig.ax.quiver(nodes[:, 0], nodes[:, 1],
                      u[:, 0]/u_norm, u[:, 1]/u_norm,
                      color=colormap(norm(colors)))

    return fig


def plot_any_field(nodes, elems, values, save=None, show=True, label=None):
    """ Plot using quiver plot or contour plot depending on field. """
    if label is None:
        label = ""
    if values.shape[1] >= 2:
        plot_quiver(nodes, elems, values[:, :2],
                    title=label, clabel=label, save=save,
                    show=show)
    else:
        plot_contour(nodes, elems, values[:, 0],
                     title=label, clabel=label, save=save,
                     show=show)


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
