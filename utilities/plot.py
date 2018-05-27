import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.tri as mtri
from matplotlib.tri import tricontour
from matplotlib.tri import TriContourSet
from mpi4py.MPI import COMM_WORLD as comm
from common.io import remove_safe
from skimage import measure
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ENABLE_TEX = True
if ENABLE_TEX:  # Hacked for pretty output
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

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
        self.xlabel = tex_escape(xlabel)
        self.ylabel = tex_escape(ylabel)
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
            self.clabel = tex_escape(clabel)
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


def tex_escape(string):
    return "${}$".format(string)


def plot_edges(pts, edges, title=None, clabel=None,
               save=None, show=True):
    nppts = np.array(pts)
    npedges = np.array(edges)
    lc = LineCollection(nppts[npedges])

    fig = Figure(title=title, clabel=clabel, save=save, show=show)
    fig.ax.add_collection(lc)
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


def plot_streamlines(nodes, elems, vals, title=None, clabel=None,
                     save=None, show=True, num_intp=200, density=0.8):
    """ Plots streamlines with contour in the background.
    Values given at nodes. """
    fig = Figure(title=title, subplots=True, clabel=clabel,
                 save=save, show=show)

    vals_norm = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2) + 1e-10
    # vals_norm_max = np.max(vals_norm)
    fig.colorbar_ax = fig.ax.tricontourf(nodes[:, 0], nodes[:, 1], elems,
                                         vals_norm)

    Lx = nodes[:, 0].max()-nodes[:, 0].min()
    Ly = nodes[:, 1].max()-nodes[:, 1].min()
    dx = max(Lx, Ly)/num_intp
    Nx = int(Lx/dx)
    Ny = int(Ly/dx)

    x_i, y_i = np.meshgrid(
        np.linspace(dx+nodes[:, 0].min(),
                    nodes[:, 0].max()-dx, Nx),
        np.linspace(dx+nodes[:, 1].min(),
                    nodes[:, 1].max()-dx, Ny))
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
    ux_interp = mtri.LinearTriInterpolator(triang, vals[:, 0])
    uy_interp = mtri.LinearTriInterpolator(triang, vals[:, 1])
    ux_i = ux_interp(x_i, y_i)
    uy_i = uy_interp(x_i, y_i)

    ux_i = np.array(ux_i.filled(0.))
    uy_i = np.array(uy_i.filled(0.))

    u_norm = np.sqrt(ux_i**2 + uy_i**2)

    lw = np.zeros_like(ux_i)
    lw[:] += 5*u_norm/(u_norm.max() + 1e-10)

    mask = np.zeros(ux_i.shape, dtype=bool)
    ux_i_2 = np.ma.array(ux_i, mask=mask)

    fig.ax.streamplot(x_i, y_i,
                      ux_i_2, uy_i,
                      color="k",
                      density=density,
                      linewidth=lw)


def plot_fancy(nodes, elems, phi=None, charge=None, u=None, charge_max=None,
               show=False, save=None, num_intp=100, title=None, clabel=None,
               animation_mode=True):
    """ Plots fancily. """
    if animation_mode:
        fig = Figure(colorbar=False, tight_layout=True, show=show,
                     xlabel="", ylabel="", save=save, ticks=False)
    else:
        fig = Figure(colorbar=True, tight_layout=False, show=show,
                     xlabel=tex_escape("x"), ylabel=tex_escape("y"),
                     save=save, ticks=True)

    if phi is None:
        phi = -np.ones(len(nodes))

    if charge is None:
        charge = np.zeros(len(nodes))

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
        Lx = nodes[:, 0].max()-nodes[:, 0].min()
        Ly = nodes[:, 1].max()-nodes[:, 1].min()
        dx = max(Lx, Ly)/num_intp
        Nx = int(Lx/dx)
        Ny = int(Ly/dx)

        x_i, y_i = np.meshgrid(
            np.linspace(dx+nodes[:, 0].min(),
                        nodes[:, 0].max()-dx, Nx),
            np.linspace(dx+nodes[:, 1].min(),
                        nodes[:, 1].max()-dx, Ny))
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
        ux_interp = mtri.LinearTriInterpolator(triang, u[:, 0])
        uy_interp = mtri.LinearTriInterpolator(triang, u[:, 1])
        phi_interp = mtri.LinearTriInterpolator(triang, phi)
        ux_i = ux_interp(x_i, y_i)
        uy_i = uy_interp(x_i, y_i)
        phi_i = phi_interp(x_i, y_i)

        ux_i = np.array(ux_i.filled(0.))
        uy_i = np.array(uy_i.filled(0.))
        phi_i = np.array(phi_i.filled(0.))

        u_norm = np.sqrt(ux_i**2 + uy_i**2)

        lw = np.zeros_like(ux_i)
        lw[:] += 5*u_norm/(u_norm.max() + 1e-10)

        mask = np.zeros(ux_i.shape, dtype=bool)
        mask[phi_i > 0.] = True
        ux_i_2 = np.ma.array(ux_i, mask=mask)

        fig.ax.streamplot(x_i, y_i,
                          ux_i_2, uy_i,
                          color="k",
                          density=0.6,
                          linewidth=lw)

        mask = np.zeros(ux_i.shape, dtype=bool)
        mask[phi_i < 0.] = True
        ux_i_2 = np.ma.array(ux_i, mask=mask)

        fig.ax.streamplot(x_i, y_i,
                          ux_i_2, uy_i,
                          color="w",
                          density=0.6,
                          linewidth=lw)

    return fig


def plot_any_field(nodes, elems, values, save=None, show=True, label=None):
    """ Plot using quiver plot or contour plot depending on field. """
    if label is None:
        label = ""
    if values.shape[1] >= 2:
        plot_streamlines(nodes, elems, values[:, :2],
                         title=label, clabel=label, save=save,
                         show=show)
    else:
        plot_contour(nodes, elems, values[:, 0],
                     title=label, clabel=label, save=save,
                     show=show)


def zero_level_set(nodes, elems, vals, show=False, save_file=False, num_intp=32):
    """ Returns the zero level set of the phase field, i.e. the phase boundary.
    GL: Possibly not a plot function?
    """
    dim = elems.shape[1]-1
    if dim == 2:
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

    elif dim == 3:
        Lx = nodes[:, 0].max()-nodes[:, 0].min()
        Ly = nodes[:, 1].max()-nodes[:, 1].min()
        Lz = nodes[:, 2].max()-nodes[:, 2].min()
        dx = max([Lx, Ly, Lz])/num_intp
        Nx = int(Lx/dx)
        Ny = int(Ly/dx)
        Nz = int(Lz/dx)

        x_i, y_i, z_i = np.meshgrid(
            np.linspace(nodes[:, 0].min(),
                        nodes[:, 0].max(), Nx),
            np.linspace(nodes[:, 1].min(),
                        nodes[:, 1].max(), Ny),
            np.linspace(nodes[:, 2].min(),
                        nodes[:, 2].max(), Nz))

        phi_i = griddata(nodes, vals, (x_i, y_i, z_i), method="nearest")

        verts, faces, normals, values = measure.marching_cubes(phi_i, 0.)

        verts[:, 0] = ((nodes[:, 0].max() -
                        nodes[:, 0].min())*verts[:, 0]/Nx
                       + nodes[:, 0].min())
        verts[:, 1] = ((nodes[:, 1].max() -
                        nodes[:, 1].min())*verts[:, 1]/Ny
                       + nodes[:, 1].min())
        verts[:, 2] = ((nodes[:, 2].max() -
                        nodes[:, 2].min())*verts[:, 2]/Nz
                       + nodes[:, 2].min())

        if rank == 0 and show:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            mesh = Poly3DCollection(verts[faces])
            mesh.set_edgecolor('k')
            ax.add_collection3d(mesh)

            ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
            ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
            ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())

            plt.tight_layout()
            plt.show()

    if rank == 0 and save_file:
        remove_safe(save_file)
        with open(save_file, "ab+") as f:
            if dim == 2:
                for path in paths:
                    np.savetxt(f, path.vertices)
                    f.write("\n")
            elif dim == 3:
                np.savetxt(f, verts)

    if dim == 2:
        paths_out = []
        for path in paths:
            paths_out.append(path.vertices)
        return paths_out
    elif dim == 3:
        return verts[faces]
