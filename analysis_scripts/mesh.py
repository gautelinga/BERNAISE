""" mesh script """
from common import info, info_cyan
import dolfin as df
from postprocess import rank
from utilities.plot import plot_faces, plot_any_field
import os
import numpy as np


def description(ts, **kwargs):
    info("Mesh info and plot.")


def method(ts, show=True, save_fig=False, latex=False, **kwargs):
    """ Mesh info and plot. """
    info_cyan("Mesh info and plot.")
    f = df.Function(ts.function_space)
    f.vector()[:] = 1.
    area = df.assemble(f*df.dx)

    info("Number of nodes:    {}".format(len(ts.nodes)))
    info("Number of elements: {}".format(len(ts.elems)))
    info("Total mesh area:    {}".format(area))
    info("Mean element area:  {}".format(area/len(ts.elems)))

    if rank == 0:
        save_fig_file = None
        if save_fig:
            save_fig_file = os.path.join(ts.plots_folder,
                                         "mesh.png")

        plot_faces(ts.nodes, ts.elems, title="Mesh",
                   save=save_fig_file, show=show, latex=latex)
