from utilities.TimeSeries import TimeSeries
import dolfin as df
from common import info, parse_command_line, \
    info_cyan, info_split, info_on_red, info_red, info_yellow
import os
import glob
import numpy as np
from mpi4py import MPI
from utilities import get_methods, get_help


"""
BERNAISE: Post-processing tool.

This module is used to read and analyze simulated data.

"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_norms(err, vector_norms=["l2", "linf"],
                  function_norms=["L2", "H1"], show=True,
                  tablefmt="simple", save=False):
    """ Compute norms, output to terminal, etc. """
    info_split("Vector norms:", ", ".join(vector_norms))
    info_split("Function norms:", ", ".join(function_norms))

    headers = ["Fields"] + vector_norms + function_norms

    table = []
    for field in err.keys():
        row = [field]
        for norm_type in vector_norms:
            row.append(df.norm(err[field].vector(), norm_type=norm_type))
        for norm_type in function_norms:
            row.append(df.norm(err[field], norm_type=norm_type))
        table.append(row)

    from tabulate import tabulate
    tab_string = tabulate(table, headers, tablefmt=tablefmt, floatfmt="e")
    if show:
        info("\n" + tab_string + "\n")

    if save and rank == 0:
        info_split("Saving to file:", save)
        with open(save, "w") as outfile:
            outfile.write(tab_string)


def path_length(paths, total_length=True):
    lengths = []
    for x in paths:
        dim = x.shape[1]
        if dim == 2:
            dx = x[:-1, :]-x[1:, :]
            length = np.sum(np.sqrt(dx[:, 0]**2 + dx[:, 1]**2))
        if dim == 3:
            # FIXME: This is actually an area...
            # Heron's formula
            a = np.linalg.norm(x[0, :] - x[1, :])
            b = np.linalg.norm(x[1, :] - x[2, :])
            c = np.linalg.norm(x[2, :] - x[0, :])
            s = (a + b + c)/2.0
            length = np.sqrt(s*(s-a)*(s-b)*(s-c))
        lengths.append(length)
    if total_length:
        return sum(lengths)
    else:
        return lengths


def index2letter(index):
    return ("x", "y", "z")[index]


def get_steps(ts, dt=None, time=None):
    """ Get steps sampled at equidistant times. """
    if time is not None:
        step, _ = get_step_and_info(ts, time)
        steps = [step]
    elif dt is not None and dt > 0.:
        steps = []
        time_max = ts.times[-1]
        time = ts.times[0]
        while time <= time_max:
            step, _ = get_step_and_info(ts, time)
            steps.append(step)
            time += dt
    else:
        steps = range(len(ts))
    return steps


def get_step_and_info(ts, time, step=0):
    if time is not None:
        step, time = ts.get_nearest_step_and_time(time)
    else:
        time = ts.get_time(step)
    info("Time = {}, timestep = {}.".format(time, step))
    return step, time


def call_method(method, methods, scripts_folder, ts, cmd_kwargs):
    # Call the specified method
    if method[-1] == "?" and method[:-1] in methods:
        m = __import__("{}.{}".format(scripts_folder,
                                      method[:-1])).__dict__[method[:-1]]
        m.description(ts, **cmd_kwargs)
    elif method in methods:
        m = __import__("{}.{}".format(scripts_folder, method)).__dict__[method]
        m.method(ts, **cmd_kwargs)
    else:
        info_on_red("The specified analysis method doesn't exist.")


def main():
    info_yellow("BERNAISE: Post-processing tool")
    cmd_kwargs = parse_command_line()

    folder = cmd_kwargs.get("folder", False)
    scripts_folder = "analysis_scripts"
    methods = get_methods(scripts_folder)

    # Get help if it was called for.
    if cmd_kwargs.get("help", False):
        get_help(methods, scripts_folder, __file__, skip=1)

    # Get sought fields
    sought_fields = cmd_kwargs.get("fields", False)
    if not sought_fields:
        sought_fields = None
    elif not isinstance(sought_fields, list):
        sought_fields = [sought_fields]

    if not folder:
        info("No folder(=[...]) specified.")
        exit()

    sought_fields_str = (", ".join(sought_fields)
                         if sought_fields is not None else "All")
    info_split("Sought fields:", sought_fields_str)
    ts = TimeSeries(folder, sought_fields=sought_fields)
    info_split("Found fields:", ", ".join(ts.fields))
    method = cmd_kwargs.get("method", "geometry_in_time")

    if len(ts.fields) == 0:
        info_on_red("Found no data.")
        exit()

    call_method(method, methods, scripts_folder, ts, cmd_kwargs)


if __name__ == "__main__":
    main()
