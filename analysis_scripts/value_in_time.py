""" value_in_time script

"""
from common import info, info_cyan, info_blue
from postprocess import get_steps, rank
import numpy as np
import dolfin as df
import os
from analysis_scripts.flux_in_time import fetch_boundaries


def description(ts, **kwargs):
    info("Plot field value in time at a boundaries.")


def method(ts, dt=0, extra_boundaries="", **kwargs):
    """ Plot value in time. """
    info_cyan("Plot value at boundary in time.")

    params = ts.get_parameters()
    steps = get_steps(ts, dt)

    problem = params["problem"]
    info("Problem: {}".format(problem))

    boundary_to_mark, ds = fetch_boundaries(
        ts, problem, params, extra_boundaries)
    
    x_ = ts.functions()

    fields = dict()
    for field, f in x_.items():
        if field == "u":
            fields["u_x"] = f[0]
            fields["u_y"] = f[1]
        else:
            fields[field] = f
    
    t = np.zeros(len(steps))
    data = dict()
    for boundary_name in boundary_to_mark:
        data[boundary_name] = dict()
        for field in fields:
            data[boundary_name][field] = np.zeros(len(steps))

    for i, step in enumerate(steps):
        info("Step {} of {}".format(step, len(ts)))

        for field in x_:
            ts.update(x_[field], field, step)

        for boundary_name, (mark, k) in boundary_to_mark.iteritems():
            for field, f in fields.items():
                data[boundary_name][field][i] = df.assemble(
                    f*ds[k](mark))

        t[i] = ts.times[step]

    savedata = dict()
    field_keys = sorted(fields.keys())
    for boundary_name in boundary_to_mark:
        savedata[boundary_name] = np.array(
            zip(steps, t, *[data[boundary_name][field]
                            for field in field_keys]))

    if rank == 0:
        header = "Step\tTime\t"+"\t".join(field_keys)
        for boundary_name in boundary_to_mark:
            filename = os.path.join(ts.analysis_folder,
                                    "value_in_time_{}.dat".format(boundary_name))
            np.savetxt(filename, savedata[boundary_name], header=header)
