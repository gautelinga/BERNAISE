""" energy_in_time script """
from common import info, info_cyan
from postprocess import get_steps, rank
import numpy as np
import dolfin as df
import os


def description(ts, **kwargs):
    info("Plot mean field values in time.")


def method(ts, dt=0, **kwargs):
    """ Plot mean field values in time. """
    info_cyan("Plot mean field values in time.")

    params = ts.get_parameters()
    steps = get_steps(ts, dt)

    problem = params["problem"]
    info("Problem: {}".format(problem))

    t = np.zeros(len(steps))

    x_ = ts.functions()

    fields = dict()
    for field, f in x_.items():
        if field == "u":
            fields["u_x"] = f[0]
            fields["u_y"] = f[1]
        else:
            fields[field] = f

    data = dict()

    for field in fields:
        data[field] = np.zeros(len(steps))

    for i, step in enumerate(steps):
        info("Step {} of {}".format(step, len(ts)))

        for field in x_.keys():
            ts.update(x_[field], field, step)

        for field, f in fields.items():
            data[field][i] = df.assemble(f*df.dx)

        t[i] = ts.times[step]

    field_keys = sorted(fields.keys())

    savedata = np.array(
        zip(steps, t, *[data[field] for field in field_keys]))

    if rank == 0:
        header = "Step\tTime\t"+"\t".join(field_keys)
        filename = os.path.join(ts.analysis_folder,
                                "value_in_time.dat")
        np.savetxt(filename, savedata, header=header)
