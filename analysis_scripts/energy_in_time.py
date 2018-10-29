""" energy_in_time script """
from common import info, info_cyan
from postprocess import get_steps, rank
import numpy as np
import dolfin as df
import os
import importlib


def description(ts, **kwargs):
    info("Plot energy in time.")


def method(ts, dt=0, **kwargs):
    """ Plot energy in time. """
    info_cyan("Plot energy in time.")

    params = ts.get_parameters()
    steps = get_steps(ts, dt)

    problem = params["problem"]
    info("Problem: {}".format(problem))

    solver = params["solver"]
    info("Solver:  {}".format(solver))

    solver_module = importlib.import_module("solvers.{}".format(solver))
    discrete_energy = solver_module.discrete_energy

    t = np.zeros(len(steps))

    x_ = ts.functions()

    F_keys = discrete_energy(None, **params)
    F = []
    for i in range(len(F_keys)):
        F.append(np.zeros(len(steps)))

    for i, step in enumerate(steps):
        info("Step {} of {}".format(step, len(ts)))

        for field in x_:
            ts.update(x_[field], field, step)

        fs = discrete_energy(x_, **params)
        for j in range(len(F_keys)):
            F[j][i] = df.assemble(fs[j]*df.dx)

        t[i] = ts.times[step]

    data = np.array(list(zip(steps, t, *F)))

    if rank == 0:
        with open(os.path.join(ts.analysis_folder,
                               "energy_in_time.dat"),
                  "w") as outfile:
            np.savetxt(outfile, data, header="Step\tTime\t"+"\t".join(F_keys))
