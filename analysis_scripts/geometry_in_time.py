import dolfin as df
from common import info, makedirs_safe, info_cyan
import os
import numpy as np
from utilities.plot import zero_level_set
from postprocess import get_steps, path_length, rank


def description(ts, **kwargs):
    info("Analyze geometry in time.")


def method(ts, dt=0, **kwargs):
    """ Analyze geometry in time."""

    info_cyan("Analyzing the evolution of the geometry through time.")

    if not ts.get_parameter("enable_PF"):
        print "Phase field not enabled."
        return False

    f_mask = df.Function(ts.function_space)
    f_mask_x = []
    f_mask_u = []
    for d in range(ts.dim):
        f_mask_x.append(df.Function(ts.function_space))
        f_mask_u.append(df.Function(ts.function_space))

    length = np.zeros(len(ts))
    area = np.zeros(len(ts))
    com = np.zeros((len(ts), ts.dim))
    u = np.zeros((len(ts), ts.dim))

    makedirs_safe(os.path.join(ts.analysis_folder, "contour"))

    steps = get_steps(ts, dt)

    for step in steps:
        info("Step " + str(step) + " of " + str(len(ts)))

        phi = ts["phi", step][:, 0]
        mask = 0.5*(1.-phi)  # 0.5*(1.-np.sign(phi))
        ts.set_val(f_mask, mask)
        for d in range(ts.dim):
            ts.set_val(f_mask_x[d], mask*ts.nodes[:, d])
            ts.set_val(f_mask_u[d], mask*ts["u", step][:, d])

        contour_file = os.path.join(ts.analysis_folder, "contour",
                                    "contour_{:06d}.dat".format(step))
        paths = zero_level_set(ts.nodes, ts.elems, phi,
                               save_file=contour_file)

        length[step] = path_length(paths)

        area[step] = df.assemble(f_mask*df.dx)
        for d in range(ts.dim):
            com[step, d] = df.assemble(f_mask_x[d]*df.dx)
            u[step, d] = df.assemble(f_mask_u[d]*df.dx)

    for d in range(ts.dim):
        com[:, d] /= area
        u[:, d] /= area

    if rank == 0:
        np.savetxt(os.path.join(ts.analysis_folder,
                                "time_data.dat"),
                   np.array(zip(np.arange(len(ts)), ts.times, length, area,
                                com[:, 0], com[:, 1], u[:, 0], u[:, 1])),
                   header=("Timestep\tTime\tLength\tArea\t"
                           "CoM_x\tCoM_y\tU_x\tU_y"))
