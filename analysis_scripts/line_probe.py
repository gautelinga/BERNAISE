""" line_probe script """
from common import info, info_cyan, info_on_red, makedirs_safe
import numpy as np
from postprocess import get_steps, index2letter, rank
from utilities.plot import plot_probes
import os
from utilities.generate_mesh import line_points


def description(ts, **kwargs):
    info("Probe along a line.")


def method(ts, dx=0.1, line="[0.,0.]--[1.,1.]", time=None, dt=None,
           **kwargs):
    """ Probe along a line. """
    info_cyan("Probe along a line.")
    try:
        x_a, x_b = [tuple(eval(pt)) for pt in line.split("--")]
        assert(len(x_a) == ts.dim)
        assert(len(x_b) == ts.dim)
        assert(all([bool(isinstance(xd, float) or
                         isinstance(xd, int))
                    for xd in list(x_a)+list(x_b)]))
    except:
        info_on_red("Faulty line format. Use 'line=[x1,y1]--[x2,y2]'.")
        exit()

    x = np.array(line_points(x_a, x_b, dx))

    info("Probes {num} points from {a} to {b}".format(
        num=len(x), a=x_a, b=x_b))

    if rank == 0:
        plot_probes(ts.nodes, ts.elems, x,
                    colorbar=False, title="Probes")

    f = ts.functions()
    probes = dict()
    from fenicstools import Probes
    for field, func in f.iteritems():
        probes[field] = Probes(x.flatten(), func.function_space())

    steps = get_steps(ts, dt, time)

    for step in steps:
        info("Step " + str(step) + " of " + str(len(ts)))
        ts.update_all(f, step)
        for field, probe in probes.iteritems():
            probe(f[field])

    probe_arr = dict()
    for field, probe in probes.iteritems():
        probe_arr[field] = probe.array()

    if rank == 0:
        for i, step in enumerate(steps):
            chunks = [x]
            header_list = [index2letter(d) for d in range(ts.dim)]
            for field, chunk in probe_arr.iteritems():
                if chunk.ndim == 1:
                    header_list.append(field)
                    chunk = chunk[:].reshape(-1, 1)
                elif chunk.ndim == 2:
                    header_list.append(field)
                    chunk = chunk[:, i].reshape(-1, 1)
                elif chunk.ndim > 2:
                    header_list.extend(
                        [field + "_" + index2letter(d)
                         for d in range(ts.dim)])
                    chunk = chunk[:, :, i]
                chunks.append(chunk)

            data = np.hstack(chunks)
            header = "\t".join(header_list)
            makedirs_safe(os.path.join(ts.analysis_folder, "probes"))
            np.savetxt(os.path.join(ts.analysis_folder, "probes",
                                    "probes_{:06d}.dat".format(step)),
                       data, header=header)
