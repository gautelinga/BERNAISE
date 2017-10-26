""" plot_dolfin script """
from common import info_cyan
from postprocess import get_step_and_info
import dolfin as df


def description(ts, **kwargs):
    info("Plot at given time/step using Dolfin.")


def method(ts, time=None, step=0, **kwargs):
    """ Plot at given time/step using dolfin. """
    info_cyan("Plotting at given timestep using Dolfin.")
    step, time = get_step_and_info(ts, time)
    f = ts.functions()
    for field in ts.fields:
        ts.update(f[field], field, step)
        df.plot(f[field])
    df.interactive()
