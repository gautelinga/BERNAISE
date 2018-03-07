""" analytic_reference script """
from common import info, info_split, info_cyan
from postprocess import get_step_and_info, rank, compute_norms
import dolfin as df
import os
from utilities.plot import plot_any_field


def description(ts, **kwargs):
    info("""Compare to analytic reference expression at given timestep.
This is done by importing the function "reference" in the problem module.""")


def method(ts, time=None, step=0, show=False,
           save_fig=False, **kwargs):
    """ Compare to analytic reference expression at given timestep.
    This is done by importing the function "reference" in the problem module.
    """
    info_cyan("Comparing to analytic reference at given time or step.")
    step, time = get_step_and_info(ts, time, step)
    parameters = ts.get_parameters(time=time)
    problem = parameters.get("problem", "intrusion_bulk")
    exec("from problems.{} import reference".format(problem))
    ref_exprs = reference(t=time, **parameters)

    info("Comparing to analytic solution.")
    info_split("Problem:", "{}".format(problem))
    info_split("Time:", "{}".format(time))

    f = ts.functions(ref_exprs.keys())

    err = dict()
    f_int = dict()
    f_ref = dict()
    for field in ref_exprs.keys():
        el = f[field].function_space().ufl_element()
        degree = el.degree()
        if bool(el.value_size() != 1):
            W = df.VectorFunctionSpace(ts.mesh, "CG", degree+3)
        else:
            W = df.FunctionSpace(ts.mesh, "CG", degree+3)
        err[field] = df.Function(W)
        f_int[field] = df.Function(W)
        f_ref[field] = df.Function(W)

    for field, ref_expr in ref_exprs.iteritems():
        ref_expr.t = time
        # Update numerical solution f
        ts.update(f[field], field, step)

        # Interpolate f to higher space
        f_int[field].assign(df.interpolate(
            f[field], f_int[field].function_space()))

        # Interpolate f_ref to higher space
        f_ref[field].assign(df.interpolate(
            ref_expr, f_ref[field].function_space()))

        err[field].vector()[:] = (f_int[field].vector().get_local() -
                                  f_ref[field].vector().get_local())

        if show or save_fig:
            # Interpolate the error to low order space for visualisation.
            err_int = df.interpolate(err[field], f[field].function_space())
            err_arr = ts.nodal_values(err_int)
            label = "Error in " + field

            if rank == 0:
                save_fig_file = None
                if save_fig:
                    save_fig_file = os.path.join(
                        ts.plots_folder, "error_{}_time{}_analytic.png".format(
                            field, time))

                plot_any_field(ts.nodes, ts.elems, err_arr,
                               save=save_fig_file, show=show, label=label)

    save_file = os.path.join(ts.analysis_folder,
                             "errornorms_time{}_analytic.dat".format(
                                 time))
    compute_norms(err, save=save_file)
