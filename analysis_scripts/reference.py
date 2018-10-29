""" reference script """
from common import info, info_split, info_on_red, info_cyan
from utilities.TimeSeries import TimeSeries
import os
from postprocess import rank, compute_norms
import dolfin as df
from utilities.plot import plot_any_field


def description(ts, **kwargs):
    info("""Compare to numerical reference at given timestep.

The reference solution is assumed to be on a finer mesh, so the reference
solution is interpolated to the coarser mesh, where the comparison is made.""")


def method(ts, ref=None, time=1., show=False, save_fig=False, **kwargs):
    """Compare to numerical reference at given timestep.

    The reference solution is assumed to be on a finer mesh, so the
    reference solution is interpolated to the coarser mesh, where the
    comparison is made.
    """
    info_cyan("Comparing to numerical reference.")
    if not isinstance(ref, str):
        info_on_red("No reference specified. Use ref=(path).")
        exit()

    ts_ref = TimeSeries(ref, sought_fields=ts.fields)
    info_split("Reference fields:", ", ".join(ts_ref.fields))

    # Compute a 'reference ID' for storage purposes
    ref_id = os.path.relpath(ts_ref.folder,
                             os.path.join(ts.folder, "../")).replace(
                                 "../", "-").replace("/", "+")

    step, time_0 = ts.get_nearest_step_and_time(time)

    step_ref, time_ref = ts_ref.get_nearest_step_and_time(
        time, dataset_str="reference")

    info("Dataset:   Time = {}, timestep = {}.".format(time_0, step))
    info("Reference: Time = {}, timestep = {}.".format(time_ref, step_ref))

    # from fenicstools import interpolate_nonmatching_mesh

    f = ts.functions()
    f_ref = ts_ref.functions()
    err = ts_ref.functions()

    ts.update_all(f, step=step)
    ts_ref.update_all(f_ref, step=step_ref)

    for field in ts_ref.fields:
        # Interpolate solution to the reference mesh.
        f_int = df.interpolate(
            f[field], err[field].function_space())

        err[field].vector().set_local(
            f_int.vector().get_local() - f_ref[field].vector().get_local())

        if show or save_fig:
            err_arr = ts_ref.nodal_values(err[field])
            label = "Error in " + field

            if rank == 0:
                save_fig_file = None
                if save_fig:
                    save_fig_file = os.path.join(
                        ts.plots_folder, "error_{}_time{}_ref{}.png".format(
                            field, time, ref_id))

                plot_any_field(ts_ref.nodes, ts_ref.elems, err_arr,
                               save=save_fig_file, show=show, label=label)

    save_file = os.path.join(ts.analysis_folder,
                             "errornorms_time{}_ref{}.dat".format(
                                 time, ref_id))

    compute_norms(err, save=save_file)
