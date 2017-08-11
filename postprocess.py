import dolfin as df
import h5py
from common import load_parameters, info, parse_command_line, makedirs_safe, \
    info_blue, info_cyan, info_split, info_on_red, info_red, info_yellow, \
    parse_xdmf, info_warning
import os
import glob
import numpy as np
from utilities.generate_mesh import numpy_to_dolfin
from mpi4py import MPI
from utilities.plot import plot_contour, plot_edges, plot_quiver, plot_faces,\
    zero_level_set, plot_probes, plot_fancy, plot_any_field
from utilities.generate_mesh import line_points

"""
BERNAISE: Post-processing tool.

This module is used to read, and perform analysis on, simulated data.

"""

__methods__ = ["geometry_in_time", "reference", "plot",
               "plot_dolfin", "line_probe", "make_gif",
               "mesh", "analytic_reference"]
__all__ = ["get_middle", "prep", "path_length",
           "index2letter"] + __methods__

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_middle(string, prefix, suffix):
    return string.split(prefix)[1].split(suffix)[0]


def prep(x_list):
    """ Prepare a list representing coordinates to be used as key in a
    dict. """
    # M = 10000
    # x_np = np.ceil(np.array(x_list)*M).astype(float)/M
    # return tuple(x_np.tolist())
    return tuple(x_list)


def path_length(paths, total_length=True):
    lengths = []
    for x in paths:
        dx = x[:-1, :]-x[1:, :]
        length = np.sum(np.sqrt(dx[:, 0]**2 + dx[:, 1]**2))
        lengths.append(length)
    if total_length:
        return sum(lengths)
    else:
        return lengths


def index2letter(index):
    return ("x", "y", "z")[index]


class TimeSeries:
    """ Class for loading timeseries """
    def __init__(self, folder, sought_fields=None, get_mesh_from=False):
        self.folder = folder

        self.settings_folder = os.path.join(folder, "Settings")
        self.timeseries_folder = os.path.join(folder, "Timeseries")
        self.statistics_folder = os.path.join(folder, "Statistics")
        self.analysis_folder = os.path.join(folder, "Analysis")
        self.plots_folder = os.path.join(folder, "Plots")
        self.tmp_folder = os.path.join(folder, ".tmp")

        self.params_prefix = os.path.join(self.settings_folder,
                                          "parameters_from_tstep_")
        self.params_suffix = ".dat"

        self.parameters = dict()

        self.nodes = None
        self.elems = None

        self.times = dict()
        self.datasets = dict()

        self._load_timeseries(sought_fields)

        if len(self.fields) > 0:
            self._load_mesh(get_mesh_from)

            self.dummy_function = df.Function(self.function_space)

            makedirs_safe(self.analysis_folder)
            makedirs_safe(self.plots_folder)
            makedirs_safe(self.tmp_folder)

    def _load_mesh(self, get_mesh_from):
        if not get_mesh_from:
            self.mesh = numpy_to_dolfin(self.nodes, self.elems)
            self.function_space = df.FunctionSpace(self.mesh, "CG", 1)
            self.vector_function_space = df.VectorFunctionSpace(
                self.mesh, "CG", 1)
            self.dim = self.function_space.mesh().topology().dim()

            self.x = self._make_dof_coords()
            self.xdict = self._make_xdict()

            indices_function = df.Function(self.function_space)
            self.set_val(indices_function, np.arange(len(self.nodes)))
            self.indices = np.asarray(indices_function.vector().array(),
                                      dtype=int)
        else:
            self.mesh = get_mesh_from.mesh
            self.function_space = get_mesh_from.function_space
            self.vector_function_space = get_mesh_from.vector_function_space
            self.dim = get_mesh_from.dim
            self.x = get_mesh_from.x
            self.xdict = get_mesh_from.xdict
            self.indices = get_mesh_from.indices

    def _load_timeseries(self, sought_fields=None):
        if bool(os.path.exists(self.settings_folder) and
                os.path.exists(self.timeseries_folder)):
            info_split("Opening folder:", self.folder)
        else:
            info_on_red("Folder does not contain "
                        "Settings or Timeseries folders.")
            exit()

        data = dict()
        for params_file in glob.glob(
                self.params_prefix + "*" + self.params_suffix):
            parameters = dict()
            from_tstep = int(get_middle(params_file,
                                        self.params_prefix,
                                        self.params_suffix))

            load_parameters(parameters, params_file)

            t_0 = float(parameters["t_0"])

            self.parameters[t_0] = parameters

            from_tstep_suffix = "_from_tstep_" + str(from_tstep) + ".h5"
            from_tstep_xml_suffix = "_from_tstep_" + str(from_tstep) + ".xdmf"
            for xml_file in glob.glob(os.path.join(
                    self.timeseries_folder, "*" + from_tstep_xml_suffix)):

                data_file = xml_file[:-4] + "h5"
                field = get_middle(data_file,
                                   self.timeseries_folder + "/",
                                   from_tstep_suffix)

                if bool(sought_fields is None or
                        field in sought_fields):
                    if bool(field not in data):
                        data[field] = dict()

                    dsets = parse_xdmf(xml_file)

                    with h5py.File(data_file, "r") as h5f:
                        if self.nodes is None or self.elems is None:
                            self.elems = np.array(h5f["Mesh/0/mesh/topology"])
                            self.nodes = np.array(h5f["Mesh/0/mesh/geometry"])
                        for time, dset_address in dsets:
                            data[field][time] = np.array(h5f[dset_address])

        for i, key in enumerate(data.keys()):
            tmps = sorted(data[key].items())
            if i == 0:
                self.times = [tmp[0] for tmp in tmps]
            self.datasets[key] = [tmp[1] for tmp in tmps]
        self.parameters = sorted(self.parameters.items())
        self.fields = self.datasets.keys()

    def _make_dof_coords(self):
        dofmap = self.function_space.dofmap()
        my_first, my_last = dofmap.ownership_range()
        x = self.function_space.tabulate_dof_coordinates().reshape(
            (-1, self.dim))
        unowned = dofmap.local_to_global_unowned()
        dofs = filter(lambda dof: dofmap.local_to_global_index(dof)
                      not in unowned,
                      xrange(my_last-my_first))
        x = x[dofs]
        return x

    def _make_xdict(self):
        if rank == 0:
            xdict = dict([(prep(list(x_list)), i) for i, x_list in
                          enumerate(self.nodes.tolist())])
        else:
            xdict = None
        xdict = comm.bcast(xdict, root=0)
        return xdict

    def set_val(self, f, f_data):
        vec = f.vector()
        values = vec.get_local()
        values[:] = [f_data[self.xdict[prep(x_val)]]
                     for x_val in self.x.tolist()]
        vec.set_local(values)
        vec.apply('insert')

    def update(self, f, field, step):
        """ Set dolfin vector f with values from field. """
        if field == "u":
            u_data = self.datasets["u"][step][:, :self.dim]
            for i in range(self.dim):
                self.set_val(self.dummy_function, u_data[:, i])
                df.assign(f.sub(i), self.dummy_function)
        else:
            f_data = self.datasets[field][step][:]
            self.set_val(f, f_data)

    def update_all(self, f, step):
        """ Set dict f of dolfin functions with values from all fields. """
        for field in self.fields:
            self.update(f[field], field, step)

    def get_parameter(self, key, time=0., default=False):
        return self.get_parameters(time).get(key, default)

    def get_parameters(self, time=0.):
        if len(self.parameters) == 1:
            return self.parameters[0][1]
        if time <= self.parameters[0][0]:
            return self.parameters[0][1]
        for i in range(len(self.parameters)-1):
            time_a = self.parameters[i][0]
            time_b = self.parameters[i+1][0]
            if time_a < time <= time_b:
                return self.parameters[i][1]
        return self.parameters[-1][1]

    def __getitem__(self, key):
        if len(key) == 1:
            return self.datasets[key]
        if len(key) == 2:
            return self.datasets[key[0]][key[1]]

    def __setitem__(self, key, val):
        self.datasets[key] = val

    def function(self, field):
        if field == "u":
            return df.Function(self.vector_function_space, name="u")
        else:
            return df.Function(self.function_space, name=field)

    def functions(self, fields=None):
        """ Returns dolfin functions for all fields. """
        f = dict()
        if fields is None:
            fields = self.fields
        for field in fields:
            f[field] = self.function(field)
        return f

    def __len__(self):
        return len(self.times)

    def get_time(self, step):
        return self.times[step]

    def get_nearest_step(self, time):
        if time < self.times[0]:
            return 0
        for step in range(len(self)-1):
            if time < self.times[step+1]:
                if self.times[step+1]-time > time-self.times[step]:
                    return step
                else:
                    return step+1
        return len(self)-1

    def get_nearest_step_and_time(self, time, dataset_str="dataset"):
        step = self.get_nearest_step(time)
        time_0 = self.get_time(step)
        if abs(time-time_0) > df.DOLFIN_EPS:
            info_warning("Could not find {} "
                         "at time={}. Using time={} instead.".format(
                             dataset_str, time, time_0))
        return step, time_0

    def _operate(self, function, field):
        return function([function(self[field, step], 0)
                         for step in range(len(self))])

    def max(self, field):
        return self._operate(np.max, field)

    def min(self, field):
        return self._operate(np.min, field)

    def mean(self, field):
        return self._operate(np.mean, field)

    def add_field(self, field, datasets):
        self[field] = datasets
        self.fields = self.datasets.keys()

    def compute_charge(self):
        """ Computing charge datasets by summing over all species. """
        solutes = self.get_parameter("solutes")
        charge_datasets = []
        for step in range(len(self)):
            charge_loc = np.zeros_like(self["phi", step])
            for solute in solutes:
                field = solute[0]
                z = solute[1]
                charge_loc[:, :] += z*self[field, step]
            charge_datasets.append(charge_loc)
        self.add_field("charge", charge_datasets)

    def nodal_values(self, f):
        """ Convert dolfin function to nodal values. """
        farray = f.vector().array()
        fdim = len(farray)/len(self.indices)
        farray = farray.reshape((len(self.indices), fdim))

        arr = np.zeros((len(self.nodes), fdim))
        arr_loc = np.zeros_like(arr)
        for i, fval in zip(self.indices, farray):
            arr_loc[i, :] = fval
        comm.Allreduce(arr_loc, arr, op=MPI.SUM)

        return arr


def geometry_in_time(ts, **kwargs):
    """ Analyze geometry in time.
    """
    info_cyan("Analyzing the evolution of the geometry through time.")
    f_mask = df.Function(ts.function_space)
    f_mask_x = []
    for d in range(ts.dim):
        f_mask_x.append(df.Function(ts.function_space))

    length = np.zeros(len(ts))
    area = np.zeros(len(ts))
    com = np.zeros((len(ts), ts.dim))

    makedirs_safe(os.path.join(ts.analysis_folder, "contour"))
    for step in range(len(ts)):
        info("Step " + str(step) + " of " + str(len(ts)))

        phi = ts["phi", step][:, 0]
        mask = 0.5*(1.-np.sign(phi))
        ts.set_val(f_mask, mask)
        for d in range(ts.dim):
            ts.set_val(f_mask_x[d], mask*ts.nodes[:, d])

        contour_file = os.path.join(ts.analysis_folder, "contour",
                                    "contour_{:06d}.dat".format(step))
        paths = zero_level_set(ts.nodes, ts.elems, phi,
                               save_file=contour_file)

        length[step] = path_length(paths)

        area[step] = df.assemble(f_mask*df.dx)
        for d in range(ts.dim):
            com[step, d] = df.assemble(f_mask_x[d]*df.dx)

    for d in range(ts.dim):
        com[:, d] /= area

    if rank == 0:
        np.savetxt(os.path.join(ts.analysis_folder,
                                "time_data.dat"),
                   np.array(zip(np.arange(len(ts)), ts.times, length, area,
                                com[:, 0], com[:, 1])),
                   header="Timestep\tTime\tLength\tCoM_x\tCoM_y")


def line_probe(ts, dx=0.1, line="[0.,0.]--[1.,1.]",
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

    for step in range(len(ts)):
        info("Step " + str(step) + " of " + str(len(ts)))
        ts.update_all(f, step)
        for field, probe in probes.iteritems():
            probe(f[field])

    probe_arr = dict()
    for field, probe in probes.iteritems():
        probe_arr[field] = probe.array()

    if rank == 0:
        for step in range(len(ts)):
            chunks = [x]
            header_list = [index2letter(d) for d in range(ts.dim)]
            for field, chunk in probe_arr.iteritems():
                if chunk.ndim == 2:
                    header_list.append(field)
                    chunk = chunk[:, step].reshape(-1, 1)
                elif chunk.ndim > 2:
                    header_list.extend(
                        [field + "_" + index2letter(d)
                         for d in range(ts.dim)])
                    chunk = chunk[:, :, step]
                chunks.append(chunk)

            data = np.hstack(chunks)
            header = "\t".join(header_list)
            makedirs_safe(os.path.join(ts.analysis_folder, "probes"))
            np.savetxt(os.path.join(ts.analysis_folder, "probes",
                                    "probes_{:06d}.dat".format(step)),
                       data, header=header)


def make_gif(ts, show=False, save=True, **kwargs):
    """ Make fancy gif animation. """
    info_cyan("Making a fancy gif animation.")
    anim_name = "animation"
    ts.compute_charge()

    for step in range(rank, len(ts), size):
        info("Step " + str(step) + " of " + str(len(ts)))
        phi = ts["phi", step][:, 0]
        charge = ts["charge", step][:, 0]
        charge_max = max(ts.max("charge"), -ts.min("charge"))

        if save:
            save_file = os.path.join(ts.tmp_folder,
                                     anim_name + "_{:06d}.png".format(step))
        else:
            save_file = None

        plot_fancy(ts.nodes, ts.elems, phi, charge,
                   charge_max=charge_max, show=show,
                   save=save_file)

    comm.Barrier()
    if save and rank == 0:
        tmp_files = os.path.join(ts.tmp_folder, anim_name + "_*.png")
        anim_file = os.path.join(ts.plots_folder, anim_name + ".gif")

        os.system(("convert {tmp_files} -trim +repage"
                   " -loop 0 {anim_file}").
                  format(tmp_files=tmp_files,
                         anim_file=anim_file))
        os.system("rm {tmp_files}".format(tmp_files=tmp_files))


def plot(ts, time=None, step=0, show=True, save=False, **kwargs):
    """ Plot at given timestep using matplotlib. """
    info_cyan("Plotting at given time/step using Matplotlib.")
    step, time = get_step_and_info(ts, time)
    if rank == 0:
        for field in ts.fields:
            if save:
                save_fig_file = os.path.join(
                    ts.plots_folder, "{}_{:06d}.png".format(field, step))
            else:
                save_fig_file = None

            plot_any_field(ts.nodes, ts.elems, ts[field, step],
                           save=save_fig_file, show=show, label=field)


def plot_dolfin(ts, time=None, step=0, **kwargs):
    """ Plot at given time/step using dolfin. """
    info_cyan("Plotting at given timestep using Dolfin.")
    step, time = get_step_and_info(ts, time)
    f = ts.functions()
    for field in ts.fields:
        ts.update(f[field], field, step)
        df.plot(f[field])
    df.interactive()


def get_step_and_info(ts, time):
    if time is not None:
        step, time = ts.get_nearest_step_and_time(time)
    else:
        time = ts.get_time(step)
    info("Time = {}, timestep = {}.".format(time, step))
    return step, time


def reference(ts, ref=None, time=1., show=False, save_fig=False, **kwargs):
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

    from fenicstools import interpolate_nonmatching_mesh

    f = ts.functions()
    err = ts.functions()
    f_ref = ts_ref.functions()

    ts.update_all(f, step=step)
    ts_ref.update_all(f_ref, step=step)

    for field in ts_ref.fields:
        F = interpolate_nonmatching_mesh(
            f_ref[field], f[field].function_space())
        err[field].vector()[:] = f[field].vector()[:] - F.vector()[:]

        if show or save_fig:
            err_arr = ts.nodal_values(err[field])
            label = "Error in " + field

            if rank == 0:
                save_fig_file = None
                if save_fig:
                    save_fig_file = os.path.join(
                        ts.plots_folder, "error_{}_time{}_ref{}.png".format(
                            field, time, ref_id))

                plot_any_field(ts.nodes, ts.elems, err_arr,
                               save=save_fig_file, show=show, label=label)

    save_file = os.path.join(ts.analysis_folder,
                             "errornorms_time{}_ref{}.dat".format(
                                 time, ref_id))

    compute_norms(err, f_ref, save=save_file)

    # info_on_red("Not implemented yet.")


def analytic_reference(ts, time=None, step=0, show=False,
                       save_fig=False, **kwargs):
    """ Compare to analytic reference expression at given timestep.
    This is done by importing the function "reference" in the problem module.
    """
    info_cyan("Comparing to analytic reference at given time or step.")
    step, time = get_step_and_info(ts, time)
    parameters = ts.get_parameters(time=time)
    problem = parameters.get("problem", "intrusion_bulk")
    exec("from problems.{} import reference".format(problem))
    ref_exprs = reference(**parameters)

    info("Comparing to analytic solution.")
    info_split("Problem:", "{}".format(problem))
    info_split("Time:", "{}".format(time))

    err = ts.functions(ref_exprs.keys())
    f = ts.functions(ref_exprs.keys())
    f_ref = ts.functions(ref_exprs.keys())
    for field, ref_expr in ref_exprs.iteritems():
        ref_expr.t = time
        f_ref[field].assign(df.interpolate(
            ref_expr, f[field].function_space()))
        ts.update(f[field], field, step)

        err[field].vector()[:] = (f[field].vector()[:] -
                                  f_ref[field].vector()[:])

        if show or save_fig:
            err_arr = ts.nodal_values(err[field])
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
    compute_norms(err, f_ref, save=save_file)


def compute_norms(err, f_ref, vector_norms=["l2", "linf"],
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


def mesh(ts, show=True, save_fig=False, **kwargs):
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
                   save=save_fig_file, show=show)


def get_help():
    info("Usage:\n   python " + os.path.basename(__file__) +
         " method=... [optional arguments]\n")
    info_cyan("{:<18} {}".format(
        "Method", "Optional arguments (=default value)"))
    for method in __methods__:
        func = globals()[method]
        opt_args_str = ""
        argcount = func.__code__.co_argcount
        if argcount > 1:
            opt_args = zip(func.__code__.co_varnames[1:argcount],
                           func.__defaults__)

            opt_args_str = ", ".join(["=".join([str(item)
                                                for item in pair])
                                      for pair in opt_args])
        info("{method:<18} {opt_args_str}".format(
            method=method, opt_args_str=opt_args_str))
    exit()


def main():
    info_yellow("BERNAISE: Post-processing tool")
    cmd_kwargs = parse_command_line()

    # Get help if it was called for.
    if cmd_kwargs.get("help", False):
        get_help()

    folder = cmd_kwargs.get("folder", False)

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

    # Call the specified method
    if method in __methods__:
        globals()[method](ts, **cmd_kwargs)
    else:
        info_on_red("The specified analysis method doesn't exist.")


if __name__ == "__main__":
    main()
