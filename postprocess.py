import dolfin as df
import h5py
from common import load_parameters, info, parse_command_line, makedirs_safe, \
    info_blue, info_cyan, info_split, info_on_red, info_red, info_yellow
import os
import glob
import numpy as np
from utilities.generate_mesh import numpy_to_dolfin
from mpi4py.MPI import COMM_WORLD
from utilities.plot import plot_contour, plot_edges, plot_quiver, plot_faces,\
    zero_level_set, plot_probes, plot_fancy
from xml.etree import cElementTree as ET
from utilities.generate_mesh import line_points

"""
BERNAISE: Post-processing tool.

This module is used to read, and perform analysis on, simulated data.

"""

__methods__ = ["geometry_in_time", "reference", "plot",
               "plot_dolfin", "plot_mesh", "line_probe", "make_gif"]
__all__ = ["get_middle", "prep", "path_length", "parse_xdmf",
           "index2letter"] + __methods__

comm = COMM_WORLD
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


def parse_xdmf(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    dsets = []
    for step in root[0][0]:
        timestamp = None
        dset_address = None
        for prop in step:
            if prop.tag == "Time":
                timestamp = float(prop.attrib["Value"])
            elif prop.tag == "Attribute":
                dset_address = prop[0].text.split(":")[1]
        dsets.append((timestamp, dset_address))
    return dsets


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
        else:
            self.mesh = get_mesh_from.mesh
            self.function_space = get_mesh_from.function_space
            self.vector_function_space = get_mesh_from.vector_function_space
            self.dim = get_mesh_from.dim
            self.x = get_mesh_from.x
            self.xdict = get_mesh_from.xdict

    def _load_timeseries(self, sought_fields=None):
        if bool(os.path.exists(self.settings_folder) and
                os.path.exists(self.timeseries_folder)):
            info_split("Opening folder:", self.folder)
        else:
            info("Folder does not contain Settings or Timeseries folders.")
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


def geometry_in_time(ts):
    """ Analyze geometry in time. """
    f_mask = df.Function(ts.function_space)
    f_mask_x = []
    for d in range(ts.dim):
        f_mask_x.append(df.Function(ts.function_space))

    lengths = np.zeros(len(ts))
    areas = np.zeros(len(ts))
    areas_x = []
    for d in range(ts.dim):
        areas_x.append(np.zeros(len(ts)))

    for step in range(len(ts)):
        info("Step " + str(step) + " of " + str(len(ts)))

        phi = ts["phi", step][:, 0]
        mask = 0.5*(1.-np.sign(phi))
        ts.set_val(f_mask, mask)
        for d in range(ts.dim):
            ts.set_val(f_mask_x[d], mask*ts.nodes[:, d])

        # plot_contour(ts.nodes, ts.elems, phi)
        contour_file = os.path.join(ts.analysis_folder,
                                    "contour_" + str(step) + ".dat")
        paths = zero_level_set(ts.nodes, ts.elems, phi,
                               save_file=contour_file)
        # paths = zero_level_set(ts.nodes, ts.elems, phi)
        lengths[step] = path_length(paths)

        areas[step] = df.assemble(f_mask*df.dx)
        for d in range(ts.dim):
            areas_x[d][step] = df.assemble(f_mask_x[d]*df.dx)

    if rank == 0:
        np.savetxt(os.path.join(ts.analysis_folder,
                                "time_data.dat"),
                   np.array(zip(ts.times, lengths, areas,
                                areas_x[0], areas_x[1])))


def line_probe(ts, dx, line_str):
    from fenicstools import Probes
    x_a, x_b = [tuple(eval(pt)) for pt in line_str.split("--")]
    x = np.array(line_points(x_a, x_b, dx))

    plot_probes(ts.nodes, ts.elems, x, colorbar=False, title="Probes")

    f = ts.functions()
    probes = dict()
    for field, func in f.iteritems():
        probes[field] = Probes(x.flatten(), func.function_space())

    for step in range(len(ts)):
        info("Step " + str(step) + " of " + str(len(ts)))
        ts.update_all(f, step)
        for field, probe in probes.iteritems():
            probe(f[field])

    for step in range(len(ts)):
        chunks = [x]
        header_list = [index2letter(d) for d in range(ts.dim)]
        for field, probe in probes.iteritems():
            chunk = probe.array()
            if chunk.ndim == 2:
                header_list.append(field)
                chunk = chunk[:, step].reshape(-1, 1)
            elif chunk.ndim > 2:
                header_list.extend(
                    [field + "_" + index2letter(d) for d in range(ts.dim)])
                chunk = chunk[:, :, step]
            chunks.append(chunk)
        data = np.hstack(chunks)
        header = "\t".join(header_list)
        np.savetxt(os.path.join(ts.analysis_folder,
                                "probes_" + str(step) + ".dat"),
                   data, header=header)


def compute_charge_datasets(ts):
    """ Computing charge datasets by summing over all species.
    GL: Should it be a member function of TimeSeries?
    """
    solutes = ts.get_parameter("solutes")
    charge_datasets = []
    for step in range(len(ts)):
        charge_loc = np.zeros_like(ts["phi", step])
        for solute in solutes:
            field = solute[0]
            z = solute[1]
            charge_loc[:, :] += z*ts[field, step]
        charge_datasets.append(charge_loc)
    return charge_datasets


def make_gif(ts, show=False, save=True):
    """ Make fancy gif animation. """
    anim_name = "animation"
    charge_datasets = compute_charge_datasets(ts)
    ts.add_field("charge", charge_datasets)

    for step in range(len(ts)):
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


def plot(ts, step, show=True, save=False):
    """ Plot at given timestep using matplotlib. """
    for field in ts.fields:
        if save:
            save_file = os.path.join(ts.plots_folder,
                                     field + "_" + str(step) + ".png")
        else:
            save_file = None

        if field == "u":
            plot_quiver(ts.nodes, ts.elems, ts[field, step],
                        title=field, clabel=field, save=save_file,
                        show=show)
        else:
            plot_contour(ts.nodes, ts.elems, ts[field, step][:, 0],
                         title=field, clabel=field, save=save_file,
                         show=show)


def plot_dolfin(ts, step):
    """ Plot at given timestep using dolfin. """
    f = ts.functions()
    for field in ts.fields:
        ts.update(f[field], field, step)
        df.plot(f[field])
    df.interactive()


def analytic_reference(ts, step):
    """ Compare to analytic reference expression at given timestep.
    This is done by importing the function "reference" in the problem module.
    """
    exec("from problems.{} import reference".format(problem))
    time = ts.get_time(step)
    parameters = ts.get_parameters(time=time)
    problem = parameters.get("problem", "intrusion_bulk")
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

    info_red("\nL2 norms")
    info("\tL2(f-f_ref) \tL2(f_ref)")
    for field, e in err.iteritems():
        info("{field} \t{e_L2} \t{f_L2}".format(
            field=field,
            e_L2=df.norm(e.vector()),
            f_L2=df.norm(f_ref[field].vector())))
        df.plot(e)
    df.interactive()


def main():
    info_yellow("BERNAISE: Post-processing tool")
    cmd_kwargs = parse_command_line()

    folder = cmd_kwargs.get("folder", False)
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

    step = cmd_kwargs.get("step", 0)

    method = cmd_kwargs.get("method", "geometry_in_time")

    if method == "geometry_in_time":
        geometry_in_time(ts)
    elif method == "reference":
        info_on_red("Not implemented yet.")
    elif method == "analytic_reference":
        analytic_reference(ts, step)
    elif method == "plot":
        show = bool(cmd_kwargs.get("show", True))
        save = bool(cmd_kwargs.get("save", False))
        plot(ts, step, show=show, save=save)
    elif method == "plot_dolfin":
        plot_dolfin(ts, step)
    elif method == "plot_mesh":
        plot_faces(ts.nodes, ts.elems, title="Mesh")
    elif method == "line_probe":
        dx = cmd_kwargs.get("dx", 0.1)
        line_str = cmd_kwargs.get("line", "[0.,0.]--[1.,1.]")
        line_probe(ts, dx, line_str)
    elif method == "make_gif":
        show = bool(cmd_kwargs.get("show", False))
        save = bool(cmd_kwargs.get("save", True))
        make_gif(ts, show=show, save=save)
    else:
        info_on_red("The specified analysis method doesn't exist.")


if __name__ == "__main__":
    main()
