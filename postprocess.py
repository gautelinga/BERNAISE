import dolfin as df
import h5py
from common import load_parameters, info, parse_command_line, makedirs_safe, \
    info_blue, info_cyan, info_split, info_on_red
import os
import glob
import numpy as np
from utilities.generate_mesh import numpy_to_dolfin
from mpi4py.MPI import COMM_WORLD
from utilities.plot import plot_contour, plot_edges, plot_quiver, plot_faces,\
    zero_level_set
from xml.etree import cElementTree as ET


comm = COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_middle(string, prefix, suffix):
    return string.split(prefix)[1].split(suffix)[0]


def prep(x_list):
    M = 1000
    x_np = np.ceil(np.array(x_list)*M).astype(float)/M
    return tuple(x_np.tolist())


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


class TimeSeries:
    """ Class for loading timeseries """
    def __init__(self, folder, sought_fields=None, get_mesh_from=False):
        self.folder = folder

        self.settings_folder = os.path.join(folder, "Settings")
        self.timeseries_folder = os.path.join(folder, "Timeseries")
        self.statistics_folder = os.path.join(folder, "Statistics")
        self.analysis_folder = os.path.join(folder, "Analysis")
        makedirs_safe(self.analysis_folder)

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
            xdict = dict([(prep(x_list), i) for i, x_list in
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

    def get_parameter(self, key, time=0.):
        for t, parameters in self.parameters:
            if t <= time:
                return parameters[key]

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

    def functions(self):
        """ Returns dolfin functions for all fields. """
        f = dict()
        for field in self.fields:
            f[field] = self.function(field)
        return f

    def __len__(self):
        return len(self.times)


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


def main():
    cmd_kwargs = parse_command_line()

    method = cmd_kwargs.get("method", "geometry_in_time")
    folder = cmd_kwargs.get("folder", False)
    sought_fields = cmd_kwargs.get("fields", False)
    if not sought_fields:
        sought_fields = None
    elif not isinstance(sought_fields, list):
        sought_fields = [sought_fields]

    if not folder:
        info("No folder(=[...]) specified.")

    sought_fields_str = (", ".join(sought_fields)
                         if sought_fields is not None else "All")

    info_split("Sought fields:", sought_fields_str)

    ts = TimeSeries(folder, sought_fields=sought_fields)

    fields = ts.fields

    info_split("Found fields:", ", ".join(fields))

    step = cmd_kwargs.get("step", 0)

    problem = cmd_kwargs.get("problem", "intrusion_bulk")

    # ts2 = TimeSeries(folder, get_mesh_from=ts)

    # f2 = df.Function(ts2.function_space)
    # ts2.update(f2, "phi", 50)


    # ts.update(u, "u", 0)
    if method == "geometry_in_time":
        geometry_in_time(ts)
    elif method == "reference":
        info_on_red("Not implemented yet.")
    elif method == "analytic_reference":
        exec("from problems.{} import reference".format(problem))
        info_on_red("Not implemented yet.")
    elif method == "plot":
        for field in fields:
            if field == "u":
                plot_quiver(ts.nodes, ts.elems, ts["u", step])
            else:
                plot_contour(ts.nodes, ts.elems, ts[field, step][:, 0])
    elif method == "plot_dolfin":
        f = ts.functions()
        for field in fields:
            ts.update(f[field], field, step)
            df.plot(f[field])
        df.interactive()
    else:
        info_on_red("The specified analysis method doesn't exist.")


if __name__ == "__main__":
    main()
