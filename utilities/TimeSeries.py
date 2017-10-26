import numpy as np
import os
import sys
from mpi4py import MPI
import h5py
import glob
# Find path to the BERNAISE root folder
bernaise_path = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-2])
# ...and append it to sys.path to get functionality from BERNAISE
sys.path.append(bernaise_path)
from generate_mesh import numpy_to_dolfin
from common import makedirs_safe, info_warning, info_split, info_on_red, \
    load_parameters, parse_xdmf
import dolfin as df


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def get_middle(string, prefix, suffix):
    return string.split(prefix)[1].split(suffix)[0]


def prep(x_list):
    """ Prepare a list representing coordinates to be used as key in a
    dict. """
    # M = 10000
    # x_np = np.ceil(np.array(x_list)*M).astype(float)/M
    # return tuple(x_np.tolist())
    return tuple(x_list)


class TimeSeries:
    """ Class for loading timeseries """
    def __init__(self, folder, sought_fields=None, get_mesh_from=False,
                 memory_modest=True):
        self.folder = folder

        self.settings_folder = os.path.join(folder, "Settings")
        self.timeseries_folder = os.path.join(folder, "Timeseries")
        self.statistics_folder = os.path.join(folder, "Statistics")
        self.analysis_folder = os.path.join(folder, "Analysis")
        self.plots_folder = os.path.join(folder, "Plots")
        self.tmp_folder = os.path.join(folder, ".tmp")

        self.memory_modest = memory_modest

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
                            # If in memory saving mode, only store
                            # address for later use.
                            if self.memory_modest:
                                data[field][time] = (data_file, dset_address)
                            else:
                                data[field][time] = np.array(h5f[dset_address])

        for i, field in enumerate(data.keys()):
            tmps = sorted(data[field].items())
            if i == 0:
                self.times = [tmp[0] for tmp in tmps]
            self[field] = [tmp[1] for tmp in tmps]
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
            u_data = self["u", step][:, :self.dim]
            for i in range(self.dim):
                self.set_val(self.dummy_function, u_data[:, i])
                df.assign(f.sub(i), self.dummy_function)
        else:
            f_data = self[field, step][:]
            self.set_val(f, f_data)

    def update_all(self, f, step):
        """ Set dict f of dolfin functions with values from all fields. """
        for field in self.fields:
            self.update(f[field], field, step)

    def get_parameter(self, key, time=0., default=False):
        """ Get a certain parameter at certain time. """
        return self.get_parameters(time).get(key, default)

    def get_parameters(self, time=0.):
        """ Get parameter set at a certain time. """
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
            if self.memory_modest:
                info_warning("TimeSeries[key]: len(key)==1 doesn't work "
                             "in memory_modest mode!")
            return self.datasets[key]
        if len(key) == 2:
            field, step = key
            if self.memory_modest:
                data_file, dset_address = self.datasets[field][step]
                with h5py.File(data_file, "r") as h5f:
                    return np.array(h5f[dset_address])
            else:
                return self.datasets[field][step]

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
        if abs(time-time_0) > 1e-10:
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
        if self.memory_modest:
            data_file = os.path.join(self.tmp_folder,
                                     field + ".h5")
            self[field] = [(data_file, field + "/" + str(step))
                           for step in range(len(datasets))]
            if rank == 0:
                with h5py.File(data_file, "w") as h5f:
                    for step, dataset in enumerate(datasets):
                        dset_address = field + "/" + str(step)
                        h5f.create_dataset(dset_address, data=dataset)
            comm.Barrier()
        else:
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
