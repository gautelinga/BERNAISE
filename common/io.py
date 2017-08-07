import os
from dolfin import MPI, mpi_comm_world, XDMFFile, HDF5File, Mesh
from cmd import info_red, info_cyan
import simplejson as json

__author__ = "Gaute Linga"
__date__ = "2017-05-26"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"

__all__ = ["mpi_is_root", "makedirs_safe", "load_parameters",
           "dump_parameters", "create_initial_folders",
           "save_solution", "save_checkpoint", "load_checkpoint",
           "load_mesh", "remove_safe"]


def mpi_is_root():
    """ Check if current MPI node is root node. """
    return MPI.rank(mpi_comm_world()) == 0


def makedirs_safe(folder):
    """ Make directory in a safe way. """
    if mpi_is_root() and not os.path.exists(folder):
        os.makedirs(folder)


def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_is_root() and os.path.exists(path):
        os.remove(path)


def dump_parameters(parameters, settingsfilename):
    """ Dump parameters to file """
    with file(settingsfilename, "w") as settingsfile:
        json.dump(parameters, settingsfile, indent=4*' ', sort_keys=True)


def load_parameters(parameters, settingsfilename):
    with file(settingsfilename, "r") as settingsfile:
        parameters.update(json.load(settingsfile))


def create_initial_folders(folder, restart_folder, fields, tstep,
                           parameters):
    """ Create initial folders """
    info_cyan("Creating folders.")

    makedirs_safe(folder)
    MPI.barrier(mpi_comm_world())
    if restart_folder:
        newfolder = restart_folder.split("Checkpoint")[0]
    else:
        previous_list = os.listdir(folder)
        if len(previous_list) == 0:
            newfolder = os.path.join(folder, "1")
        else:
            previous = max([int(entry) if entry.isdigit() else 0
                            for entry in previous_list])
            newfolder = os.path.join(folder, str(previous+1))

    MPI.barrier(mpi_comm_world())
    tstepfolder = os.path.join(newfolder, "Timeseries")
    makedirs_safe(tstepfolder)
    makedirs_safe(os.path.join(newfolder, "Statistics"))
    settingsfolder = os.path.join(newfolder, "Settings")
    makedirs_safe(settingsfolder)
    makedirs_safe(os.path.join(newfolder, "Checkpoint"))

    # Initialize timestep files
    tstepfiles = dict()
    for field in fields:
        filename = os.path.join(tstepfolder,
                                field + "_from_tstep_{}.xdmf".format(tstep))
        tstepfiles[field] = XDMFFile(mpi_comm_world(), filename)
        tstepfiles[field].parameters["rewrite_function_mesh"] = False
        tstepfiles[field].parameters["flush_output"] = True

    # Dump settings
    if mpi_is_root():
        dump_parameters(parameters, os.path.join(
            settingsfolder, "parameters_from_tstep_{}.dat".format(tstep)))

    return newfolder, tstepfiles


def save_solution(tstep, t, T, w_, w_1, folder, newfolder,
                  save_intv, checkpoint_intv,
                  parameters, tstepfiles, subproblems, **namespace):
    """ Save solution either to  """
    if tstep % save_intv == 0:
        # Save snapshot to xdmf
        save_xdmf(t, w_, subproblems, tstepfiles)

    stop = check_if_kill(folder) or t >= T
    if tstep % checkpoint_intv == 0 or stop:
        # Save checkpoint
        save_checkpoint(tstep, t, w_, w_1, newfolder, parameters)

    return stop


def check_if_kill(folder):
    """ Check if user has ordered to kill the simulation. """
    found = 0
    if "kill" in os.listdir(folder):
        found = 1
    found_all = MPI.sum(mpi_comm_world(), found)
    if found_all > 0:
        remove_safe(os.path.join(folder, "kill"))
        info_red("Stopping simulation.")
        return True
    else:
        return False


def save_xdmf(t, w_, subproblems, tstepfiles):
    """ Save snapshot of solution to xdmf file. """
    for name, subproblem in subproblems.iteritems():
        q_ = w_[name].split()
        if len(subproblem) > 1:
            for s, q in zip(subproblem, q_):
                field = s["name"]
                if field in tstepfiles:
                    q.rename(field, "tmp")
                    tstepfiles[field].write(q, float(t))
        else:
            field = subproblem[0]["name"]
            if field in tstepfiles:
                q = w_[name]
                q.rename(field, "tmp")
                tstepfiles[field].write(q, float(t))


def save_checkpoint(tstep, t, w_, w_1, newfolder, parameters):
    """ Save checkpoint files.

    A part of this is taken from the Oasis code."""
    checkpointfolder = os.path.join(newfolder, "Checkpoint")
    parameters["num_processes"] = MPI.size(mpi_comm_world())
    parameters["t_0"] = t
    parameters["tstep"] = tstep
    parametersfile = os.path.join(checkpointfolder, "parameters.dat")
    parametersfile_old = parametersfile + ".old"
    if mpi_is_root():
        # In case of failure, keep old file.
        if os.path.exists(parametersfile):
            os.system("mv {0} {1}".format(parametersfile,
                                          parametersfile_old))
        dump_parameters(parameters, parametersfile)

    MPI.barrier(mpi_comm_world())
    h5filename = os.path.join(checkpointfolder, "fields.h5")
    h5filename_old = h5filename + ".old"
    # In case of failure, keep old file.
    if mpi_is_root() and os.path.exists(h5filename):
        os.system("mv {0} {1}".format(h5filename, h5filename_old))
    h5file = HDF5File(mpi_comm_world(), h5filename, "w")
    h5file.flush()
    for field in w_:
        info_red("Storing subproblem: " + field)
        MPI.barrier(mpi_comm_world())
        h5file.write(w_[field], field + "/current")
        if field in w_1:
            h5file.write(w_1[field], field + "/previous")
        MPI.barrier(mpi_comm_world())
    h5file.close()
    # Since program is still running, delete the old files.
    remove_safe(h5filename_old)
    MPI.barrier(mpi_comm_world())
    remove_safe(parametersfile_old)


def load_checkpoint(checkpointfolder, w_, w_1):
    if checkpointfolder:
        h5filename = os.path.join(checkpointfolder, "fields.h5")
        h5file = HDF5File(mpi_comm_world(), h5filename, "r")
        for field in w_:
            info_red("Loading subproblem: " + field)
            h5file.read(w_[field], field + "/current")
            h5file.read(w_1[field], field + "/previous")
        h5file.close()


def load_mesh(filename):
    """ Loads in the mesh specified by the argument filename. """
    info_cyan("Loading mesh: " + filename)
    mesh = Mesh()
    h5file = HDF5File(mesh.mpi_comm(), filename, "r")
    h5file.read(mesh, "mesh", False)
    h5file.close()
    return mesh
