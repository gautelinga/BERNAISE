import dolfin as df
from common import *

"""
This module contains general functions that can or should be overloaded by
problem specific functions.

"""
# Start with defining some common compiler options.
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
# df.parameters["form_compiler"]["representation"] = "quadrature"
df.parameters["linear_algebra_backend"] = "PETSc"
df.parameters["std_out_all_processes"] = False
# df.parameters["krylov_solver"]["nonzero_initial_guess"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
# df.set_log_active(False)

parameters = dict(
    folder="results"  # default folder to store results in
)


def initialize(**namespace):
    """ Initialize solution """
    pass


def create_bcs(fields, **namespace):
    """ Return a dict of DirichletBCs. """
    return dict((field, []) for field in fields)


def start_hook(**namespace):
    """ Called just before entering the time loop. """
    return dict()


def end_hook(**namespace):
    """ Called just before program ends. """
    pass


def import_problem_hook(parameters, mesh, cmd_kwargs, **namespace):
    """ Called after importing problem. """
    for key, val in cmd_kwargs.iteritems():
        if isinstance(val, dict):
            parameters[key].update(val)
        else:
            parameters[key] = val

    # Internalize the mesh
    if callable(mesh):
        mesh = mesh(**parameters)
    assert(isinstance(mesh, df.Mesh))

    namespace_dict = dict(mesh=mesh)
    namespace_dict.update(parameters)
    return namespace_dict
