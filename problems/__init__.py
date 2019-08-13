import dolfin as df
from common import *
from common.cmd import info_cyan, info_blue, info_red, info_green, info_error

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

# Default base elements
# Format: name : (family, degree, is_vector)
base_elements = dict(u=["Lagrange", 2, True],
                     p=["Lagrange", 1, False],
                     phi=["Lagrange", 1, False],
                     g=["Lagrange", 1, False],
                     c=["Lagrange", 1, False],
                     V=["Lagrange", 1, False],
                     p0=["Real", 0, False],
                     c0=["Real", 0, False],
                     V0=["Real", 0, False])

# Set default parameters
parameters = dict(
    folder="results",  # default folder to store results in
    restart_folder=False,
    info_intv=10,
    use_iterative_solvers=False,
    use_pressure_stabilization=False,
    dump_subdomains=False,
    V_lagrange=False,
    p_lagrange=False,
    base_elements=base_elements,
    c_cutoff=0.,
    q_rhs=dict(),
    EC_scheme="NL2",
    grav_dir=[1., 0],
    pf_mobility_coeff=1.,
    friction_coeff=0.,
    grav_const=0.,
    surface_tension=0.,
    interface_thickness=0.,
    reactions=[],
    density_per_concentration=None,
    viscosity_per_concentration=None,
    comoving_velocity=[0., 0., 0.],
    testing=False,
    tstep=0,
    enable_PF=True,
    enable_EC=True,
    enable_NS=True,
    save_intv=5,
    checkpoint_intv=50,
    stat_intv=5,
    solve_initial=False,
    freeze_NSPF=False
)


def constrained_domain(**namespace):
    """ Returns e.g. periodic domain. """
    return None


def initialize(**namespace):
    """ Initialize solution """
    pass


def create_bcs(fields, **namespace):
    """ Return a dict of DirichletBCs. """
    return dict((field, []) for field in fields)


def start_hook(**namespace):
    """ Called just before entering the time loop. """
    return dict()


def tstep_hook(**namespace):
    """ Called in the beginning of timestep loop. """
    pass


def end_hook(**namespace):
    """ Called just before program ends. """
    pass


def internalize_cmd_kwargs(parameters, cmd_kwargs):
    """ Integrate command line arguments into parameters.
    Consider moving this to common/cmd.py
    """
    for key, val in cmd_kwargs.items():
        if isinstance(val, dict):
            parameters[key].update(val)
        else:
            parameters[key] = val


def import_problem_hook(parameters, mesh, cmd_kwargs, **namespace):
    """ Called after importing problem. """
    internalize_cmd_kwargs(parameters, cmd_kwargs)

    # Internalize the mesh
    if callable(mesh):
        mesh = mesh(**parameters)
    assert(isinstance(mesh, df.Mesh))

    namespace_dict = dict(mesh=mesh)
    namespace_dict.update(parameters)
    return namespace_dict


def rhs_source(**namespace):
    """ External source terms on the right hand side of the conservation equations. """
    return dict()


def pf_mobility(phi, gamma):
    """ Default phase field mobility function. """
    return gamma
