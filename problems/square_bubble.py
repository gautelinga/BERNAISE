import dolfin as df
import os
from . import *
from common.io import mpi_is_root
import math
__author__ = "Gaute Linga"


info_cyan("Welcome to the square bubble problem!")

# Define solutes
# Format: name, valency, diffusivity in phase 1, diffusivity in phase
#         2, beta in phase 1, beta in phase 2
solutes = [["c_p",  1, 1., 1., 1., 1.],
           ["c_m", -1, 1., 1., 1., 1.]]

# Format: name : (family, degree, is_vector)
base_elements = dict(u=["Lagrange", 1, True],
                     p=["Lagrange", 1, False],
                     phi=["Lagrange", 1, False],
                     g=["Lagrange", 1, False],
                     c=["Lagrange", 1, False],
                     V=["Lagrange", 1, False])

# Default parameters to be loaded unless starting from checkpoint.
parameters.update(
    solver="basic",
    folder="results_square_bubble",
    restart_folder=False,
    enable_NS=True,
    enable_PF=True,
    enable_EC=False,
    save_intv=5,
    stats_intv=5,
    checkpoint_intv=50,
    tstep=0,
    dt=0.1,
    t_0=0.,
    T=10.,
    Nx=64,  # 257,
    Ny=64,  # 257,
    interface_thickness=0.04,
    solutes=solutes,
    base_elements=base_elements,
    Lx=2.,
    Ly=2.,
    rad_init=0.5,
    #
    V_top=1.,
    V_btm=0.,
    surface_tension=0.01/(3./(2*math.sqrt(2))),
    grav_const=0.0,
    #
    pf_mobility_coeff=2e-3/0.02,
    density=[10., 1.],
    viscosity=[10., 1.],
    permittivity=[1., 1.],
    #
    use_iterative_solvers=False
)


class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary)


def mesh(Lx=1, Ly=1, Nx=16, Ny=16, **namespace):
    msh = df.RectangleMesh(df.Point(-Lx/2., -Ly/2.),
                           df.Point(Lx/2., Ly/2.),
                           Nx, Ny)
    return msh


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ Create the initial state.
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.

    Note: You only need to specify the initial states that are nonzero.
    """
    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                0., 0., rad_init, interface_thickness,
                field_to_subspace["phi"].collapse())

    return w_init_field


def create_bcs(field_to_subspace, Lx, Ly, solutes,
               V_top, V_btm,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundary conditions are defined in terms of field. """
    bcs_fields = dict()

    # Navier-Stokes
    if enable_NS:
        dbc = DirichletBoundary()
        noslip = df.DirichletBC(field_to_subspace["u"],
                                df.Constant((0., 0.)),
                                dbc)
        bcs_fields["u"] = [noslip]

        p_pin = df.DirichletBC(
            field_to_subspace["p"],
            df.Constant(0.),
            "x[0] < -{Lx}/2+DOLFIN_EPS && x[1] < -{Ly}/2+DOLFIN_EPS".format(
                Lx=Lx, Ly=Ly),
            "pointwise")
        bcs_fields["p"] = [p_pin]

    return bcs_fields


def initial_phasefield(x0, y0, rad, eps, function_space):
    expr_str = ("tanh(sqrt(2)*(pow(pow(x[0]-x0, exponent)" +
                "+pow(x[1]-y0, exponent), 1./exponent) - rad)/eps)")
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad,
                                  eps=eps, degree=2, exponent=20)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, enable_PF, **namespace):
    info_blue("Timestep = {}".format(tstep))
    if stats_intv and tstep % stats_intv == 0 and enable_PF:
        # GL: Seems like a rather awkward way of doing this, but any
        # other way seems to fuck up the simulation.  Anyhow, a better
        # idea could be to move some of this to a post-processing
        # stage.
        subproblem_name, subproblem_i = field_to_subproblem["phi"]
        Q = w_[subproblem_name].split(deepcopy=True)[subproblem_i]
        bubble = df.interpolate(Q, field_to_subspace["phi"].collapse())
        bubble = 0.5*(1.-df.sign(bubble))
        mass = df.assemble(bubble*df.dx)
        massy = df.assemble(
            bubble*df.Expression("x[1]", degree=1)*df.dx)
        if mpi_is_root():
            with file(statsfile, "a") as outfile:
                outfile.write("{} {} {} \n".format(t, mass, massy))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # func = 1.-phi**2
    # return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func
    return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
