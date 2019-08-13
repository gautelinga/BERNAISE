import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
from common.functions import max_value
__author__ = "Asger Bolet"


info_cyan("Charged wall for a test of Debye layers.")


class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.) and on_boundary)


class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx) and on_boundary)


class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[1], 0.0) and on_boundary)


class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly) and on_boundary)


def problem():
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-1, 1.e-1, 1., 1.],
               ["c_m", -1, 1e-1, 1.e-1, 1., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 1./2

    # SurfaceCharge = 1.9e-6 # [C]/[m]^2
    # DebyeLength = sqrt(epsilon/2*c0)
    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_charged_wall",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.8,
        t_0=0.,
        T=100.,
        grid_spacing=factor*1./16,
        interface_thickness=factor*0.080,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=5.,
        rad_init=0.25,
        #
        surface_charge=1.0,
        surface_tension=24.5,
        grav_const=0.0,
        #
        pf_mobility_coeff=factor*0.000040,
        density=[1000., 1000.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
        #
        initial_interface="flat",
        #
        use_iterative_solvers=False,
        use_pressure_stabilization=False
    )
    return parameters


def mesh(Lx=1, Ly=5, grid_spacing=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, initial_interface,
               **namespace):
    """ Create the initial state.
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.

    Note: You only need to specify the initial states that are nonzero.
    """

    c0 = [1., 1.]
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for ci, solute in zip(c0, solutes):
                solute_name = solute[0]
                w_init_field[solute_name] = initial_c(
                    ci,
                    field_to_subspace[solute_name].collapse())
            V_init_expr = df.Expression("0.", degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                Lx/5, Lx/2, rad_init, interface_thickness,
                field_to_subspace["phi"].collapse(), shape=initial_interface)

    return w_init_field


def create_bcs(Lx, Ly, solutes, surface_charge,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(0)],
        bottom=[Bottom(0)],
        top=[Top(Ly)]
    )
    c0 = [1.0, 1.0]
    noslip = Fixed((0., 0.))
    V_ground = Fixed(0.0)
    surfacecharge = Charged(surface_charge)

    bcs = dict()
    bcs_pointwise = dict()
    bcs["right"] = dict()
    bcs["left"] = dict()
    bcs["top"] = dict()
    bcs["bottom"] = dict()

    if enable_NS:
        bcs["right"]["u"] = noslip
        bcs["left"]["u"] = noslip
        bcs["top"]["u"] = noslip
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs["top"]["V"] = surfacecharge
        bcs["bottom"]["V"] = V_ground
        for ci, solute in zip(c0, solutes):
            bcs["bottom"][solute[0]] = Fixed(ci)

    # Apply pointwise BCs e.g. to pin pressure.

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, rad, eps, function_space, shape="circle"):
    if shape == "flat":
        expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    elif shape == "sine":
        expr_str = "tanh((x[0]-x0-10*eps*sin(x[1]*pi))/(sqrt(2)*eps))"
    elif shape == "circle":
        expr_str = ("tanh(sqrt(2)*(sqrt(pow(x[0]-x0,2)" +
                    "+pow(x[1]-y0,2))-rad)/eps)")
    else:
        info_red("Unrecognized shape: " + shape)
        exit()
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad,
                                  eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * max_value(0., func)


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def initial_c(c_init, function_space):
    # expr_str = ("{c_init}*0.5*(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
    #             "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
    #                 x=x, y=y, rad=rad, eps=eps, c_init=c_init)
    expr_str = ("c_init")
    c_init_expr = df.Expression(expr_str, c_init=c_init, degree=1)
    return df.interpolate(c_init_expr, function_space)
