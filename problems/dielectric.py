import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
__author__ = "Gaute Linga"


class PeriodicBoundary(df.SubDomain):
    # Left boundary is target domain
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.Lx
        y[1] = x[1]


class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[1], 0.) and on_boundary)


class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly) and on_boundary)


def problem():
    info_cyan("A two-phase dielectricum.")

    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 1e-2, 4., 1.],
               ["c_m", -1, 1e-4, 1e-2, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 1./4
    sigma_e = 10.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_dielectric",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.08,
        t_0=0.,
        T=20.,
        grid_spacing=factor*1./16,
        interface_thickness=factor*0.080,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=1.,
        undulation_amplitude=0.02,
        undulation_periods=1.,
        surface_charge_top=sigma_e,
        surface_charge_bottom=-sigma_e,
        concentration_init=2.,
        #
        surface_tension=1.25,  # 24.5,
        grav_const=0.0,
        #
        pf_mobility_coeff=factor*0.000010,
        density=[10., 10.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
    )
    return parameters


def constrained_domain(Ly, **namespace):
    return PeriodicBoundary(Ly)


def mesh(Lx=1., Ly=1., grid_spacing=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))


def initialize(Lx, Ly,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               concentration_init,
               undulation_amplitude, undulation_periods,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                Ly/2, undulation_amplitude, undulation_periods,
                interface_thickness, field_to_subspace["phi"])
        if enable_EC:
            for solute in solutes:
                c_init = initial_phasefield(
                    Ly/2, undulation_amplitude, undulation_periods,
                    interface_thickness, field_to_subspace["phi"])
                # Only have ions in phase 2 (phi=-1)
                c_init.vector()[:] = concentration_init*0.5*(
                    1.-c_init.vector().array())
                # w_init_field[solute[0]] = initial_c(
                #     Lx/2, 0., Ly/6, concentration_init,
                #     field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init

    return w_init_field


def create_bcs(Lx, Ly, surface_charge_top, surface_charge_bottom,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        top=[Top(Ly)],
        bottom=[Bottom()]
    )

    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    noslip = Fixed((0., 0.))

    if enable_NS:
        bcs["top"]["u"] = noslip
        bcs["bottom"]["u"] = noslip
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs["top"]["V"] = Charged(surface_charge_top)
        bcs["bottom"]["V"] = Charged(surface_charge_bottom)
        bcs_pointwise["V"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")
    return boundaries, bcs, bcs_pointwise


def initial_phasefield(y0, A, n_periods, eps, function_space):
    expr_str = "tanh((x[1]+A*sin(2*n*pi*x[0])-y0)/(sqrt(2)*eps))"
    phi_init_expr = df.Expression(expr_str, y0=y0, A=A, n=n_periods, eps=eps,
                                  degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space.collapse())
    return phi_init


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    # func = 1.-phi**2
    # return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func
    return gamma


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))

    if False and stats_intv and tstep % stats_intv == 0:
        # GL: Seems like a rather awkward way of doing this,
        # but any other way seems to fuck up the simulation.
        # Anyhow, a better idea could be to move some of this to a post-processing stage.
        # GL: Move into common/utilities at a certain point.
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


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def initial_c(x, y, rad, c_init, function_space):
#    expr_str = ("c_init*1./(2*pi*pow(sigma, 2)) * "
#                "exp(- 0.5*pow((x[0]-x0)/sigma, 2)"
#               " - 0.5*pow((x[1]-y0)/sigma, 2))")
    expr_str = ("2*c_init*1./(2*pi*pow(sigma, 2)) * "
                "exp(- 0.5*pow((x[1]-y0)/sigma, 2))")
    c_init_expr = df.Expression(expr_str, x0=x, y0=y, sigma=rad,
                                c_init=c_init, degree=2)
    return df.interpolate(c_init_expr, function_space)
