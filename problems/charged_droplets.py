import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed

__author__ = "Gaute Linga"

info_cyan("Two charged droplets")


class Wall(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary)


def problem():
    # Define solutes
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 1.e-3, 3., 1.],
               ["c_m", -1, 1e-4, 1.e-3, 3., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_charged_droplets",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.02,
        t_0=0.,
        T=20.,
        dx=1./64,
        interface_thickness=0.02,
        solutes=solutes,
        base_elements=base_elements,
        Lx=3.,
        Ly=1.,
        rad_init=0.2,
        #
        V_boundary=0.,
        surface_tension=24.5,
        grav_const=0.,
        #
        pf_mobility_coeff=0.000010,
        density=[100., 100.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
        #
        use_iterative_solvers=False,
        use_pressure_stabilization=False
    )
    return parameters


def mesh(Lx=1, Ly=5, dx=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/dx), int(Ly/dx))


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ Create the initial state. """
    x0 = [Lx/3, 2*Lx/3]
    y0 = [Ly/2, Ly/2]
    rad0 = [rad_init, rad_init]
    c0 = [10., 10.]

    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_pf(
                x0, y0, rad0, interface_thickness,
                field_to_subspace["phi"].collapse())

        # Electrochemistry
        if enable_EC:
            for x, y, rad, ci, solute in zip(x0, y0, rad0, c0, solutes):
                c_init = initial_c(x, y, rad/3., ci, interface_thickness,
                                   field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression("0.", degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

    return w_init_field


def create_bcs(field_to_subspace, Lx, Ly, solutes,
               V_boundary,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundary conditions are defined in terms of field. """

    boundaries = dict(wall=[Wall()])

    bcs = dict(
        wall=dict()
    )
    bcs_pointwise = dict()

    noslip = Fixed((0., 0.))

    # Navier-Stokes
    if enable_NS:
        bcs["wall"]["u"] = noslip
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    # Electrochemistry
    if enable_EC:
        bcs["wall"]["V"] = Fixed(V_boundary)

    return boundaries, bcs, bcs_pointwise


def initial_pf(x0, y0, rad0, eps, function_space):
    expr_str = "1."
    for x, y, rad in zip(x0, y0, rad0):
        expr_str += ("-(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
                     "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
                        x=x, y=y, rad=rad, eps=eps)
    phi_init_expr = df.Expression(expr_str, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def initial_c(x, y, rad, c_init, eps, function_space):
    # expr_str = ("{c_init}*0.5*(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
    #             "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
    #                 x=x, y=y, rad=rad, eps=eps, c_init=c_init)
    expr_str = ("c_init*1./(2*pi*pow(sigma, 2)) * "
                "exp(- 0.5*pow((x[0]-x0)/sigma, 2)"
                " - 0.5*pow((x[1]-y0)/sigma, 2))")
    c_init_expr = df.Expression(expr_str, x0=x, y0=y, sigma=rad,
                                c_init=c_init, degree=2)
    return df.interpolate(c_init_expr, function_space)


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))

    if False and stats_intv and tstep % stats_intv == 0:
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
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
