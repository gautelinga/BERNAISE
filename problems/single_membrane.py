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
    info_cyan("Single-phase electrohydrodynamic flow "
              "near a permselective membrane.")

    Re = 0.01
    lambda_D = 2e-2
    kappa = 0.5

    solutes = [["c_p",  1, 1., 1., 0., 0.],
               ["c_m", -1, 1., 1., 0., 0.]]

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_single_membrane",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=1e-5,
        t_0=0.,
        T=20.,
        grid_spacing=1./128,
        solutes=solutes,
        Lx=1.,  # 6.,
        Ly=1.,
        concentration_top=1.,
        concentration_bottom=2.,
        V_top=40.,
        V_bottom=0.,
        #
        grav_const=0.0,
        #
        density=[Re, Re],
        viscosity=[2.*lambda_D**2/kappa, 2.*lambda_D**2/kappa],
        permittivity=[2.*lambda_D**2, 2.*lambda_D**2],
    )
    return parameters


def constrained_domain(Lx, **namespace):
    return PeriodicBoundary(Lx)


def mesh(Lx=1., Ly=1., grid_spacing=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               concentration_top,
               concentration_bottom,
               V_top,
               V_bottom,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for solute in solutes:
                c_bottom = concentration_bottom
                if solute[0] == "c_m":
                    c_bottom = concentration_top
                c_init_expr = df.Expression(
                    "c_bottom+(c_top-c_bottom)*x[1]/Ly",
                    c_top=concentration_top,
                    c_bottom=c_bottom,
                    Ly=Ly,
                    degree=1)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression(
                "V_bottom+(V_top-V_bottom)*x[1]/Ly",
                V_top=V_top,
                V_bottom=V_bottom,
                Ly=Ly,
                degree=1)
            V_init = df.interpolate(
                V_init_expr,
                field_to_subspace["V"].collapse())
            w_init_field["V"] = V_init

    return w_init_field


def create_bcs(Lx, Ly,
               concentration_top, concentration_bottom,
               V_top, V_bottom,
               solutes,
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
        for solute in solutes:
            bcs["top"][solute[0]] = Fixed(concentration_top)
            if solute[0] == "c_p":
                bcs["bottom"][solute[0]] = Fixed(concentration_bottom)
        bcs["top"]["V"] = Fixed(V_top)
        bcs["bottom"]["V"] = Fixed(V_bottom)
    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def start_hook(w_, w_1, test_functions,
               solutes,
               permittivity,
               dx, ds, normal,
               dirichlet_bcs, neumann_bcs, boundary_to_mark,
               use_iterative_solvers,
               V_lagrange, **namespace):
    from solvers.stable_single import equilibrium_EC_PNP
    info_blue("Equilibrating with a non-linear solver.")
    equilibrium_EC_PNP(**vars())
    w_1["EC"].assign(w_["EC"])
    return dict()
