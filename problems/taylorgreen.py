import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
__author__ = "Gaute Linga"


class PeriodicBoundary(df.SubDomain):
    # Left boundary is target domain
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool((df.near(x[0], 0.) or df.near(x[1], 0.)) and
                    (not ((df.near(x[0], 0.) and df.near(x[1], self.Ly)) or
                          (df.near(x[0], self.Lx) and df.near(x[1], 0.))))
                    and on_boundary)

    def map(self, x, y):
        if df.near(x[0], self.Lx) and df.near(x[1], self.Ly):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - self.Ly


def problem():
    info_cyan("Taylor-Green vortex flow with electrohydrodynamics.")

    solutes = [["c_p",  1, 2., 2., 0., 0.],
               ["c_m", -1, 2., 2., 0., 0.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_taylorgreen",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.002,
        t_0=0.,
        T=1.,
        N=32,
        solutes=solutes,
        base_elements=base_elements,
        Lx=2.*np.pi,
        Ly=2.*np.pi,
        concentration_init=1.,
        concentration_init_dev=0.5,
        #
        density=[1., 1.],
        viscosity=[1., 1.],
        permittivity=[2., 2.],
        EC_scheme="L1",
        use_iterative_solvers=True,
        grav_const=0.,
        c_cutoff=0.1
    )
    return parameters


def constrained_domain(Lx, Ly, **namespace):
    return PeriodicBoundary(Lx, Ly)


def mesh(Lx=2.*np.pi, Ly=2.*np.pi, N=16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly), N, N)


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               concentration_init,
               concentration_init_dev,
               permittivity,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    veps = permittivity[0]
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for solute in solutes:
                c_init_expr = df.Expression(
                    "c0*(1 + zi*chi*cos(x[0])*cos(x[1]))",
                    c0=concentration_init,
                    zi=solute[1],
                    chi=concentration_init_dev,
                    degree=2)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
        if enable_NS:
            u_init_expr = df.Expression(
                ("cos(x[0])*sin(x[1])", "-sin(x[0])*cos(x[1])"),
                degree=2)
            w_init_field["u"] = df.interpolate(
                u_init_expr, field_to_subspace["u"].collapse())
            p_init_expr = df.Expression(
                "(-0.25+factor)*(cos(2*x[0]) + cos(2*x[1]))"
                "+ factor*cos(2*x[0])*cos(2*x[1])",
                factor=concentration_init**2*concentration_init_dev**2/veps,
                degree=2)
            w_init_field["p"] = df.interpolate(
                p_init_expr, field_to_subspace["p"].collapse())
    return w_init_field


def create_bcs(Lx, Ly,
               enable_NS, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict()
    bcs = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    if enable_NS:
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs_pointwise["V"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")
    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def rhs_source(solutes):
    q = []
    for solute in solutes:
        q.append(df.Expression("0.", degree=1))
    return q
