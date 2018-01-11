import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged, Pressure
__author__ = "Gaute Linga"


class ChannelSubDomain(df.SubDomain):
    def __init__(self, Lx, Ly, Lx_inner):
        self.Lx = Lx
        self.Ly = Ly
        self.Lx_inner = Lx_inner
        df.SubDomain.__init__(self)

    def on_wall(self, x, on_boundary):
        return bool(df.near(x[1], 0.) or df.near(x[1], self.Ly))

    def within_inner(self, x, on_boundary):
        return bool((x[0] > self.Lx/2 - self.Lx_inner/2) and
                    (x[0] < self.Lx/2 + self.Lx_inner/2))


class Left(ChannelSubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.0) and
                    not self.on_wall(x, on_boundary) and
                    on_boundary)


class Right(ChannelSubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx) and
                    not self.on_wall(x, on_boundary) and
                    on_boundary)


class OuterWall(ChannelSubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and
                    self.on_wall(x, on_boundary) and
                    not self.within_inner(x, on_boundary))


class InnerWall(ChannelSubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and
                    self.on_wall(x, on_boundary) and
                    self.within_inner(x, on_boundary))


class Wall(ChannelSubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and self.on_wall(x, on_boundary))


def problem():
    info_cyan("Flow in a channel with single-phase electrohydrodynamics.")

    Re = 0.001
    Pe = 1./2.189
    lambda_D = 1.5  # 1.5
    c_inf = 1.0
    sigma_e = -6.  #-6.
    f = 0.02

    solutes = [["c_p",  1, 1./Pe, 1./Pe, 0., 0.],
               ["c_m", -1, 1./Pe, 1./Pe, 0., 0.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_single_channel",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.01,
        t_0=0.,
        T=100*1.0,
        grid_spacing=1./4,
        solutes=solutes,
        base_elements=base_elements,
        Lx=50.,  # 40.,
        Lx_inner=20.,
        Ly=6.,
        concentration_init=c_inf,
        surface_charge=sigma_e,
        #
        density=[Re, Re],
        viscosity=[1., 1.],
        permittivity=[2.*lambda_D**2, 2.*lambda_D**2],
        EC_scheme="NL2",
        use_iterative_solvers=True,
        grav_const=f/Re,
        grav_dir=[1., 0],
        c_cutoff=0.1
    )
    return parameters


def mesh(Lx=1., Ly=2., grid_spacing=1./16., **namespace):
    m = df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))
    x = m.coordinates()

    beta = 1.

    x[:, 1] = beta*x[:, 1] + (1.-beta)*0.5*Ly*(1 + np.arctan(
        np.pi*((x[:, 1]-0.5*Ly) / (0.5*Ly)))/np.arctan(np.pi))
    return m


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               concentration_init,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for solute in solutes:
                c_init_expr = df.Expression("c0",
                                            c0=concentration_init, degree=2)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
    return w_init_field


def create_bcs(Lx, Ly, Lx_inner,
               solutes,
               concentration_init,
               mesh,
               surface_charge,
               enable_NS, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx, Ly, Lx_inner)],
        left=[Left(Lx, Ly, Lx_inner)],
        #inner_wall=[InnerWall(Lx, Ly, Lx_inner)],
        #outer_wall=[OuterWall(Lx, Ly, Lx_inner)]
        wall=[Wall(Lx, Ly, Lx_inner)]
    )

    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    noslip = Fixed((0., 0.))

    if enable_NS:
        bcs["left"]["p"] = Pressure(0.)
        bcs["right"]["p"] = Pressure(0.)
        #bcs["inner_wall"]["u"] = noslip
        #bcs["outer_wall"]["u"] = noslip
        bcs["wall"]["u"] = noslip

    if enable_EC:
        bcs["left"]["V"] = Fixed(0.)
        bcs["right"]["V"] = Charged(0.)
        #bcs["right"]["V"] = Fixed(0.)
        #bcs["outer_wall"]["V"] = Charged(0.)
        #bcs["inner_wall"]["V"] = Charged(surface_charge)
        bcs["wall"]["V"] = Charged(df.Expression(
            "0.5*sigma_e*(tanh((x[0]-0.5*(Lx-Lx_inner))/delta)-"
            "tanh((x[0]-0.5*(Lx+Lx_inner))/delta))",
            sigma_e=surface_charge, delta=2., Lx=Lx, Lx_inner=Lx_inner,
            degree=1))
        for solute in solutes:
            bcs["left"][solute[0]] = Fixed(concentration_init)
            bcs["right"][solute[0]] = Fixed(concentration_init)

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
    from solvers.stable_single import equilibrium_EC
    info_blue("Equilibrating with a non-linear solver.")
    equilibrium_EC(**vars())
    w_1["EC"].assign(w_["EC"])
    return dict()
