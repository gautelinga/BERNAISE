import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
__author__ = "Gaute Linga"


class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.0) and on_boundary)


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
    info_cyan("Flow in a cell with single-phase electrohydrodynamics.")

    solutes = [
        ["c_p",  1, 0.01, 0.01, 0., 0.],
        ["c_m", -1, 0.01, 0.01, 0., 0.]  # ,
        # ["c_n", 0, 0.01, 0.01, 0., 0.]
    ]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_single_cell",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.02,
        t_0=0.,
        T=10.0,
        grid_spacing=1./64,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=2.,
        concentration_init=3.,
        rad=0.25,
        surface_charge=1.,
        #
        density=[1., 1.],
        viscosity=[0.08, 0.08],
        permittivity=[0.5, 0.5],
        EC_scheme="NL2",
        use_iterative_solvers=True,
        grav_const=0.,
        c_cutoff=0.1,
        density_per_concentration=[0.02, 0.02, 0.04],
        viscosity_per_concentration=[0.001, 0.001, 0.002],
        reactions=[[1.0, [-1, -1, 1]]]
    )
    return parameters


def mesh(Lx=1., Ly=2., grid_spacing=1./16., **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               concentration_init, rad,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for solute in solutes:
                c_init_expr = df.Expression(
                    "c0*1./(2*DOLFIN_PI*rad*rad)*exp("
                    "- (pow(x[0]-0.5*Lx+0.5*zi*rad, 2)"
                    "+ pow(x[1]-0.5*Ly+2.0*zi*rad, 2))/(2*rad*rad))",
                    Lx=Lx,
                    Ly=Ly,
                    c0=concentration_init,
                    rad=rad,
                    zi=solute[1],
                    degree=2)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
    return w_init_field


def create_bcs(Lx, Ly, mesh,
               surface_charge,
               enable_NS, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(0)],
        bottom=[Bottom(0)],
        top=[Top(Ly)]
    )

    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    noslip = Fixed((0., 0.))

    if enable_NS:
        bcs["left"]["u"] = noslip
        bcs["right"]["u"] = noslip
        bcs["top"]["u"] = noslip
        bcs["bottom"]["u"] = noslip
        bcs_pointwise["p"] = (
            0.,
            "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs["top"]["V"] = Fixed(0.)
        bcs["bottom"]["V"] = Charged(surface_charge)

    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
