import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
__author__ = "Gaute Linga"


class Left(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], -self.Lx/2) and on_boundary)


class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx/2) and on_boundary)


class Bottom(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], -self.Ly/2) and on_boundary)


class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly/2) and on_boundary)


def problem():
    info_cyan("Flow and reaction in a cell with single-phase electrohydrodynamics.")

    solutes = [["c_p",  1, 0.01, 0.01, -np.log(3.), -np.log(1.)],
               ["c_m", -1, 0.01, 0.01, -np.log(3.), -np.log(1.)],
               ["c_n", 0, 0.01, 0.01, -np.log(1.), -np.log(1.)]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single_fracstep",
        folder="results_single_reaction",
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
        T=10.0,
        grid_spacing=1./64,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=1.,
        concentration_init=10.,
        rad=0.15,
        surface_charge=2.0,
        #
        density=[1., 1.],
        viscosity=[0.1, 0.1],
        permittivity=[0.1, 0.1],
        EC_scheme="NL2",
        use_iterative_solvers=True,
        grav_const=0.,
        c_cutoff=0.1,
        reactions=[[10., [-1, -1, 1]]],
        density_per_concentration=[0.1, 0.1, 0.2],
        viscosity_per_concentration=[0.02, 0.02, 0.04]
    )
    return parameters


def mesh(Lx=1., Ly=1., grid_spacing=1./16., **namespace):
    return df.RectangleMesh(df.Point(-Lx/2, -Ly/2), df.Point(Lx/2, Ly/2),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               concentration_init, rad,
               enable_NS, enable_EC,
               dx,
               surface_charge,
               permittivity,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for solute in ["c_p", "c_m"]:
                w_init_field[solute] = df.interpolate(
                    df.Constant(1e-4), field_to_subspace[solute].collapse())
            c_init = df.interpolate(
                df.Expression("1./(2*DOLFIN_PI*rad*rad)*exp("
                              "- (pow(x[0], 2) + pow(x[1], 2))/(2*rad*rad))",
                              Lx=Lx, Ly=Ly, rad=rad, degree=2),
                field_to_subspace["c_n"].collapse())
            C_tot = df.assemble(c_init*dx)
            c_init.vector()[:] *= concentration_init*Lx*Ly/C_tot
            w_init_field["c_n"] = c_init

            V_0 = -surface_charge*Ly/permittivity[0]
            w_init_field["V"] = df.interpolate(
                df.Expression("V_0*(x[1]/Ly-0.5)", Ly=Ly, V_0=V_0, degree=1),
                field_to_subspace["V"].collapse())
    return w_init_field


def create_bcs(Lx, Ly, mesh,
               surface_charge,
               enable_NS, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(Lx)],
        bottom=[Bottom(Ly)],
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
            "x[0] < DOLFIN_EPS-{Lx}/2 && "
            "x[1] < DOLFIN_EPS-{Ly}/2".format(Lx=Lx, Ly=Ly))

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
