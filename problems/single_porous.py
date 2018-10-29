import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Charged, Pressure, Open
from .porous import Obstacles
from mpi4py import MPI
__author__ = "Gaute Linga"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


class RectBoun(df.SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        df.SubDomain.__init__(self)


class PeriodicBoundary(RectBoun):
    # Left boundary is target domain
    def inside(self, x, on_boundary):
        return bool(df.near(x[1], -self.Ly/2) and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - self.Ly


class Left(RectBoun):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], -self.Lx/2) and on_boundary)


class Right(RectBoun):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx/2) and on_boundary)


def problem():
    info_cyan("Porous media flow with electrohydrodynamics.")
    # GL 2018-01-11: Updating to realistic ones.

    Re = 0.1
    Pe = 1./2.189
    lambda_D = 1.5
    c_inf = 1.0
    sigma_e = -5.0
    f = 0.02

    solutes = [["c_p",  1, 1./Pe, 1./Pe, 0., 0.],
               ["c_m", -1, 1./Pe, 1./Pe, 0., 0.]]

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="stable_single",
        folder="results_single_porous",
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
        solutes=solutes,
        Lx=60.,
        Ly=30.,
        rad=3.0,
        num_obstacles=8,  # 100,
        grid_spacing=1.,  # 0.25,  # 1.,  # 1./4,
        #
        density=[Re, Re],
        viscosity=[2.*lambda_D**2, 2.*lambda_D**2],
        permittivity=[2.*lambda_D**2, 2.*lambda_D**2],
        surface_charge=sigma_e,
        concentration_init=c_inf,
        V_left=0.,
        #V_right=0.,
        #
        EC_scheme="NL2",
        use_iterative_solvers=True,
        V_lagrange=False,
        p_lagrange=False,
        #
        grav_const=2*lambda_D**2*f/Re,
        grav_dir=[1., 0.],
        c_cutoff=0.1
    )
    return parameters


def constrained_domain(Lx, Ly, **namespace):
    return PeriodicBoundary(Lx, Ly)


def mesh(Lx=8., Ly=8., rad=0.25, num_obstacles=100,
         grid_spacing=0.05, **namespace):
    mesh = load_mesh(
        "meshes/periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}.h5".format(
            Lx, Ly, rad, num_obstacles, grid_spacing))
    return mesh


def initialize(Lx, Ly,
               solutes, restart_folder,
               field_to_subspace,
               num_obstacles, rad,
               surface_charge,
               concentration_init,
               enable_NS, enable_EC,
               **namespace):
    """ Create the initial state. """
    # Enforcing the compatibility condition.
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for i, solute in enumerate(solutes):
                c_init_expr = df.Expression(
                    "c0",
                    c0=concentration_init,
                    degree=2)
                c_init = df.interpolate(
                    c_init_expr,
                    field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
    return w_init_field


def create_bcs(Lx, Ly,
               grid_spacing, rad, num_obstacles,
               surface_charge, solutes, enable_NS, enable_EC,
               p_lagrange, V_lagrange,
               concentration_init, V_left,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    data = np.loadtxt(
        "meshes/periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}.dat".format(
            Lx, Ly, rad, num_obstacles, grid_spacing))
    centroids = data[:, :2]
    rad = data[:, 2]

    boundaries = dict(
        left=[Left(Lx, Ly)],
        right=[Right(Lx, Ly)],
        obstacles=[Obstacles(Lx, centroids, rad, grid_spacing)]
    )

    # Allocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    noslip = Fixed((0., 0.))

    if enable_NS:
        bcs["left"]["p"] = Pressure(0.)
        bcs["right"]["p"] = Pressure(0.)
        bcs["obstacles"]["u"] = noslip

    if enable_EC:
        for solute in solutes:
            bcs["left"][solute[0]] = Fixed(concentration_init)
            bcs["right"][solute[0]] = Fixed(concentration_init)
        bcs["left"]["V"] = Fixed(V_left)
        bcs["right"]["V"] = Charged(0.)
        # bcs["right"]["V"] = Fixed(V_right)
        bcs["obstacles"]["V"] = Charged(surface_charge)

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
