import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed
from common.functions import max_value, sign
import numpy as np
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


class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly) and on_boundary)


class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[1], 0.) and on_boundary)


def problem():
    info_cyan("Phase separation simulation.")
    
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
    parameters = dict(
        solver="TDLUES",
        folder="results_phase_separation",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=2e-4,
        t_0=0.,
        T=1.,
        grid_spacing=1./96.,
        interface_thickness=1e-2,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=1.,
        #
        V_top=2.,
        V_btm=0.,
        surface_tension=2.0,
        grav_const=0.,
        grav_dir=[0, -1.],
        #
        pf_mobility_coeff=1e-3,
        density=[0.5, 1.0],
        viscosity=[.2, .1],
        permittivity=[1., 10.],
        #
        use_iterative_solvers=True,
        use_pressure_stabilization=False
    )
    return parameters


def constrained_domain(Lx, **namespace):
    return PeriodicBoundary(Lx)


def mesh(Lx=1, Ly=5, grid_spacing=1./16, **namespace):
    m = df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                         int(Lx/(grid_spacing)),
                         int(Ly/(grid_spacing)))
    return m


def initialize(Lx, Ly,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, **namespace):

    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            phi_pert = 0.01
            w_init_field["phi"] = df.Function(
                field_to_subspace["phi"].collapse())
            w_init_field["phi"].vector().set_local(
                phi_pert*(
                    np.random.rand(
                        len(w_init_field["phi"].vector().get_local()))-0.5))

        # Electrochemistry
        if enable_EC:
            # c_init = df.Function(field_to_subspace[solutes[0][0]].collapse())
            # c_init.vector()[:] = 0.
            # for solute in solutes:
            #     w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression("x[1]/Ly", Ly=Ly, degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

    return w_init_field


def create_bcs(Ly, V_top, V_btm,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        top=[Top(Ly)],
        bottom=[Bottom()]
    )

    bcs = dict()
    bcs_pointwise = dict()
    bcs["top"] = dict()
    bcs["bottom"] = dict()

    noslip = Fixed((0., 0.))
    if enable_NS:
        bcs["top"]["u"] = noslip
        bcs["bottom"]["u"] = noslip
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs["top"]["V"] = Fixed(V_top)
        bcs["bottom"]["V"] = Fixed(V_btm)

    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_,
               enable_PF, dx, ds, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * max_value(0., func)
    # return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def end_hook(x_, enable_NS, dx, **namespace):
    u_norm = 0.
    if enable_NS:
        u_norm = df.assemble(df.dot(x_["u"], x_["u"])*dx)
    info("Velocity norm = {:e}".format(u_norm))
