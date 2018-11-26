import dolfin as df
import os
from . import *
from common.io import mpi_is_root, load_mesh, mpi_comm, mpi_is_root
from common.bcs import Fixed, Charged
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
        y[2] = x[2]


class Bottom(df.SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(x[1] < df.DOLFIN_EPS
                    and not x[2] < df.DOLFIN_EPS
                    and not x[2] > self.Lz - df.DOLFIN_EPS
                    and on_boundary)


class Top(df.SubDomain):
    def __init__(self, Ly, Lz):
        self.Ly = Ly
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly)
                    and not x[2] < df.DOLFIN_EPS
                    and not x[2] > self.Lz - df.DOLFIN_EPS
                    and on_boundary)


class Wall(df.SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool((x[2] < df.DOLFIN_EPS
                     or x[2] > self.Lz - df.DOLFIN_EPS)
                    and on_boundary)


def problem():
    info_cyan("Desnotting Snoevsen (extruded).")

    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-3, 1e-1, 4., 1.],
               ["c_m", -1, 1e-3, 1e-1, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 1, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 1./4
    sigma_e = -10.0  # 0.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="TDLUES",
        folder="results_snoevsen_extruded",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.05,
        t_0=0.,
        T=100.,
        res=60,
        interface_thickness=factor*0.1,
        solutes=solutes,
        base_elements=base_elements,
        Lx=3.,
        Ly=1.,
        Lz=2.5,
        R=0.3,
        surface_potential=False,
        surface_charge=sigma_e,
        concentration_init=2.,
        velocity_top=0.5,
        #
        surface_tension=1.0,
        grav_const=0.0,
        grav_dir=[1., 0., 0.],
        #
        pf_mobility_coeff=factor*0.000010,
        density=[10., 10.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
        #
        use_iterative_solvers=True
    )
    return parameters


def constrained_domain(Lx, Lz, **namespace):
    return PeriodicBoundary(Lx)


def mesh(Lx, Ly, res, **namespace):
    mesh = load_mesh("meshes/snoevsen_res{}_extruded.h5".format(res))
    return mesh


def initialize(Lx, Ly, Lz, R,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               concentration_init,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                Lx/2, 0., Lz/2, 2*R, interface_thickness,
                field_to_subspace["phi"])
        if enable_EC:
            for solute in solutes:
                concentration_init_loc = concentration_init/abs(solute[1])
                c_init = initial_phasefield(
                    Lx/2, 0., Lz/2, 2*R, interface_thickness,
                    field_to_subspace["phi"])
                # Only have ions in phase 2 (phi=-1)
                c_init.vector()[:] = concentration_init_loc*0.5*(
                    1.-c_init.vector().get_local())
                w_init_field[solute[0]] = c_init

    return w_init_field


def create_bcs(Lx, Ly, Lz,
               velocity_top, solutes,
               concentration_init, surface_charge, surface_potential,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        top=[Top(Ly, Lz)],
        bottom=[Bottom(Lz)],
        wall=[Wall(Lz)]
    )

    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    u_top = Fixed((velocity_top, 0., 0.))
    noslip = Fixed((0., 0., 0.))

    if enable_NS:
        bcs["top"]["u"] = u_top
        bcs["bottom"]["u"] = noslip
        bcs["wall"]["u"] = noslip
        bcs_pointwise["p"] = (
            0.,
            "x[0] < DOLFIN_EPS && "
            "x[1] > {Ly}-DOLFIN_EPS && "
            "x[2] < DOLFIN_EPS".format(Ly=Ly))

    if enable_EC:
        for solute in solutes:
            bcs["top"][solute[0]] = Fixed(concentration_init/abs(solute[1]))
        bcs["top"]["V"] = Fixed(0.)
        if surface_potential:
            bcs["bottom"]["V"] = Fixed(surface_charge)
        else:
            charge = df.Expression("q*(1.0+x[1]/Ly)",
                                   q=surface_charge, Ly=Ly, degree=1)
            bcs["bottom"]["V"] = Charged(charge)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, z0, R, eps, function_space):
    expr_str = ("-tanh(max("
                "sqrt(pow(x[1]-y0+R, 2) + pow(x[2]-z0, 2)) - R, "
                "sqrt(pow(x[0]-x0, 2))-R)"
                "/(sqrt(2)*eps))")
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, z0=z0, R=R,
                                  eps=eps, degree=2)
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


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
