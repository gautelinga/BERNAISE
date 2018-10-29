import dolfin as df
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Charged, Pressure
import numpy as np
__author__ = "Gaute Linga"


class Boundary(df.SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside_box(self, x):
        return bool(x[0] < self.Lx - df.DOLFIN_EPS and
                    x[0] > df.DOLFIN_EPS and
                    x[1] < self.Ly - df.DOLFIN_EPS and
                    x[1] > df.DOLFIN_EPS)


class Outer(Boundary):
    def inside(self, x, on_boundary):
        return bool(not self.inside_box(x) and on_boundary)

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0],0.0) and on_boundary)

class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx) and on_boundary )

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[1],0.0) and on_boundary)

class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], self.Ly) and on_boundary )

class Inner(Boundary):
    def inside(self, x, on_boundary):
        return bool(self.inside_box(x) and on_boundary)


def problem():
    info_cyan("Cleaning up a dolphin covered in oil spill.")
    # Define solutes
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 5e-3, 4., 1.],
               ["c_m", -1, 1e-4, 5e-3, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    sigma_e = -10.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_dolphin",
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
        grid_spacing=0.02,
        interface_thickness=0.020,
        solutes=solutes,
        base_elements=base_elements,
        Lx=2.,
        Ly=1.,
        R=0.35,
        surface_charge=sigma_e,
        concentration_init=5.,
        #
        surface_tension=1.,  # 2.45,
        grav_const=0.0,
        pressure_left=50.,
        pressure_right=0.,
        #
        pf_mobility_coeff=0.000010,
        density=[10., 10.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
    )
    return parameters


def constrained_domain(**namespace):
    return None


def mesh(Lx, Ly, grid_spacing, **namespace):
    # You have to run generate_mesh.py mesh=extended_polygon ... first
    mesh = load_mesh("meshes/flipper_dx{}_Lx{}_Ly{}.h5".format(
        grid_spacing, Lx, Ly))
    return mesh


def initialize(Lx, Ly, R,
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
                Lx*3./10., Ly/2., R, interface_thickness,
                field_to_subspace["phi"])
        if enable_EC:
            for solute in solutes:
                c_init = initial_phasefield(
                    Lx*3./10., Ly/2., R, interface_thickness,
                    field_to_subspace["phi"])
                # Only have ions in phase 2 (phi=-1)
                c_init.vector().set_local(
                    concentration_init*0.5*(
                        1.-c_init.vector().get_local()))
                w_init_field[solute[0]] = c_init

    return w_init_field


def create_bcs(Lx, Ly,
               solutes,
               concentration_init, surface_charge,
               pressure_left, pressure_right,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        inner=[Inner(Lx, Ly)],
        bottom=[Bottom()],
        top=[Top(Ly)],
        left=[Left()],
        right=[Right(Lx)],
        )

    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    noslip = Fixed((0., 0.))
    phi_inlet = Fixed(-1.0)
    p_inlet = Pressure(pressure_left)
    p_outlet = Pressure(pressure_right)

    if enable_NS:
        bcs["top"]["u"] = noslip
        bcs["bottom"]["u"] = noslip
        bcs["inner"]["u"] = noslip
        bcs["left"]["p"] = p_inlet
        bcs["right"]["p"] = p_outlet
        #bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")
        # bcs["outer"]["p"] = Pressure(0.)

    if enable_EC:
        for solute in solutes:
            bcs["left"][solute[0]] = Fixed(concentration_init)
            bcs["right"][solute[0]] = Fixed(concentration_init)
        bcs["top"]["V"] = Fixed(0.)
        bcs["bottom"]["V"] = Fixed(0.)
        bcs["left"]["V"] = Fixed(0.)
        bcs["right"]["V"] = Fixed(0.)
        bcs["inner"]["V"] = Charged(surface_charge)

    if enable_PF:  
        bcs["left"]["phi"] = phi_inlet
        #bcs["right"]["phi"] = phi_inlet

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, R, eps, function_space):
    expr_str = "-tanh((sqrt(pow(x[1]-y0, 2) + pow(x[0]-x0, 2))-R)/(sqrt(2)*eps))"
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, R=R,
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
