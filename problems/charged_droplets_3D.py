import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed
import numpy as np
__author__ = "Gaute Linga"


class Top(df.SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[2], self.Lz) and on_boundary)


class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[2], 0.) and on_boundary)


class Wall(df.SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(not df.near(x[2], self.Lz)
                    and not df.near(x[2], 0.)
                    and on_boundary)


def problem():
    info_cyan("Charged droplet in 3D")

    # Define solutes
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 0.0001, 0.1, 2., 0.],
               ["c_m", -1, 0.0001, 0.1, 2., 0.]]

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
        folder="results_charged_droplets_3D",
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
        grid_spacing=1./24.,
        interface_thickness=0.03,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=1.,
        Lz=2.,
        rad_init=0.2,
        #
        V_top=20.,
        V_btm=0.,
        surface_tension=2.0,  # 24.5,
        grav_const=0.98*0.,
        grav_dir=[0., 0., -1.],
        concentration_init=20.0,
        #
        pf_mobility_coeff=0.000010,
        density=[500., 50.],
        viscosity=[1., .5],
        permittivity=[1., 2.],
        #
        use_iterative_solvers=True
    )
    return parameters


def constrained_domain(Lx, **namespace):
    return None


def mesh(Lx=1., Ly=1., Lz=2., grid_spacing=1./16, **namespace):
    m = df.BoxMesh(df.Point(0., 0., 0.), df.Point(Lx, Ly, Lz),
                   int(Lx/(2*grid_spacing)),
                   int(Ly/(2*grid_spacing)),
                   int(Lz/(2*grid_spacing)))
    m = df.refine(m)
    return m


def initialize(Lx, Ly, Lz, V_top, V_btm, rad_init, concentration_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ Create the initial state.
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.

    Note: You only need to specify the initial states that are nonzero.
    """
    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = df.Function(
                field_to_subspace["phi"].collapse())
            for solute in solutes:
                zi = solute[1]
                phi_init = initial_phasefield(
                    Lx/2, Ly/2, Lz/2 + zi*Lx/2, rad_init, interface_thickness,
                    field_to_subspace["phi"].collapse())
                w_init_field["phi"].vector()[:] += \
                    phi_init.vector()
            w_init_field["phi"].vector()[:] -= float(len(solutes)-1)

        # Electrochemistry
        if enable_EC:
            for solute in solutes:
                zi = solute[1]
                c_init = initial_c_field(
                    concentration_init,
                    Lx/2, Ly/2, Lz/2 + zi*Lx/2, rad_init, interface_thickness,
                    field_to_subspace[solutes[0][0]].collapse())
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression(
                "V_btm+(V_top-V_btm)*x[2]/Lz",
                V_top=V_top, V_btm=V_btm, Lz=Lz, degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

    return w_init_field


def create_bcs(Lz, V_top, V_btm,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        wall=[Wall(Lz)],
        top=[Top(Lz)],
        bottom=[Bottom()]
    )

    bcs = dict()
    bcs_pointwise = dict()
    bcs["wall"] = dict()
    bcs["top"] = dict()
    bcs["bottom"] = dict()

    noslip = Fixed((0., 0., 0.))
    if enable_NS:
        bcs["top"]["u"] = noslip
        bcs["bottom"]["u"] = noslip
        bcs["wall"]["u"] = noslip
        bcs_pointwise["p"] = (0.,
                              "x[0] < DOLFIN_EPS && "
                              "x[1] < DOLFIN_EPS && "
                              "x[2] < DOLFIN_EPS")

    if enable_EC:
        bcs["top"]["V"] = Fixed(V_top)
        bcs["bottom"]["V"] = Fixed(V_btm)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, z0, rad, eps, function_space):
    expr_str = ("tanh((sqrt(pow(x[0]-x0,2)"
                "+pow(x[1]-y0,2)+pow(x[2]-z0,2))-rad)/(sqrt(2)*eps))")
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, z0=z0,
                                  rad=rad, eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def initial_c_field(c0, x0, y0, z0, rad, eps, function_space):
    expr_str = ("c0*pow(rad*sqrt(2*pi), -3)*exp("
                "-(pow(x[0]-x0,2)+pow(x[1]-y0,2)+pow(x[2]-z0,2))"
                "/(2*pow(rad, 2)))")
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, z0=z0,
                                  rad=rad/3., eps=eps,
                                  c0=c0*4.*np.pi*rad**3/3.,
                                  degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, enable_PF,
               dx, ds, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def end_hook(x_, enable_NS, dx, **namespace):
    u_norm = 0.
    if enable_NS:
        u_norm = df.assemble(df.dot(x_["u"], x_["u"])*dx)
    info("Velocity norm = {:e}".format(u_norm))
