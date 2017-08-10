import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed
__author__ = "Gaute Linga"


class Wall(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(x[0] >= df.DOLFIN_EPS and
                    x[0] <= self.Lx-df.DOLFIN_EPS and on_boundary)


class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < df.DOLFIN_EPS and on_boundary)


class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(x[0] > self.Lx-df.DOLFIN_EPS and on_boundary)


def problem():
    info_cyan("Charged droplet in an electric field.")

    # Define solutes
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    solutes = [["c_p",  1, 1e-5, 1e-3, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_charged_droplet",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.08,  # 0.02,
        t_0=0.,
        T=8.,
        grid_spacing=1./32,  # 1./64,
        interface_thickness=0.03,  # 0.02,
        solutes=solutes,
        base_elements=base_elements,
        Lx=2.,
        Ly=1.,
        rad_init=0.25,
        #
        V_left=10.,
        V_right=0.,
        surface_tension=5.,  # 24.5,
        grav_const=0.,
        concentration_init=10.,
        #
        pf_mobility_coeff=0.00002,  # 0.000010,
        density=[200., 100.],
        viscosity=[10., 1.],
        permittivity=[1., 1.],
        #
        use_iterative_solvers=False,
        use_pressure_stabilization=False
    )
    return parameters


def mesh(Lx=1, Ly=5, grid_spacing=1./16, **namespace):
    m = df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                         int(Lx/(2*grid_spacing)),
                         int(Ly/(2*grid_spacing)))
    m = df.refine(m)
    return m


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes,
               concentration_init,
               restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ Create the initial state. """
    x0 = [Lx/4]
    y0 = [Ly/2]
    rad0 = [rad_init]
    c0 = [concentration_init]

    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_pf(
                x0, y0, rad0, interface_thickness,
                field_to_subspace["phi"].collapse())

        # Electrochemistry
        if enable_EC:
            for x, y, rad, ci, solute in zip(x0, y0, rad0, c0, solutes):
                c_init = initial_c(x, y, rad/3., ci, interface_thickness,
                                   field_to_subspace[solute[0]].collapse())
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression("0.", degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

    return w_init_field


def create_bcs(field_to_subspace, Lx, Ly, solutes,
               V_left, V_right,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundary conditions are defined in terms of field. """

    boundaries = dict(
        wall=[Wall(Lx)],
        left=[Left()],
        right=[Right(Lx)]
    )

    noslip = Fixed((0., 0.))

    bcs = dict()
    bcs_pointwise = dict()

    bcs["wall"] = dict()
    bcs["left"] = dict()
    bcs["right"] = dict()

    if enable_NS:
        bcs["wall"]["u"] = noslip
        bcs["left"]["u"] = noslip
        bcs["right"]["u"] = noslip
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_EC:
        bcs["left"]["V"] = Fixed(V_left)
        bcs["right"]["V"] = Fixed(V_right)

    return boundaries, bcs, bcs_pointwise


def initial_pf(x0, y0, rad0, eps, function_space):
    expr_str = "1."
    for x, y, rad in zip(x0, y0, rad0):
        expr_str += ("-(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
                     "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
                        x=x, y=y, rad=rad, eps=eps)
    phi_init_expr = df.Expression(expr_str, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def initial_c(x, y, rad, c_init, eps, function_space):
    # expr_str = ("{c_init}*0.5*(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
    #             "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
    #                 x=x, y=y, rad=rad, eps=eps, c_init=c_init)
    expr_str = ("c_init*1./(2*pi*pow(sigma, 2)) * "
                "exp(- 0.5*pow((x[0]-x0)/sigma, 2)"
                " - 0.5*pow((x[1]-y0)/sigma, 2))")
    c_init_expr = df.Expression(expr_str, x0=x, y0=y, sigma=rad,
                                c_init=c_init, degree=2)
    return df.interpolate(c_init_expr, function_space)


def tstep_hook(t, tstep, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func
    # return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
