import dolfin as df
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Pressure, Charged
import numpy as np
__author__ = "Gaute Linga"


class PeriodicBoundary(df.SubDomain):
    # Left boundary is target domain
    def __init__(self, Ly, grid_spacing):
        self.Ly = Ly
        self.grid_spacing = grid_spacing
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], -self.Ly/2) and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - self.Ly


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


class Obstacles(df.SubDomain):
    def __init__(self, Lx, centroids, rad, grid_spacing):
        self.Lx = Lx
        self.centroids = centroids
        self.rad = rad
        self.grid_spacing = grid_spacing
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        dx = self.centroids - np.outer(np.ones(len(self.centroids)), x)
        dist = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
        return bool(on_boundary
                    and x[0] > -self.Lx/2 + df.DOLFIN_EPS
                    and x[0] < self.Lx/2 - df.DOLFIN_EPS
                    and any(dist < self.rad + 0.1*self.grid_spacing))


def problem():
    info_cyan("Intrusion of one fluid into another in a porous medium.")

    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 1e-2, 4., 1.],
               ["c_m", -1, 1e-4, 1e-2, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 1./2.
    sigma_e = -1.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_porous",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.08,
        t_0=0.,
        T=20.,
        grid_spacing=0.05,
        interface_thickness=factor*0.060,
        solutes=solutes,
        base_elements=base_elements,
        Lx=4.,
        Ly=3.,
        rad_init=0.25,
        #
        surface_tension=24.5,
        grav_const=0.0,
        # inlet_velocity=0.1,
        pressure_left=1000.,
        pressure_right=0.,
        V_left=0.,
        V_right=0.,
        surface_charge=sigma_e,
        concentration_init=1.,
        front_position_init=0.1,  # percentage "filled" initially
        #
        pf_mobility_coeff=factor*0.000040,
        density=[1000., 1000.],
        viscosity=[100., 10.],
        permittivity=[1., 1.],
        #
        initial_interface="flat",
        #
        use_iterative_solvers=False,
        use_pressure_stabilization=False
    )
    return parameters


def constrained_domain(Ly, grid_spacing, **namespace):
    return PeriodicBoundary(Ly, grid_spacing)


def mesh(Lx=4., Ly=3., grid_spacing=0.04, **namespace):
    return load_mesh("meshes/periodic_porous_dx" + str(grid_spacing) + ".h5")


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,  # inlet_velocity,
               front_position_init, concentration_init,
               pressure_left, pressure_right,
               enable_NS, enable_PF, enable_EC, initial_interface,
               **namespace):
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
        # if enable_NS:
        #     try:
        #         subspace = field_to_subspace["u"].collapse()
        #     except:
        #         subspace = field_to_subspace["u"]
        #     w_init_field["u"] = initial_velocity(0.,
        #                                          subspace)
        # Phase field
        x_0 = -Lx/2 + Lx*front_position_init

        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                x_0, Ly/2, rad_init, interface_thickness,
                field_to_subspace["phi"].collapse(), shape=initial_interface)

        if enable_EC:
            for solute in solutes:
                c_init = initial_phasefield(
                    x_0, Ly/2, rad_init, interface_thickness,
                    field_to_subspace["phi"].collapse(),
                    shape=initial_interface)
                # Only have ions in phase 1 (phi=1)
                c_init.vector()[:] = concentration_init*0.5*(
                    1.-c_init.vector().array())
                w_init_field[solute[0]] = c_init
    return w_init_field


def create_bcs(Lx, Ly, grid_spacing,  # inlet_velocity,
               concentration_init, solutes,
               surface_charge, V_left, V_right,
               pressure_left, pressure_right,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """

    data = np.loadtxt("meshes/periodic_porous_dx" + str(grid_spacing) + ".dat")
    centroids = data[:, :2]
    rad = data[:, 2]

    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(Lx)],
        obstacles=[Obstacles(Lx, centroids, rad, grid_spacing)]
    )

    # Allocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    # u_inlet = Fixed((inlet_velocity, 0.))
    noslip = Fixed((0., 0.))

    p_inlet = Pressure(pressure_left)
    p_outlet = Pressure(pressure_right)
    phi_inlet = Fixed(-1.0)
    phi_outlet = Fixed(1.0)

    if enable_NS:
        # bcs["left"]["u"] = u_inlet
        bcs["obstacles"]["u"] = noslip
        bcs["right"]["p"] = p_outlet
        bcs["left"]["p"] = p_inlet
        # bcs_pointwise["p"] = (0., "x[0] < -{Lx}/2+DOLFIN_EPS && x[1] > {Ly}/2-DOLFIN_EPS".format(Lx=Lx, Ly=Ly))

    if enable_PF:
        bcs["left"]["phi"] = phi_inlet
        bcs["right"]["phi"] = phi_outlet

    if enable_EC:
        for solute in solutes:
            bcs["left"][solute[0]] = Fixed(concentration_init)
            # bcs["right"][solute[0]] = Fixed(0.)
        bcs["left"]["V"] = Fixed(V_left)
        bcs["right"]["V"] = Fixed(V_right)
        bcs["obstacles"]["V"] = Charged(surface_charge)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, rad, eps, function_space, shape="flat"):
    if shape == "flat":
        expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    elif shape == "sine":
        expr_str = "tanh((x[0]-x0-eps*sin(2*x[1]*pi))/(sqrt(2)*eps))"
    elif shape == "circle":
        expr_str = ("tanh(sqrt(2)*(sqrt(pow(x[0]-x0,2)" +
                    "+pow(x[1]-y0,2))-rad)/eps)")
    else:
        info_red("Unrecognized shape: " + shape)
        exit()
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad,
                                  eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def initial_velocity(inlet_velocity, function_space):
    #u_init_expr = df.Constant((inlet_velocity, 0.))
    u_init_expr = df.Constant((0., 0.))
    u_init = df.interpolate(u_init_expr, function_space)
    return u_init


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))
    if stats_intv and tstep % stats_intv == 0:
        # GL: Seems like a rather awkward way of doing this,
        # but any other way seems to fuck up the simulation.
        # Anyhow, a better idea could be to move some of this to a post-processing stage.
        # GL: Move into common/utilities at a certain point.
        subproblem_name, subproblem_i = field_to_subproblem["phi"]
        phi = w_[subproblem_name].split(deepcopy=True)[subproblem_i]
        bubble = 0.5*(1.-df.sign(phi))
        mass = df.assemble(bubble*df.dx)
        massy = df.assemble(
            bubble*df.Expression("x[1]", degree=1)*df.dx)
        if mpi_is_root():
            with file(statsfile, "a") as outfile:
                outfile.write("{} {} {} \n".format(t, mass, massy))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    # func = 1.-phi**2
    # return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func
    return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
