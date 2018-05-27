import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed
__author__ = "Gaute Linga"

info_cyan("Welcome to the simple problem in 3D!")


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
        solver="basic",
        folder="results_simple_3D",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.005,
        t_0=0.,
        T=20.,
        grid_spacing=1./64.,
        interface_thickness=0.02,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=1.,
        Lz=1.,
        rad_init=0.25,
        #
        V_top=1.,
        V_btm=0.,
        surface_tension=5.0,  # 24.5,
        grav_const=0.98*1.,
        grav_dir=[0., 0., -1.],
        concentration_init=4.5,
        #
        pf_mobility_coeff=0.000010,
        density=[1000., 100.],
        viscosity=[10., 1.],
        permittivity=[1., 10.],
        #
        use_iterative_solvers=True,
        use_pressure_stabilization=False
    )
    return parameters


def constrained_domain(Lx, **namespace):
    return None


def mesh(Lx=1., Ly=1., Lz=2., grid_spacing=1./16, **namespace):
    return df.BoxMesh(df.Point(0., 0., 0.), df.Point(Lx, Ly, Lz),
                      int(Lx/grid_spacing), int(Ly/grid_spacing),
                      int(Lz/grid_spacing))


def initialize(Lx, Ly, Lz, rad_init, concentration_init,
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
            w_init_field["phi"] = initial_phasefield(
                Lx/2, Ly/2, Lx/2, rad_init, interface_thickness,
                field_to_subspace["phi"].collapse())

        # Electrochemistry
        if enable_EC:
            # c_init = df.Function(field_to_subspace[solutes[0][0]].collapse())
            c_init = initial_phasefield(
                Lx/2, Ly/2, Lx/2, rad_init, interface_thickness,
                field_to_subspace[solutes[0][0]].collapse())
            c_init.vector()[:] = 0.5*(
                1.-c_init.vector().get_local()) * concentration_init
            for solute in solutes:
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression("x[2]/Lz", Lz=Lz, degree=1)
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


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, enable_PF,
               dx, ds, **namespace):
    info_blue("Timestep = {}".format(tstep))

    if False and stats_intv and tstep % stats_intv == 0 and enable_PF:
        # GL: Seems like a rather awkward way of doing this, but any
        # other way seems to fuck up the simulation.  Anyhow, a better
        # idea could be to move some of this to a post-processing
        # stage.
        subproblem_name, subproblem_i = field_to_subproblem["phi"]
        Q = w_[subproblem_name].split(deepcopy=True)[subproblem_i]
        bubble = df.interpolate(Q, field_to_subspace["phi"].collapse())
        bubble = 0.5*(1.-df.sign(bubble))
        mass = df.assemble(bubble*dx)
        massy = df.assemble(
            bubble*df.Expression("x[1]", degree=1)*dx)
        if mpi_is_root():
            with file(statsfile, "a") as outfile:
                outfile.write("{} {} {} \n".format(t, mass, massy))


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
