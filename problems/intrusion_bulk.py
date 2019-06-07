import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Pressure
# from ufl import max_value
__author__ = "Asger Bolet; Gaute Linga"


class PeriodicBoundary(df.SubDomain):
    # Left boundary is target domain
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[1], 0.) and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - self.Ly


class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0.0) and on_boundary)


class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], self.Lx) and on_boundary)


def problem():
    info_cyan("Bulk intrusion of a front of one fluid into another.")
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

    factor = 1.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_intrusion_bulk",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.08,
        t_0=0.,
        T=20.,
        grid_spacing=factor*1./16,
        interface_thickness=factor*0.030,
        solutes=solutes,
        base_elements=base_elements,
        Lx=5.,
        Ly=2.,
        rad_init=0.25,
        front_position_init=2.4,
        concentration_left=1.,
        #
        surface_tension=2.45,  # 24.5,
        grav_const=0.0,
        inlet_velocity=0.1,
        comoving_velocity=[0.0, 0.0],
        V_0=10.,
        friction_coeff=100.,
        #
        pf_mobility_coeff=factor*0.000040,
        density=[1., 1.],
        viscosity=[1000., .1],
        permittivity=[1., 10.],
        #
        initial_interface="random",
        #
        use_iterative_solvers=True,
        use_pressure_stabilization=False,
    )
    return parameters

def constrained_domain(Ly, **namespace):
    return PeriodicBoundary(Ly)

def mesh(Lx=1, Ly=5, grid_spacing=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/grid_spacing), int(Ly/grid_spacing))

def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               inlet_velocity, front_position_init, concentration_left,
               enable_NS, enable_PF, enable_EC, initial_interface, **namespace):
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
        if enable_NS:
            try:
                subspace = field_to_subspace["u"].collapse()
            except:
                subspace = field_to_subspace["u"]
            w_init_field["u"] = initial_velocity(inlet_velocity,
                                                 subspace)
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                front_position_init, Lx/2, rad_init, interface_thickness,
                field_to_subspace["phi"].collapse(), shape=initial_interface)

        if enable_EC:
            for solute in solutes:
                w_init_field[solute[0]] = initial_phasefield(
                    front_position_init, Lx/2, rad_init, interface_thickness,
                    field_to_subspace[solute[0]].collapse(),
                    shape=initial_interface)
                w_init_field[solute[0]].vector()[:] = \
                    concentration_left*(
                        - w_init_field[solute[0]].vector()[:]
                        + 1.0)/2.0

    return w_init_field


def create_bcs(Lx, Ly, inlet_velocity, V_0, solutes,
               concentration_left,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(0)]
    )

    # Alocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    bcs["left"] = dict()
    bcs["right"] = dict()

    inlet_velocity = Fixed((inlet_velocity, 0.))
    #pressurein_out = Pressure(0.0)
    phi_inlet = Fixed(-1.0) 
    phi_outlet = Fixed(1.0) 
    V_left = Fixed(V_0)
    V_right = Fixed(0.)

    if enable_NS:
        bcs["left"]["u"] = inlet_velocity
        bcs["right"]["u"] = inlet_velocity
        # bcs["left"]["p"] = pressurein_out
        bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    if enable_PF:
        bcs["left"]["phi"] = phi_inlet
        bcs["right"]["phi"] = phi_outlet

    if enable_EC:
        bcs["left"]["V"] = V_left
        bcs["right"]["V"] = V_right
        for solute in solutes:
            bcs["left"][solute[0]] = Fixed(concentration_left)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, rad, eps, function_space, shape="flat"):
    if shape == "flat":
        expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    elif shape == "sine":
        expr_str = "tanh((x[0]-x0-eps*sin(4*x[1]*pi))/(sqrt(2)*eps))"
    elif shape == "random":
        expr_str = ("tanh((x[0]-x0-"
                    "0.01*sin(1*x[1]*pi+12)+"
                    "0.01*sin(2*x[1]*pi+1235)+"
                    "0.01*sin(3*x[1]*pi+1233)+"
                    "0.01*sin(4*x[1]*pi+623)+"
                    "0.01*sin(5*x[1]*pi+234)+"
                    "0.01*sin(6*x[1]*pi+23445)+"
                    "0.01*sin(7*x[1]*pi+4234)+"
                    "0.01*sin(8*x[1]*pi+2346)+"
                    "0.01*sin(9*x[1]*pi+6544)+"
                    "0.01*sin(10*x[1]*pi+67)+"
                    "0.01*sin(11*x[1]*pi+234)+"
                    "0.01*sin(12*x[1]*pi+4525)+"
                    "0.01*sin(13*x[1]*pi+756)+"
                    "0.01*sin(14*x[1]*pi+24)"
                    ")/(sqrt(2)*eps))")
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
    u_init_expr = df.Constant((inlet_velocity, 0.))
    u_init = df.interpolate(u_init_expr, function_space)
    return u_init


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))

def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    # func = 1.-phi**2
    # return 0.75 * gamma * max_value(func, 0.)
    return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def reference(t_0, front_position_init, inlet_velocity, interface_thickness,
              **namespace):
    """ This contains the analytical reference for convergence analysis. """
    expr_str_phi = "tanh((x[0]-x0-u0x*t)/(sqrt(2)*eps))"
    expr = dict()
    expr["phi"] = df.Expression(expr_str_phi, t=t_0,
                                x0=front_position_init, u0x=inlet_velocity,
                                eps=interface_thickness, degree=2)
    expr["u"] = df.Expression(("u0x", "0."), u0x=inlet_velocity, degree=2)
    expr["p"] = df.Expression("0.", degree=2)
    expr["g"] = df.Expression("0.", degree=2)  # ?
    return expr
