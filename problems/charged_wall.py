import dolfin as df
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged  
__author__ = "Asger Bolet"

info_cyan("Charge wall for at test of debye layers")

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


def problem():
    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  1, 1e-4, 1.e-3, 3., 1.],
               ["c_m", -1, 1e-4, 1.e-3, 3., 1.]]

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
        folder="results_charge_wall",
        restart_folder=False,
        enable_NS=True,
        enable_PF=False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.08,
        t_0=0.,
        T=20.,
        dx=factor*1./16,
        interface_thickness=factor*0.080,
        solutes=solutes,
        base_elements=base_elements,
        Lx=1.,
        Ly=5.,
        rad_init=0.25,
        #
        V_top=1.,
        V_btm=0.,
        surface_tension=24.5,
        grav_const=0.0,
        #
        pf_mobility_coeff=factor*0.000040,
        density=[1000., 1000.],
        viscosity=[1., 1.],
        permittivity=[1., 5.],
        #
        initial_interface="flat",
        #
        use_iterative_solvers=False,
        use_pressure_stabilization=False
    )
    return parameters

def mesh(Lx=1, Ly=5, dx=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/dx), int(Ly/dx))

def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC, initial_interface, **namespace):
    """ Create the initial state.
    The initial states are specified in a dict indexed by field. The format
    should be
                w_init_field[field] = 'df.Function(...)'.
    The work dicts w_ and w_1 are automatically initialized from these
    functions elsewhere in the code.

    Note: You only need to specify the initial states that are nonzero.
    """

    c0 = [.001, .001]
    w_init_field = dict()
    if not restart_folder:
        if enable_EC:
            for ci, solute in zip(c0, solutes):
                solute_name = solute[0]
                w_init_field[solute_name] = initial_c(
                    ci,
                    field_to_subspace[solute_name].collapse())
            V_init_expr = df.Expression("0.", degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                Lx/5, Lx/2, rad_init, interface_thickness,
                field_to_subspace["phi"].collapse(), shape=initial_interface)

    return w_init_field


def create_bcs(Lx, Ly, solutes, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        right=[Right(Lx)],
        left=[Left(0)],
        bottom=[Bottom(0)],
        top=[Top(Ly)]
    )
    c0 = [.001, .001]
    noslip = Fixed((0., 0.))
    V_ground = Fixed(0.0)
    surfacecharge = Charged(0.0001)

    bcs = dict()

    bcs["right"] = dict(
        u=noslip,
    )
    bcs["left"] = dict(
        u=noslip
    )

    bcs["top"] = dict(
        u=noslip,
        V=surfacecharge
    )
    bcs["bottom"] = dict(
        V=V_ground
    )
    for ci, solute in zip(c0, solutes):
        bcs["bottom"][solute[0]] = Fixed(ci)

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()
    bcs_pointwise["p"] = (0., "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS")

    return boundaries, bcs, bcs_pointwise

def create_bcs_old(field_to_subspace, Lx, Ly, solutes,
               V_top, V_btm,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundary conditions are defined in terms of field. """
    bcs_fields = dict()

    # Navier-Stokes
    
    # Define the FacetFunction an mark all
    boundary_markers = df.FacetFunction('size_t', mesh)
    boundary_markers.set_all(mark[0])
    # Find the differet parts of the boundary
    right = Right(Lx)
    left = Left(0)
    bottom = Bottom(0)  
    top = Top(Ly)
    

    
    #Mark the differt parts of boundary in the FacetFunction
    right.mark(boundary_markers,mark["wall"])
    left.mark(boundary_markers,mark["openside"])
    top.mark(boundary_markers,mark["openside"])
    bottom.mark(boundary_markers,mark["openside"])


    if enable_NS:
        freeslip = df.DirichletBC(field_to_subspace["u"].sub(1),
                                  df.Constant(0.),
                                  bottom)
        leftwall = df.DirichletBC(field_to_subspace["u"],
                                df.Constant((0., 0.)),
                                left)
        rightwall = df.DirichletBC(field_to_subspace["u"],
                                df.Constant((0., 0.)),
                                right)
        topwall = df.DirichletBC(field_to_subspace["u"],
                                df.Constant((0.,0.)),
                                top)

        bcs_fields["u"] = [leftwall, rightwall, topwall, freeslip]
        # The pressure gauge
        p_pin = df.DirichletBC(field_to_subspace["p"],
                               df.Constant(0.),
                               " x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS",
                               "pointwise")
        bcs_fields["p"] = [p_pin]

    # Phase field
    #if enable_PF:
        #phiinlet = df.DirichletBC(field_to_subspace["phi"],
        #                        df.Constant(-1.0),
        #                        left)

        #phioutlet = df.DirichletBC(field_to_subspace["phi"],
        #                        df.Constant(1.0),
        #                        right)
        #bcs_fields["phi"] = [phiinlet, phioutlet]
         #bcs_fields["g"] = []

    # Electrochemistry

    if enable_EC:
        bc_V_bottom = df.DirichletBC(field_to_subspace["V"],
                                    df.Constant(V_top),
                                    bottom)
        bcs_fields["EC"] = [bc_V_top, bc_V_btm]

        for solute in solutes:
             bcs_fields[solute[0]] = []
             bcs_fields[solute[0]] = []
        bcs_fields["V"] = [bc_V_top, bc_V_btm]
    return bcs_fields



def initial_phasefield(x0, y0, rad, eps, function_space, shape="circle"):
    if shape == "flat":
        expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    elif shape == "sine":
        expr_str = "tanh((x[0]-x0-10*eps*sin(x[1]*pi))/(sqrt(2)*eps))"
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


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)

def initial_c(c_init, function_space):
    # expr_str = ("{c_init}*0.5*(1.-tanh(sqrt(2)*(sqrt(pow(x[0]-{x}, 2)"
    #             "+pow(x[1]-{y}, 2))-{rad})/{eps}))").format(
    #                 x=x, y=y, rad=rad, eps=eps, c_init=c_init)
    expr_str = ("c_init")
    c_init_expr = df.Expression(expr_str, c_init=c_init, degree=1)
    return df.interpolate(c_init_expr, function_space)
