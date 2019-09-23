import dolfin as df
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Pressure, NoSlip
#
from ufl import max_value
__author__ = "Gaute Linga"


class Left(df.SubDomain):
    def __init__(self, H):
        self.H = H
        df.SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return on_boundary and bool(df.near(x[0], 0.0) and
                                    not df.near(x[1], 0.) and
                                    not df.near(x[1], self.H))


class Right(df.SubDomain):
    def __init__(self, L, H):
        self.L = L
        self.H = H
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and bool(df.near(x[0], self.L) and
                                    not df.near(x[1], 0.) and
                                    not df.near(x[1], self.H))


class Wall(df.SubDomain):
    def __init__(self, L, H):
        self.L = L
        self.H = H
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and bool(df.near(x[1], 0.) or
                                    df.near(x[1], self.H))


class Cylinder(df.SubDomain):
    def __init__(self, L, H):
        self.L = L
        self.H = H
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and bool(not df.near(x[0], 0.) and
                                    not df.near(x[0], self.L) and
                                    not df.near(x[1], 0.) and
                                    not df.near(x[1], self.H)) 


def problem():
    info_cyan("Flow around cylinder benchmark.")
    #         2, beta in phase 1, beta in phase 2
    #solutes = [["c_p",  1, 1e-4, 1e-2, 4., 1.],
    #           ["c_m", -1, 1e-4, 1e-2, 4., 1.]]
    solutes = [["c_p",  0, 1e-3, 1e-2, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 2, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 2
    
    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_flow_around_cylinder",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=False,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.015/factor,
        t_0=0.,
        T=8.,
        res=factor*96,
        interface_thickness=0.015/factor,
        solutes=solutes,
        base_elements=base_elements,
        L=2.2,
        H=0.41,
        x0=0.2,
        y0=0.2,
        R=0.05,
        concentration_left=1.,
        #
        surface_tension=.04,  # 24.5,
        grav_const=0.0,
        inlet_velocity=1.5,
        V_0=0.,
        #
        pf_mobility_coeff=0.0020,
        density=[1., 1.],
        viscosity=[1e-3, 1e-3],
        permittivity=[1., 1.],
        #
        use_iterative_solvers=True,
        use_pressure_stabilization=False,
    )
    return parameters


def mesh(H=0.41, L=2.2, x0=0.2, y0=0.2, R=0.05, res=96, **namespace):
    mesh = load_mesh(
        "meshes/cylinderinchannel_H{}_L{}_x{}_y{}_r{}_res{}.h5".format(
            H, L, x0, y0, R, res))
    return mesh


def initialize(L, H, R,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               inlet_velocity, concentration_left,
               enable_NS, enable_PF, enable_EC,
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
        if enable_NS:
            try:
                subspace = field_to_subspace["u"].collapse()
            except:
                subspace = field_to_subspace["u"]
            u_init = velocity_init(L, H, inlet_velocity)
            w_init_field["u"] = df.interpolate(u_init, subspace)
        # Phase field
        if enable_PF:
            w_init_field["phi"] = df.interpolate(
                df.Constant(1.),
                field_to_subspace["phi"].collapse())

    #     if enable_EC:
    #         for solute in solutes:
    #             w_init_field[solute[0]] = initial_phasefield(
    #                 front_position_init, Lx/2, rad_init, interface_thickness,
    #                 field_to_subspace[solute[0]].collapse(),
    #                 shape=initial_interface)
    #             w_init_field[solute[0]].vector()[:] = \
    #                 concentration_left*(
    #                     - w_init_field[solute[0]].vector()[:]
    #                     + 1.0)/2.0

    return w_init_field


def create_bcs(L, H, inlet_velocity,
               V_0, solutes,
               concentration_left,
               interface_thickness,
               enable_NS, enable_PF, enable_EC, **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        wall=[Wall(L, H)],
        cylinder=[Cylinder(L, H)],
        right=[Right(L, H)],
        left=[Left(H)]
    )

    # Alocating the boundary dicts
    bcs = dict()
    bcs_pointwise = dict()
    for boundary in boundaries:
        bcs[boundary] = dict()

    velocity_expr = velocity_init(L, H, inlet_velocity)
    velocity_in = Fixed(velocity_expr)
    pressure_out = Pressure(0.0)
    noslip = NoSlip()

    V_left = Fixed(V_0)
    V_right = Fixed(0.)

    if enable_NS:
        bcs["left"]["u"] = velocity_in
        bcs["right"]["p"] = pressure_out
        bcs["wall"]["u"] = noslip
        bcs["cylinder"]["u"] = noslip

    if enable_PF:
        phi_expr = df.Expression(
            "tanh((abs(x[1]-H/2)-H/16)/(sqrt(2)*eps))",
            H=H, eps=interface_thickness,
            degree=2)
        phi_inlet = Fixed(phi_expr)
        bcs["left"]["phi"] = phi_inlet
        # bcs["right"]["phi"] = Fixed(df.Constant(1.))

    if enable_EC:
        bcs["left"]["V"] = V_left
        bcs["right"]["V"] = V_right
        for solute in solutes:
            c_expr = df.Expression("c0*exp(-pow(x[1]-H/2, 2)/(2*0.01*0.01))",
                                   H=H, c0=concentration_left, degree=2)
            bcs["left"][solute[0]] = Fixed(c_expr)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, rad, eps, function_space):
    expr_str = "tanh((x[0]-x0)/(sqrt(2)*eps))"
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad,
                                  eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def velocity_init(L, H, inlet_velocity, degree=2):
    return df.Expression(
        ("4*U*x[1]*(H-x[1])/pow(H, 2)", "0.0"),
        L=L, H=H, U=inlet_velocity, degree=degree)


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    # func = 1.-phi**2 + 0.0001
    # return 0.75 * gamma * max_value(func, 0.)
    return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
