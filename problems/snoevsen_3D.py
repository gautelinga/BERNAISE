import dolfin as df
import os
from . import *
from common.io import mpi_is_root, load_mesh
from common.bcs import Fixed, Charged
from common.functions import max_value
__author__ = "Gaute Linga"


class Top(df.SubDomain):
    def __init__(self, Lz):
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[2], self.Lz) and on_boundary)


class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[2] < df.DOLFIN_EPS)


class PeriodicDomain(df.SubDomain):
    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool((df.near(x[0], -self.Lx/2.) or
                     df.near(x[1], -self.Ly/2.)) and
                    (not (df.near(x[0], self.Lx/2.) or
                          df.near(x[1], self.Ly/2.))) and on_boundary)

    def map(self, x, y):
        if df.near(x[0], self.Lx/2.) and df.near(x[1], self.Ly/2.):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
            y[2] = x[2]
        elif df.near(x[0], self.Lx/2.):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[1], Ly/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly
            y[2] = x[2]


def problem():
    info_cyan("Desnotting Snoevsen in 3D.")

    #         2, beta in phase 1, beta in phase 2
    solutes = [["c_p",  2, 1e-4, 1e-2, 4., 1.],
               ["c_m", -2, 1e-4, 1e-2, 4., 1.]]

    # Format: name : (family, degree, is_vector)
    base_elements = dict(u=["Lagrange", 1, True],
                         p=["Lagrange", 1, False],
                         phi=["Lagrange", 1, False],
                         g=["Lagrange", 1, False],
                         c=["Lagrange", 1, False],
                         V=["Lagrange", 1, False])

    factor = 1./4
    sigma_e = -10.  # 0.

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="TDLUES",
        folder="results_snoevsen_3D",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,  # True,
        enable_EC=False,  # True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=factor*0.04,
        t_0=0.,
        T=20.,
        res=48,
        interface_thickness=factor*0.250,
        solutes=solutes,
        base_elements=base_elements,
        Lx=6.,
        Ly=3.,
        Lz=2.,
        R=1.0,
        r=0.5,
        surface_charge=sigma_e,
        concentration_init=2.,
        velocity_top=.4,
        #
        surface_tension=1.45,
        grav_const=0.0,
        grav_dir=[0, 0, 1.],
        #
        pf_mobility_coeff=factor*0.000010,
        density=[10., 10.],
        viscosity=[1., 1.],
        permittivity=[1., 1.],
        use_iterative_solvers=True,
        solve_initial=False
    )
    return parameters


def constrained_domain(Lx, Ly, Lz, **namespace):
    return PeriodicDomain(Lx, Ly, Lz)


def mesh(res, **namespace):
    return load_mesh("meshes/snoevsen_3d_periodic_res{}.h5".format(res))


def initialize(Lx, Ly, Lz, R, r,
               interface_thickness, solutes, restart_folder,
               field_to_subspace,
               concentration_init,
               velocity_top,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()
    if not restart_folder:
        # Velocity
        if enable_NS:
            u_expr = df.Expression(("0.5*u0*(x[2] + abs(x[2]))/Lz", "0.", "0."),
                                   u0=velocity_top, Lz=Lz, degree=1)
            w_init_field["u"] = df.interpolate(u_expr, field_to_subspace["u"])

        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_phasefield(
                0., 0., 0., R, r, interface_thickness,
                field_to_subspace["phi"])

        # Chemicals
        # if enable_EC:
        #     for solute in solutes:
        #         concentration_init_loc = concentration_init/abs(solute[1])
        #         c_init = initial_phasefield(
        #             Lx/2, 0., R, interface_thickness,
        #             field_to_subspace["phi"])
        #         # Only have ions in phase 2 (phi=-1)
        #         c_init.vector()[:] = concentration_init_loc*0.5*(
        #             1.-c_init.vector().get_local())
        #         w_init_field[solute[0]] = c_init
        pass

    return w_init_field


def create_bcs(Lx, Ly, Lz,
               velocity_top, solutes,
               concentration_init, surface_charge,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        top=[Top(Lz)],
        bottom=[Bottom()]
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
        bcs_pointwise["p"] = (0.,
                              "x[0] < {x0} + DOLFIN_EPS && "
                              "x[1] < {y0} + DOLFIN_EPS && "
                              "x[2] > {Lz} - DOLFIN_EPS".format(
                                  x0=-Lx/2., y0=-Ly/2., Lz=Lz))

    if enable_EC:
        for solute in solutes:
            bcs["top"][solute[0]] = Fixed(concentration_init/abs(solute[1]))
        bcs["top"]["V"] = Fixed(0.)
        bcs["bottom"]["V"] = Charged(surface_charge)

    return boundaries, bcs, bcs_pointwise


def initial_phasefield(x0, y0, z0, R, r, eps, function_space):
    expr_str = "-tanh(max(x[2]-z0, sqrt(pow(x[0]-x0, 2) + pow(x[1]-y0, 2))-R-r)/(sqrt(2)*eps))"
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, z0=z0, R=R, r=r,
                                  eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space.collapse())
    return phi_init


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    func = 1.-phi**2
    return 0.75 * gamma * max_value(0., func)


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
