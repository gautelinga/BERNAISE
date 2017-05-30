import dolfin as df
import os
from . import *
__author__ = "Gaute Linga"

info_cyan("Welcome to the simple problem!")

# Define solutes.

# Format: name, valency, diffusivity in phase 1, diffusivity in phase
#         2, beta in phase 1, beta in phase 2
solutes = [("c_p",  1, 1., 1., 1., 1.),
           ("c_m", -1, 1., 1., 1., 1.)]

# Format: name : (family, degree, is_vector)
base_elements = dict(u=("Lagrange", 2, True),
                     p=("Lagrange", 1, False),
                     phi=("Lagrange", 1, False),
                     g=("Lagrange", 1, False),
                     c=("Lagrange", 1, False),
                     V=("Lagrange", 1, False))
subproblems = dict(NS=[dict(name="u", element="u"),
                       dict(name="p", element="p")],
                   PF=[dict(name="phi", element="phi"),
                       dict(name="g", element="g")],
                   EC=[dict(name=solute[0], element="c") for solute in solutes] + [dict(name="V", element="V")])

factor = 1./4.

# Default parameters to be loaded unless starting from checkpoint.
parameters.update(
    solver="basic",
    folder="results_simple",
    restart_folder=False,
    enable_NS=True,
    enable_PF=True,
    enable_EC=True,
    save_intv=5,
    checkpoint_intv=10,
    tstep=0,
    dt=factor*0.08,
    t_0=0.,
    T=20.,
    dx=factor*1./16,
    interface_thickness=factor*0.040,
    solutes=solutes,
    base_elements=base_elements,
    subproblems=subproblems,
    Lx=1.,
    Ly=2.,
    rad_init=0.25,
    #
    V_top=1.,
    V_btm=0.,
    surface_tension=24.5,
    grav_const=0.98,
    #
    pf_mobility_coeff=factor*0.000040,
    density=(1000., 100.),
    viscosity=(10., 1.),
    permittivity=(1., 5.)
)


def mesh(Lx=1, Ly=5, dx=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/dx), int(Ly/dx))


def initialize(w_, w_1, subproblems, Lx, Ly, rad_init,
               interface_thickness, solutes, restart_folder, **namespace):
    if not restart_folder:
        """ Create the initial state. """
        # Phase field
        phi_init = initial_phasefield(
            Lx/2, Lx/2, rad_init, interface_thickness,
            w_["PF"].function_space().sub(0).collapse())
        g_init = df.interpolate(df.Constant(0.),
                                w_["PF"].function_space().sub(1).collapse())
        w_PF_init = df.project(df.as_vector((phi_init, g_init)),
                               w_["PF"].function_space())
        w_["PF"].interpolate(w_PF_init)
        w_1["PF"].interpolate(w_PF_init)

        # Electrochemistry
        W_EC = w_["EC"].function_space()
        c_init = df.Function(W_EC.sub(0).collapse())
        V_init_expr = df.Expression("x[1]/Ly", Ly=Ly, degree=1)
        V_init = df.interpolate(V_init_expr, W_EC.sub(len(solutes)).collapse())
        c_init.vector()[:] = 0.
        w_EC_init = df.project(df.as_vector(
            tuple([c_init]*len(solutes) + [V_init])), W_EC)
        w_["EC"].interpolate(w_EC_init)
        w_1["EC"].interpolate(w_EC_init)


def create_bcs(spaces, Lx, Ly, solutes, V_top, V_btm, **namespace):
    bcs = dict()

    # Navier-Stokes
    freeslip = df.DirichletBC(
        spaces["NS"].sub(0).sub(0), df.Constant(0.),
        "on_boundary && (x[0] < DOLFIN_EPS || x[0] > {Lx}-DOLFIN_EPS)".format(Lx=Lx))
    noslip = df.DirichletBC(
        spaces["NS"].sub(0), df.Constant((0., 0.)),
        "on_boundary && (x[1] < DOLFIN_EPS || x[1] > {Ly}-DOLFIN_EPS)".format(Ly=Ly))
    bcs["NS"] = [noslip, freeslip]

    # Phase field
    bcs["PF"] = None

    # Electrochemistry
    bc_V_top = df.DirichletBC(
        spaces["EC"].sub(len(solutes)), df.Constant(V_top),
        "on_boundary && x[1] > {Ly}-DOLFIN_EPS".format(Ly=Ly))
    bc_V_btm = df.DirichletBC(spaces["EC"].sub(2), df.Constant(V_btm),
                              "on_boundary && x[1] < DOLFIN_EPS")
    bcs["EC"] = [bc_V_top, bc_V_btm]
    return bcs


def initial_phasefield(x0, y0, rad, eps, function_space, shape="circle"):
    if shape == "flat":
        expr_str = "tanh((x[1]-y0)/(sqrt(2)*eps))"
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


def tstep_hook(tstep, **namespace):
    info_blue("Timestep = {}".format(tstep))
