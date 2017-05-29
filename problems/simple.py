from . import *
import dolfin as df
__author__ = "Gaute Linga"

info_cyan("Welcome to the simple problem!")

# Define solutes.

# Format: name, charge, diffusivity in phase 1, diffusivity in phase
#         2, beta in phase 1, beta in phase 2
solutes = [("cp", 1., 1., 1., 1., 1., 1.),
           ("cm", -1, 1., 1., 1., 1., 1.)]

"""
    factor = 1./4.
    h_int = factor * 1./16
    eps = factor * 0.040
    dt = factor * 0.08
    gamma = factor * 0.000040

    Lx, Ly = 1., 2.
    rad_init = 0.25
    t0 = 0.
    T = 20.

    rho_1 = 1000.
    nu_1 = 10.
    rho_2 = 100.
    nu_2 = 1.

    sigma = 24.5
    grav_const = 0.98

    mesh = get_mesh(Lx, Ly, h_int)
    sigma_bar = sigma*3./(2*math.sqrt(2))
    grav = df.Constant((0., -grav_const))
"""


parameters.update(
    solver="basic",
    folder="results_simple",
    restart_folder=False,
    enable_NS=True,
    enable_PF=True,
    enable_EC=True,
    tstep=0,
    dt=0.01,
    t_0=0.,
    T=20.,
    h_int=1./16,
    interface_thickness=0.040,
    solutes=solutes,
    fields=["u", "p"] + ["phi", "g"] + ["cp", "cm"] + ["V"],
    # Format: name : (family, degree, is_vector)
    base_elements={"u": ("Lagrange", 2, True),
                   "p": ("Lagrange", 1, False),
                   "phi": ("Lagrange", 1, False),
                   "g": ("Lagrange", 1, False),
                   "c": ("Lagrange", 1, False),
                   "V": ("Lagrange", 1, False)},
    subproblems={"NS": ["u", "p"],
                 "PF": ["phi", "g"],
                 "EC": ["c"]*len(solutes) + ["V"]},
    subproblems_order=["PF", "EC", "NS"],
    Lx=1.,
    Ly=5.,
    rad_init=0.25,
    #
    V_top=1.,
    V_btm=0.,
    surface_tension=24.5,
    grav_const=0.98,
    #
    pf_mobility_coeff=0.000040,
    density=(1000., 100.),
    viscosity=(10., 1.),
    permittivity=(1., 5.)
)


def mesh(Lx=1, Ly=5, h=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/h), int(Ly/h))


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