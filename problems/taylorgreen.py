import dolfin as df
import numpy as np
import os
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, Charged
import cPickle
__author__ = "Gaute Linga"



class Wall(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def problem():
    info_cyan("Taylor-Green vortex flow with two-phase electrohydrodynamics.")

    solutes = [["c_p",  1, 3., 1., 2., -2.],
               ["c_m", -1, 3., 1., 2., -2.]]

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",  # "stable_single",
        folder="results_taylorgreen",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,  # False,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.001,
        t_0=0.,
        T=.25,
        N=32,
        solutes=solutes,
        base_elements=base_elements,
        Lx=2.*np.pi,
        Ly=2.*np.pi,
        concentration_init=1.,
        concentration_init_dev=0.5,
        #
        density=[1.5, 0.5],
        viscosity=[3., 1.],
        permittivity=[2., 2.],
        EC_scheme="NL2",
        use_iterative_solvers=False,
        grav_const=0.,
        c_cutoff=0.1,
        p_lagrange=False,
        #
        surface_tension=1.0,
        interface_thickness=1.0,  # /np.sqrt(2.0),
        pf_mobility_coeff=1.0,
        pf_init=1.
    )
    return parameters


def constrained_domain(Lx, Ly, **namespace):
    return None  # PeriodicBoundary(Lx, Ly)


def mesh(Lx=2.*np.pi, Ly=2.*np.pi, N=16, **namespace):
    m = df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly), N, N)
    return m


def initialize(Lx, Ly,
               surface_tension, interface_thickness,
               pf_init, pf_mobility_coeff,
               solutes,
               density, viscosity, permittivity,
               restart_folder,
               field_to_subspace,
               concentration_init,
               concentration_init_dev,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state. """
    w_init_field = dict()

    #import matplotlib.pyplot as plt

    if not restart_folder:
        exprs = reference(t=0., **vars())
        for field, expr in exprs.items():
            w_init_field[field] = df.interpolate(
                expr, field_to_subspace[field].collapse())
            #fig = df.plot(w_init_field[field])
            #plt.title(field)
            #plt.colorbar(fig)
            #plt.show()

    return w_init_field


def create_bcs(t_0, Lx, Ly,
               concentration_init, concentration_init_dev,
               surface_tension, interface_thickness,
               pf_init, pf_mobility_coeff,
               solutes, density, permittivity, viscosity,
               p_lagrange,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundaries and boundary conditions are defined here. """
    boundaries = dict(
        wall=[Wall()]
    )
    bcs = dict()
    for boundary_name in boundaries.keys():
        bcs[boundary_name] = dict()

    # Apply pointwise BCs e.g. to pin pressure.
    bcs_pointwise = dict()

    exprs = reference(t=t_0, **vars())

    for field, expr in exprs.items():
        if field != "p":
            bcs["wall"][field] = Fixed(expr)
    
    if enable_NS and not p_lagrange:
        bcs_pointwise["p"] = (
            exprs["p"], ("x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS"))

    return boundaries, bcs, bcs_pointwise


def tstep_hook(t, tstep, stats_intv, statsfile, field_to_subspace,
               field_to_subproblem, subproblems, w_, **namespace):
    info_blue("Timestep = {}".format(tstep))


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)


def rhs_source(t, solutes,
               surface_tension, interface_thickness, pf_mobility_coeff, pf_init,
               density, viscosity, permittivity,
               concentration_init, concentration_init_dev,
               enable_PF, enable_NS, enable_EC,
               **namespace):
    """ Source term on the right hand side of the conservation equations. """
    with open("data/taylorgreen_rhs.dat", "r") as infile:
        code = cPickle.load(infile)
    
    replace_vals = replace_dict(concentration_init, concentration_init_dev,
                                surface_tension, interface_thickness,
                                density, permittivity, viscosity, solutes,
                                pf_mobility_coeff, pf_init,
                                enable_NS, enable_PF, enable_EC)

    keys = []
    q = dict()
    if enable_NS and "u" in code:
        keys.append("u")
    if enable_PF and "phi" in code:
        keys.append("phi")
    if enable_EC:
        for solute in solutes:
            if solute[0] in code:
                keys.append(solute[0])
        if "V" in code:
            keys.append("V")

    for key in keys:
        q[key] = df.Expression(code[key], t=t, degree=2, **replace_vals)

    return q


def reference(t,
              concentration_init, concentration_init_dev,
              surface_tension, interface_thickness,
              pf_init, pf_mobility_coeff,
              solutes, density, permittivity, viscosity,
              enable_NS, enable_PF, enable_EC,
              **namespace):
    """ This contains the analytical reference for convergence analysis. """
    with open("data/taylorgreen_reference.dat", "r") as infile:
        code = cPickle.load(infile)

    replace_vals = replace_dict(concentration_init, concentration_init_dev,
                                surface_tension, interface_thickness,
                                density, permittivity, viscosity, solutes,
                                pf_mobility_coeff, pf_init,
                                enable_NS, enable_PF, enable_EC)

    code_strings = dict()
    if enable_NS:
        code_strings["u"] = code["u"]
        code_strings["p"] = code["p"]
    if enable_PF:
        code_strings["phi"] = code["phi"]
        code_strings["g"] = code["g"]
    if enable_EC:
        for solute in solutes:
            code_strings[solute[0]] = code[solute[0]]
        code_strings["V"] = code["V"]
    
    expr = dict()
    for key, code_string in code_strings.iteritems():
        expr[key] = df.Expression(code_string, t=t, degree=2,
                                  **replace_vals)
    return expr


def mean(a):
    """ Same as np.mean """
    return (a[0]+a[1])/2

def dmean(a):
    """ Consider moving somewhere else. """
    return (a[0]-a[1])/2


def replace_dict(concentration_init, concentration_init_dev,
                 surface_tension, interface_thickness,
                 density, permittivity, viscosity, solutes,
                 pf_mobility_coeff, pf_init,
                 enable_NS, enable_PF, enable_EC):
    diffusivity_p = solutes[0][2:4]
    diffusivity_m = solutes[1][2:4]
    solubility_p = solutes[0][4:6]
    solubility_m = solutes[1][4:6]
    replace_vals = dict(
        U0=1. if enable_NS else 0,
        c0=concentration_init if enable_EC else 0,
        chi=concentration_init_dev if enable_EC else 0,
        sigma_tilde=surface_tension*3./(2.*np.sqrt(2.)) if enable_PF else 0,
        eps=interface_thickness if enable_PF else 1.0/np.sqrt(2),
        veps=mean(permittivity) if enable_PF and enable_EC else permittivity[0],
        dveps=dmean(permittivity) if enable_PF and enable_EC else 0,
        D_p=mean(diffusivity_p) if enable_PF else diffusivity_p[0],
        dD_p=dmean(diffusivity_p) if enable_PF else 0,
        D_m=mean(diffusivity_m) if enable_PF else diffusivity_m[0],
        dD_m=dmean(diffusivity_m) if enable_PF else 0,
        beta_p=mean(solubility_p) if enable_PF else 0,  # solubility_p[0],
        dbeta_p=dmean(solubility_p) if enable_PF else 0,
        beta_m=mean(solubility_m) if enable_PF else 0,  # solubility_m[0],
        dbeta_m=dmean(solubility_m) if enable_PF else 0,
        mu=mean(viscosity) if enable_PF else viscosity[0],
        dmu=dmean(viscosity) if enable_PF else 0,
        rho=mean(density) if enable_PF else density[0],
        drho=dmean(density) if enable_PF else 0,
        M=pf_mobility_coeff if enable_PF else 0,
        Phi0=pf_init if enable_PF else 0
    )
    return replace_vals
