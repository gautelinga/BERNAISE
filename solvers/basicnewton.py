"""
This module defines the basic newton solver.

Binary electrohydrodynamics solved using monolitic newton solver 
and with a implicit euler timeintegartion. 

AB, 2017-06-1(based on basic.py) 
"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential_linearised
from . import *
from . import __all__

def get_subproblems(base_elements, solutes,
                    enable_NS, enable_PF, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if (enable_NS and enable_PF and enable_EC):
        subproblems["NSPFEC"] = ([dict(name="u", element="u"),
                                  dict(name="p", element="p"),
                                  dict(name="phi", element="phi"),
                                  dict(name="g", element="g")] +
                                 [dict(name=solute[0], element="c")
                                        for solute in solutes] +
                                 [dict(name="V", element="V")])
    else if (enable_PF and enable_EC):
        subproblems["NSPFEC"]
    else if (enable_NS and enable_PF):
        subproblems["NSPFEC"]
    else if (enable_NS and enable_EC):
        subproblems["NSPFEC"]                    
    else if enable_NS:
        subproblems["NSPFEC"] = [dict(name="u", element="u"),
                             dict(name="p", element="p")]
    else if enable_PF:
        subproblems["NSPFEC"] = [dict(name="phi", element="phi"),
                             dict(name="g", element="g")]
    else if enable_EC:
        subproblems["NSPFEC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
    return subproblems


def setup(test_functions, trial_functions, w_, w_1, bcs, permittivity,
          density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility_coeff,
          **namespace):
    """ Set up problem. """
    # Constant
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness

    # Navier-Stokes ([u,v],[p,q]) and Phase field ([phi,psi],[g,h])
    v, q, psi, h = test_functions["NSPF"]

    
    # Electrochemistry (At the moment only auxillary feilds)
    num_solutes = len(trial_functions["EC"])-1
    assert(num_solutes == len(solutes))
    c = trial_functions["EC"][0:num_solutes]
    V = trial_functions["EC"][num_solutes]
    b = test_functions["EC"][0:num_solutes]
    U = test_functions["EC"][num_solutes]

    # Solution feilds and old-feilds follows the same concentions as the trial/test-functions
    u_, p_, phi_, g_ = df.split(w_["NSPF"])
    u_1, p_1, phi_1, g_1 = df.split(w_1["NSPF"])

    cV_ = df.split(w_["EC"])
    cV_1 = df.split(w_1["EC"])
    c_, V_ = cV_[:num_solutes], cV_[num_solutes]
    c_1, V_1 = cV_1[:num_solutes], cV_1[num_solutes]

    M_ = pf_mobility(phi, gamma)
    nu_ = ramp(phi, viscosity)
    veps_ = ramp(phi, permittivity)
    rho_ = ramp(phi, density)
    dveps = dramp(permittivity)
    drho = dramp(density)

    dbeta = []  # Diff. in beta
    z = []  # Charge z[species]
    K_ = []  # Diffusivity K[species]
    beta_ = []  # Conc. jump func. beta[species]

    for solute in solutes:
        z.append(solute[1])
        K_.append(ramp(phi_, [solute[2], solute[3]]))
        beta_.append(ramp(phi_, [solute[4], solute[5]]))
        dbeta.append(dramp([solute[4], solute[5]]))

    rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial functions
    rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of current sol.
    rho_e_1 = sum([c_e*z_e for c_e, z_e in zip(c_1, z)])  # Sum of current sol.

    solver = dict()
    solver["NSPFEC"] = setup_NSPFEC(w_["NSPFEC"],
                                    u, p, v, q,
                                    phi, g, psi, h,
                                    c, V, b, U, rho_e,
                                    bcs["NSPFEC"],
                                    M_,nu_,veps_,rho_,
                                    per_tau, sigma_bar, eps, dbeta, dveps, drho, grav,
                                    enable_NS, enable_PF, enable_EC)
    return dict(solvers=solver)


def setup_NSPFEC(w_NSPFEC,w_1NSPFEC,
                 v, q, psi, h, b, U, rho_e,
                 bcs_NSPFEC,
                 M_,nu_,veps_,rho_,
                 per_tau, sigma_bar, eps, dbeta, dveps, drho, grav,
                 enable_NS, enable_PF, enable_EC):
    """ The Full problem of electrohydrodynamics in two pahase.
    Note that it is possioble to trun off the dirffent parts at will.
    """
    # The setup of the Navier-Stokes part of F
    F = []
    if enable_NS:
        F_NS = (per_tau * rho_ * df.dot(u_ - u_1, v)*df.dx
                + df.inner(
                    df.grad(u_),
                    df.outer(rho_*u_ - drho*M_*df.grad(g_),v))*df.dx
                + 2*nu_*df.inner(df.sym(df.grad(u_)), df.grad(v))*df.dx
                - p_*df.div(v)*df.dx
                + df.div(u_)*q*df.dx
                - df.dot(rho_*grav, v)*df.dx)
        if enable_PF:
            F_NS += - sigma_bar*eps*df.inner(df.outer(df.grad(phi_),
                df.grad(phi_)),df.grad(v))*df.dx
        if enable_EC:
            F_NS += rho_e_*df.dot(df.grad(V_), v)*df.dx
        if enable_PF and enable_EC:
            F_NS += dveps*df.dot(df.grad(phi_), v)*df.dot(df.grad(V_),
                                                        df.grad(V_))*df.dx
        F.append(F_NS)
    # The setup of the Phase feild equations 
    if enable_PF:
        F_PF_phi = (per_tau*(phi_-phi_1)*psi*df.dx +
                    M_*df.dot(df.grad(g_), df.grad(psi))*df.dx)
        if enable_NS:
            F_PF_phi += df.dot(u_, df.grad(phi_))*psi*df.dx

        F_PF_g = (g_*h*df.dx
               - sigma_bar*eps*df.dot(df.grad(phi_), df.grad(h))*df.dx
               - sigma_bar/eps*diff_pf_potential(phi_)*h*df.dx)
        if enable_EC:
            F_PF_g += (-sum([dbeta_i*ci_*h*df.dx
                         for dbeta_i, ci_ in zip(dbetai, ci_)])
                            + dveps*df.dot(df.grad(V_), df.grad(V_))*h*df.dx)
        F_PF = F_PF_phi + F_PF_g
        F.append(F_PF)
    # The setup of the Electrochemistry
    if enable_EC:
        F_E_c = []
        for ci_, ci_1, bi, Ki_, zi in zip(ci_, ci_1, bi, Ki_, zi):
            F_E_c_i = (per_tau*(ci_-ci_1)*bi*df.dx
                       + Ki_*df.dot(df.grad(ci_), df.grad(bi))*df.dx
                       + zi*ci_*df.dot(df.grad(V_), df.grad(bi))*df.dx)
        if enable_NS:
            F_E_c_i += df.dot(u0, df.grad(ci_))*bi*df.dx
        F_E_c.append(F_E_c_i)
        F_E_V = (veps_*df.dot(df.grad(V_), df.grad(U))*df.dx
                 + rho_e_*U*df.dx)
        F_E = sum(F_E_c) + F_E_V
        F.append(F_E)
    
    F = sum(F)
    J = df.derivative(F, w_NSPFEC)
    problem_NSPFEC = df.NonlinearVariationalProblem(F, w_NSPFEC, bcs_NSPFEC, J)
    solver_NSPFEC = df.NonlinearVariationalSolver(problem_NSPFEC)
    return solver_NSPFEC

def solve(solvers, **namespace):
    """ Solve equations. """
    solvers["NSPFEC"].solve()
   
def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    w_1["NSPFEC"].assign(w_["NSPFEC"])

