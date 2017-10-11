"""
This module defines the basic newton solver.

Binary electrohydrodynamics solved using monolitic newton solver 
and with a implicit euler timeintegartion. 

AB, 2017-06-1(based on basic.py) 
"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential
from . import *
from . import __all__


def get_subproblems(base_elements, solutes,
                    enable_NS, enable_PF, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    NSPFCE = []

    if enable_NS:
        NS = ([dict(name="u", element="u"),
              dict(name="p", element="p")])
        NSPFCE += NS

    if enable_PF: 
        PF = ([dict(name="phi", element="phi"),
              dict(name="g", element="g")])
        NSPFCE += PF
    if enable_EC: 
        EC = ([dict(name=solute[0], element="c")
                for solute in solutes] +
             [dict(name="V", element="V")])
        NSPFCE += EC

    subproblems["NSPFEC"] = NSPFCE

    return subproblems


def setup(test_functions, trial_functions, w_, w_1, bcs, permittivity,
          density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility_coeff, pf_mobility,
          use_iterative_solvers,
          **namespace):
    """ Set up problem. """
    # Constant
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness

    funs_ = df.split(w_["NSPFEC"])
    funs_1 = df.split(w_1["NSPFEC"])
    field_number = 0
    if enable_NS:
        v = test_functions["NSPFEC"][field_number]
        u_ = funs_[field_number]
        u_1 = funs_1[field_number]
        field_number += 1
        q = test_functions["NSPFEC"][field_number]
        p_ = funs_[field_number]
        p_1 = funs_1[field_number]
        field_number += 1
    if enable_PF:
        psi = test_functions["NSPFEC"][field_number]
        phi_ = funs_[field_number]
        phi_1 = funs_1[field_number]
        field_number += 1
        h = test_functions["NSPFEC"][field_number]
        g_ = funs_[field_number]
        g_1 = funs_1[field_number]
        field_number += 1
    if enable_EC:
        num_solutes = len(test_functions["NSPFEC"])-field_number-1
        b = test_functions["NSPFEC"][field_number:(num_solutes+field_number)]
        c_ = funs_[field_number:(num_solutes+field_number)]
        c_1 = funs_1[field_number:(num_solutes+field_number)]
        U = test_functions["NSPFEC"][num_solutes+field_number]
        V_ = funs_[num_solutes+field_number]
        V_1 = funs_1[num_solutes+field_number]

    M_ = pf_mobility(phi_, gamma)
    M_1 = pf_mobility(phi_1, gamma)
    nu_ = ramp(phi_, viscosity)
    nu_1 = ramp(phi_1, viscosity)
    veps_ = ramp(phi_, permittivity)
    veps_1 = ramp(phi_1, permittivity)
    rho_ = ramp(phi_, density)
    rho_1 = ramp(phi_1, density)
    dveps = dramp(permittivity)
    drho = dramp(density)

    dbeta = []  # Diff. in beta
    z = []  # Charge z[species]
    K_ = []  # Diffusivity K[species]
    K_1 = []
    beta_ = []  # Conc. jump func. beta[species]
    beta_1 = []

    for solute in solutes:
        z.append(solute[1])
        K_.append(ramp(phi_, [solute[2], solute[3]]))
        K_1.append(ramp(phi_1, [solute[2], solute[3]]))
        beta_.append(ramp(phi_, [solute[4], solute[5]]))
        beta_1.append(ramp(phi_, [solute[4], solute[5]]))
        dbeta.append(dramp([solute[4], solute[5]]))

    if enable_EC:
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of curr. sol.
        rho_e_1 = sum([c_e*z_e for c_e, z_e in zip(c_1, z)])  # prev sol.

    solver = dict()
    solver["NSPFEC"] = setup_NSPFEC(w_["NSPFEC"], w_1["NSPFEC"], bcs["NSPFEC"],
                                    trial_functions["NSPFEC"],
                                    v, q, psi, h, b, U,
                                    u_, p_, phi_, g_, c_, V_,
                                    u_1, p_1, phi_1, g_1, c_1, V_1,
                                    M_, nu_, veps_, rho_, K_, rho_e_,
                                    M_1, nu_1, veps_1, rho_1, K_1, rho_e_1,
                                    dbeta, dveps, drho,
                                    per_tau, sigma_bar, eps, grav, z,
                                    enable_NS, enable_PF, enable_EC,
                                    use_iterative_solvers)
    return dict(solvers=solver)


def setup_NSPFEC(w_NSPFEC, w_1NSPFEC, bcs_NSPFEC, trial_func_NSPFEC,
                 v, q, psi, h, b, U,
                 u_, p_, phi_, g_, c_, V_,
                 u_1, p_1, phi_1, g_1, c_1, V_1,
                 M_, nu_, veps_, rho_, K_, rho_e_,
                 M_1, nu_1, veps_1, rho_1, K_1, rho_e_1,
                 dbeta, dveps, drho,
                 per_tau, sigma_bar, eps, grav, z,
                 enable_NS, enable_PF, enable_EC,
                 use_iterative_solvers):
    """ The full problem of electrohydrodynamics in two pahase.
    Note that it is possioble to trun off the dirffent parts at will.
    """

    F_imp = NSPFEC_action(u_, u_1, phi_, phi_1, c_, c_1,
                          u_, p_, phi_, g_, c_, V_,
                          v, q, psi, h, b, U,
                          rho_, M_, nu_, rho_e_, K_, veps_,
                          drho, grav, sigma_bar, eps, dveps, dbeta, z,
                          per_tau,
                          enable_NS, enable_PF, enable_EC)
    F_exp = NSPFEC_action(u_, u_1, phi_, phi_1, c_, c_1,
                          u_1, p_1, phi_1, g_1, c_1, V_1,
                          v, q, psi, h, b, U,
                          rho_1, M_1, nu_1, rho_e_1, K_1, veps_1,
                          drho, grav, sigma_bar, eps, dveps, dbeta, z,
                          per_tau,
                          enable_NS, enable_PF, enable_EC)
    F = 0.5*(F_imp + F_exp)
    J = df.derivative(F, w_NSPFEC)

    problem_NSPFEC = df.NonlinearVariationalProblem(F, w_NSPFEC, bcs_NSPFEC, J)
    solver_NSPFEC = df.NonlinearVariationalSolver(problem_NSPFEC)
    if use_iterative_solvers:
        solver_NSPFEC.parameters['newton_solver']['linear_solver'] = 'gmres'
        solver_NSPFEC.parameters['newton_solver']['preconditioner'] = 'ilu'

    return solver_NSPFEC


def solve(solvers, **namespace):
    """ Solve equations. """
    solvers["NSPFEC"].solve()


def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    w_1["NSPFEC"].assign(w_["NSPFEC"])


def NSPFEC_action(u_, u_1, phi_, phi_1, c_, c_1,
                  u, p, phi, g, c, V,
                  v, q, psi, h, b, U,
                  rho, M, nu, rho_e, K, veps,
                  drho, grav, sigma_bar, eps, dveps, dbeta, z,
                  per_tau,
                  enable_NS, enable_PF, enable_EC):
    # The setup of the Navier-Stokes part of F
    F = []
    if enable_NS:
        F_NS = (per_tau * rho * df.dot(u_ - u_1, v)*df.dx
                + df.inner(
                    df.grad(u),
                    df.outer(rho*u - drho*M*df.grad(g), v))*df.dx
                + 2*nu*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
                - p*df.div(v)*df.dx
                + df.div(u)*q*df.dx
                - df.dot(rho*grav, v)*df.dx)
        if enable_PF:
            F_NS += - sigma_bar*eps*df.inner(
                df.outer(df.grad(phi),
                         df.grad(phi)), df.grad(v))*df.dx
        if enable_EC and rho_e != 0:
            F_NS += rho_e*df.dot(df.grad(V), v)*df.dx
        if enable_PF and enable_EC:
            F_NS += dveps*df.dot(
                df.grad(phi), v)*df.dot(df.grad(V),
                                        df.grad(V))*df.dx
        F.append(F_NS)

    # The setup of the Phase feild equations
    if enable_PF:
        F_PF_phi = (per_tau*(phi_-phi_1)*psi*df.dx +
                    M*df.dot(df.grad(g), df.grad(psi))*df.dx)
        if enable_NS:
            F_PF_phi += df.dot(u, df.grad(phi))*psi*df.dx

        F_PF_g = (g*h*df.dx
                  - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*df.dx
                  - sigma_bar/eps*diff_pf_potential(phi)*h*df.dx)
        if enable_EC:
            F_PF_g += (-sum([dbeta_i*ci*h*df.dx
                             for dbeta_i, ci in zip(dbeta, c)])
                       + dveps*df.dot(df.grad(V), df.grad(V))*h*df.dx)
        F_PF = F_PF_phi + F_PF_g
        F.append(F_PF)

    # The setup of the Electrochemistry
    if enable_EC:
        F_E_c = []
        for ci, ci_, ci_1, bi, Ki, zi in zip(c, c_, c_1, b, K, z):
            F_E_ci = (per_tau*(ci_-ci_1)*bi*df.dx
                      + Ki*df.dot(df.grad(ci), df.grad(bi))*df.dx)
            if zi != 0:
                F_E_ci += Ki*zi*ci*df.dot(df.grad(V),
                                          df.grad(bi))*df.dx
            if enable_NS:
                F_E_ci += df.dot(u, df.grad(ci))*bi*df.dx
            F_E_c.append(F_E_ci)
        F_E_V = veps*df.dot(df.grad(V), df.grad(U))*df.dx
        if rho_e != 0:
            F_E_V += -rho_e*U*df.dx
        F_E = sum(F_E_c) + F_E_V
        F.append(F_E)

    F = sum(F)
    J = df.derivative(F, w_NSPFEC)

    problem_NSPFEC = df.NonlinearVariationalProblem(F, w_NSPFEC, bcs_NSPFEC, J)
    solver_NSPFEC = df.NonlinearVariationalSolver(problem_NSPFEC)
    if use_iterative_solvers:
        solver_NSPFEC.parameters['newton_solver']['linear_solver'] = 'gmres'
        solver_NSPFEC.parameters['newton_solver']['preconditioner'] = 'ilu'

    return solver_NSPFEC

def solve(solvers, **namespace):
    """ Solve equations. """
    solvers["NSPFEC"].solve()

def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    w_1["NSPFEC"].assign(w_["NSPFEC"])
