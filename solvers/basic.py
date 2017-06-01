"""
This module defines the basic solver.

Binary electrohydrodynamics solved using a partial splitting approach and
linearisation. The problem is split between the following subproblems.
* The phase-field equation is solved simultaneously with the phase-field
  chemical potential (considered as a separate field), with a linearised
  double-well potential to make the problem linear.
* Solute concentrations are solved simultaneously as the electric potential,
  with a linearization of the c \grad V term, to make the problem linear.
* The Navier-Stokes equations are solved simultaneously for the velocity and
  pressure fields, where the intertial term is linearised to make the whole
  subproblem linear.

GL, 2017-05-29
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
    if enable_NS:
        subproblems["NS"] = [dict(name="u", element="u"),
                             dict(name="p", element="p")]
    if enable_PF:
        subproblems["PF"] = [dict(name="phi", element="phi"),
                             dict(name="g", element="g")]
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
    return subproblems


def setup(test_functions, trial_functions, w_, w_1, bcs, permittivity,
          density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility, pf_mobility_coeff,
          **namespace):
    """ Set up problem. """
    # Constant
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness

    # Navier-Stokes
    u, p = trial_functions["NS"]
    v, q = test_functions["NS"]

    # Phase field
    phi, g = trial_functions["PF"]
    psi, h = test_functions["PF"]

    # Electrochemistry
    num_solutes = len(trial_functions["EC"])-1
    assert(num_solutes == len(solutes))
    c = trial_functions["EC"][:num_solutes]
    V = trial_functions["EC"][num_solutes]
    b = test_functions["EC"][:num_solutes]
    U = test_functions["EC"][num_solutes]

    phi_, g_ = df.split(w_["PF"])
    phi_1, g_1 = df.split(w_1["PF"])

    u_, p_ = df.split(w_["NS"])
    u_1, p_1 = df.split(w_1["NS"])

    cV_ = df.split(w_["EC"])
    cV_1 = df.split(w_1["EC"])
    c_, V_ = cV_[:num_solutes], cV_[num_solutes]
    c_1, V_1 = cV_1[:num_solutes], cV_1[num_solutes]

    M_ = pf_mobility(phi_, gamma)
    M_1 = pf_mobility(phi_1, gamma)
    nu_ = ramp(phi_, viscosity)
    veps_ = ramp(phi_, permittivity)
    rho_ = ramp(phi_, density)
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

    solvers = dict()
    if enable_PF:
        solvers["PF"] = setup_PF(w_["PF"], phi, g, psi, h, bcs["PF"],
                                 phi_1, u_1, M_1, c_1, V_1,
                                 per_tau, sigma_bar, eps, dbeta, dveps,
                                 enable_NS, enable_EC)

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e, bcs["EC"], c_1,
                                 u_1, K_, veps_, per_tau, z, enable_NS)

    if enable_NS:
        solvers["NS"] = setup_NS(w_["NS"], u, p, v, q, bcs["NS"], u_1, phi_,
                                 rho_, g_, M_, nu_, rho_e_, V_,
                                 per_tau, drho, sigma_bar, eps, dveps, grav,
                                 enable_PF, enable_EC)
    return dict(solvers=solvers)


def setup_NS(w_NS, u, p, v, q, bcs,
             u_1, phi_, rho_, g_, M_, nu_, rho_e_, V_,
             per_tau, drho, sigma_bar, eps, dveps, grav,
             enable_PF, enable_EC):
    """ Set up the Navier-Stokes subproblem. """

    F = (per_tau * rho_ * df.dot(u - u_1, v)*df.dx
         + df.inner(
             df.grad(u),
             df.outer(rho_*u_1 - drho*M_*df.grad(g_), v))*df.dx
         + 2*nu_*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
         - p * df.div(v)*df.dx
         + df.div(u)*q*df.dx
         - df.dot(rho_*grav, v)*df.dx)
    if enable_PF:
        F += - sigma_bar*eps*df.inner(df.outer(df.grad(phi_),
                                               df.grad(phi_)),
                                      df.grad(v))*df.dx
    if enable_EC:
        F += rho_e_*df.dot(df.grad(V_), v)*df.dx
    if enable_PF and enable_EC:
        F += dveps * df.dot(df.grad(phi_), v)*df.dot(df.grad(V_),
                                                     df.grad(V_))*df.dx

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NS, bcs)
    solver = df.LinearVariationalSolver(problem)
    return solver


def setup_PF(w_PF, phi, g, psi, h, bcs,
             phi_1, u_1, M_1, c_1, V_1,
             per_tau, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC):
    """ Set up phase field subproblem. """

    F_phi = (per_tau*(phi-phi_1)*psi*df.dx +
             M_1*df.dot(df.grad(g), df.grad(psi))*df.dx)
    if enable_NS:
        F_phi += df.dot(u_1, df.grad(phi))*psi*df.dx
    F_g = (g*h*df.dx
           - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*df.dx
           - sigma_bar/eps*diff_pf_potential_linearised(phi, phi_1)*h*df.dx)
    if enable_EC:
        F_g += (-sum([dbeta_i*c_i_1*h*df.dx
                      for dbeta_i, c_i_1 in zip(dbeta, c_1)])
                + dveps*df.dot(df.grad(V_1), df.grad(V_1))*h*df.dx)
    F = F_phi + F_g
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_PF)
    solver = df.LinearVariationalSolver(problem)
    return solver


def setup_EC(w_EC, c, V, b, U, rho_e, bcs,
             c_1, u_1, K_, veps_,
             per_tau, z,
             enable_NS):
    """ Set up electrochemistry subproblem. """
    F_c = []
    for ci, ci_1, bi, Ki_, zi in zip(c, c_1, b, K_, z):
        F_ci = (per_tau*(ci-ci_1)*bi*df.dx
                + Ki_*df.dot(df.grad(ci), df.grad(bi))*df.dx
                + zi*ci_1*df.dot(df.grad(V), df.grad(bi))*df.dx)
        if enable_NS:
            F_ci += df.dot(u_1, df.grad(ci))*bi*df.dx
        F_c.append(F_ci)
    F_V = (veps_*df.dot(df.grad(V), df.grad(U))*df.dx + rho_e*U*df.dx)
    F = sum(F_c) + F_V
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_EC, bcs)
    solver = df.LinearVariationalSolver(problem)
    return solver


def solve(solvers, enable_PF, enable_EC, enable_NS, **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_NS, enable_EC, enable_PF]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            solvers[subproblem].solve()
            timer_inner.stop()
    timer_outer.stop()


def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_NS, enable_EC, enable_PF]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])
