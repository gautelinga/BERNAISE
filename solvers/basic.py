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
from . import *
from . import __all__


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

    # Navier-Stokes
    u, p = trial_functions["NS"]
    v, q = test_functions["NS"]

    # Phase field
    phi, g = trial_functions["PF"]
    psi, h = test_functions["PF"]

    # Electrochemistry
    num_solutes = len(trial_functions["EC"])-1
    assert(num_solutes == len(solutes))
    c = trial_functions["EC"][0:num_solutes]
    V = trial_functions["EC"][num_solutes]
    b = test_functions["EC"][0:num_solutes]
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
    # rho_e_1 = sum([c_e*z_e for c_e, z_e in zip(c_1, z)])  # sum of prev. sol.

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


def setup_NS(w1_NS, u, p, v, q, bcs_NS,
             u0, phi1, rho1, g1, M1, nu1, rho_e_, V1,
             per_tau, drho, sigma_bar, eps, dveps, grav,
             enable_PF, enable_EC):
    """ Set up the Navier-Stokes subproblem. """

    F_NS = (per_tau * rho1 * df.dot(u - u0, v)*df.dx
            + df.inner(
                df.grad(u),
                df.outer(rho1*u0 - drho*M1*df.grad(g1), v))*df.dx
            + 2*nu1*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
            - p * df.div(v)*df.dx
            + df.div(u)*q*df.dx)
    if enable_PF:
        F_NS += - sigma_bar*eps*df.inner(df.outer(df.grad(phi1),
                                                  df.grad(phi1)),
                                         df.grad(v))*df.dx
    if enable_EC:
        F_NS += rho_e_*df.dot(df.grad(V1), v)*df.dx
    if enable_PF and enable_EC:
        F_NS += dveps * df.dot(df.grad(phi1), v)*df.dot(df.grad(V1),
                                                        df.grad(V1))*df.dx

    a_NS, L_NS = df.lhs(F_NS), df.rhs(F_NS)

    problem_NS = df.LinearVariationalProblem(a_NS, L_NS, w1_NS, bcs_NS)
    solver_NS = df.LinearVariationalSolver(problem_NS)
    return solver_NS


def setup_PF(w1_PF, phi, g, psi, h, bcs_PF,
             phi0, u0, M0, c0, V0,
             per_tau, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC):
    """ Set up phase field subproblem. """

    F_PF_phi = (per_tau*(phi-phi0)*psi*df.dx +
                M0*df.dot(df.grad(g), df.grad(psi))*df.dx)
    if enable_NS:
        F_PF_phi += df.dot(u0, df.grad(phi))*psi*df.dx
    F_PF_g = (g*h*df.dx
              - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*df.dx
              - sigma_bar/eps*diff_pf_potential_linearised(phi, phi0)*h*df.dx)
    if enable_EC:
        F_PF_g += (-sum([dbeta_i*c0_i*h*df.dx
                         for dbeta_i, c0_i in zip(dbeta, c0)])
                   + dveps*df.dot(df.grad(V0), df.grad(V0))*h*df.dx)
    F_PF = F_PF_phi + F_PF_g
    a_PF, L_PF = df.lhs(F_PF), df.rhs(F_PF)

    problem_PF = df.LinearVariationalProblem(a_PF, L_PF, w1_PF)
    solver_PF = df.LinearVariationalSolver(problem_PF)
    return solver_PF


def setup_EC(w1_E, c, V, b, U, rho_e, bcs_E,
             c0, u0, K1, veps1,
             per_tau, z,
             enable_NS):
    """ Set up electrochemistry subproblem. """
    F_E_c = []
    for c_i, c0_i, b_i, K1_i, z_i in zip(c, c0, b, K1, z):
        F_E_c_i = (per_tau*(c_i-c0_i)*b_i*df.dx
                   + K1_i*df.dot(df.grad(c_i), df.grad(b_i))*df.dx
                   + z_i*c0_i*df.dot(df.grad(V), df.grad(b_i))*df.dx)
        if enable_NS:
            F_E_c_i += df.dot(u0, df.grad(c_i))*b_i*df.dx
        F_E_c.append(F_E_c_i)
    F_E_V = (veps1*df.dot(df.grad(V), df.grad(U))*df.dx
             + rho_e*U*df.dx)
    F_E = sum(F_E_c) + F_E_V
    a_E, L_E = df.lhs(F_E), df.rhs(F_E)

    problem_E = df.LinearVariationalProblem(a_E, L_E, w1_E, bcs_E)
    solver_E = df.LinearVariationalSolver(problem_E)
    return solver_E

def solve(solvers, **namespace):
    solvers["PF"].solve()
    solvers["EC"].solve()
    solvers["NS"].solve()

def diff_pf_potential_linearised(phi, phi0):
    """ Linearised phase field potential. """
    return phi0**3-phi0+(3*phi0**2-1.)*(phi-phi0)


def ramp(phi, A):
    """ Ramps between A[0] and A[1] according to phi. """
    return A[0]*0.5*(1.+phi) + A[1]*0.5*(1.-phi)


def dramp(A):
    """ Derivative of ramping function. Returns df.Constant."""
    return df.Constant(0.5*(A[0]-A[1]))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. Should be moved to problem. """
    return gamma * (phi**2-1.)**2
