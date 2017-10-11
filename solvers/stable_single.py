"""This module implements the solver described in the paper.

GL, 2017

"""
import dolfin as df
import math
import numpy as np
from common.functions import ramp, dramp, diff_pf_potential_linearised, \
    unit_interval_filter, max_value
from . import *
from . import __all__


def get_subproblems(base_elements, solutes,
                    enable_NS, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_NS:
        subproblems["NS"] = [dict(name="u", element="u"),
                             dict(name="p", element="p")]
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
    return subproblems


def setup(test_functions, trial_functions, w_, w_1,
          ds, dx, normal,
          dirichlet_bcs, neumann_bcs, boundary_to_mark,
          permittivity, density, viscosity,
          solutes, enable_EC, enable_NS,
          dt,
          grav_const,
          use_iterative_solvers,
          EC_scheme,
          rhs_source,
          c_cutoff,
          concentration_init, concentration_init_dev,
          **namespace):
    """ Set up problem. """
    # Constant
    grav = df.Constant((0., -grav_const))
    nu = viscosity[0]
    veps = permittivity[0]
    rho = density[0]

    if EC_scheme in ["NL1", "NL2"]:
        nonlinear_EC = True
    else:
        nonlinear_EC = False

    # Navier-Stokes
    if enable_NS:
        u, p = trial_functions["NS"]
        v, q = test_functions["NS"]

        u_, p_ = df.split(w_["NS"])
        u_1, p_1 = df.split(w_1["NS"])
    else:
        u_ = p_ = None
        u_1 = p_1 = None

    # Electrochemistry
    if enable_EC:
        num_solutes = len(trial_functions["EC"])-1
        assert(num_solutes == len(solutes))

        cV_ = df.split(w_["EC"])
        cV_1 = df.split(w_1["EC"])
        c_, V_ = cV_[:num_solutes], cV_[num_solutes]
        c_1, V_1 = cV_1[:num_solutes], cV_1[num_solutes]

        if not nonlinear_EC:
            c = trial_functions["EC"][:num_solutes]
            V = trial_functions["EC"][num_solutes]
        else:
            c = c_
            V = V_
        b = test_functions["EC"][:num_solutes]
        U = test_functions["EC"][num_solutes]
    else:
        c_ = V_ = c_1 = V_1 = None

    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]
    beta = []

    for solute in solutes:
        z.append(solute[1])
        K.append(solute[2])
        beta.append(solute[4])

    if enable_EC:
        rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial func.
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of curr. sol.
    else:
        rho_e_ = None

    q_rhs = rhs_source(t=0., **vars())
    if enable_EC:
        grad_g_c = []
        grad_g_c_ = []
        for ci, ci_, ci_1, zi in zip(c, c_, c_1, z):
            grad_g_c.append(df.grad(
                alpha_prime_approx(ci, ci_1, EC_scheme, c_cutoff) + zi*V))
            grad_g_c_.append(df.grad(
                alpha_prime_approx(ci_, ci_1, EC_scheme, c_cutoff) + zi*V_))
        c_reg = regulate(ci, ci_1, EC_scheme, c_cutoff)
    else:
        grad_g_c = None
        grad_g_c_ = None
        c_reg = None

    solvers = dict()
    if enable_EC:
        w_EC = w_["EC"]
        dirichlet_bcs_EC = dirichlet_bcs["EC"]
        solvers["EC"] = setup_EC(**vars())

    if enable_NS:
        w_NS = w_["NS"]
        dirichlet_bcs_NS = dirichlet_bcs["NS"]
        solvers["NS"] = setup_NS(**vars())
    return dict(solvers=solvers, q_rhs=q_rhs)


def setup_NS(w_NS, u, p, v, q,
             dx, ds, normal,
             dirichlet_bcs_NS, neumann_bcs, boundary_to_mark,
             u_1, nu, c_1, grad_g_c_,
             dt, grav,
             enable_EC,
             use_iterative_solvers, **namespace):
    """ Set up the Navier-Stokes subproblem. """
    F = (1./dt * df.dot(u - u_1, v) * dx
         + df.inner(df.grad(u), df.outer(u_1, v)) * dx
         + nu*df.inner(df.grad(u), df.grad(v)) * dx
         - p * df.div(v) * dx
         - q * df.div(u) * dx
         - df.dot(grav, v) * dx)

    for boundary_name, pressure in neumann_bcs["p"].iteritems():
        F += pressure * df.inner(
            normal, v) * ds(boundary_to_mark[boundary_name])

    if enable_EC:
        F += sum([ci_1*df.dot(grad_g_ci_, v)*dx
                  for ci_1, grad_g_ci_ in zip(c_1, grad_g_c_)])

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NS, dirichlet_bcs_NS)
    solver = df.LinearVariationalSolver(problem)

    # if use_iterative_solvers:
        # solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "ilu"

    return solver


def alpha(c):
    return c*(df.ln(c)-1)


def alpha_c(c):
    return df.ln(c)


def alpha_cc(c):
    return 1./c


def alpha_cc_reg(c, c_cutoff):
    return alpha_cc(max_value(c, c_cutoff))


def alpha_c_reg(c, c_cutoff):
    c_max = max_value(c, c_cutoff)
    dc = c_max - c_cutoff
    return (alpha_c(c_max) - alpha_c(c_cutoff)
            + alpha_cc(c_cutoff)*c
            - alpha_cc(c_cutoff)*dc)


def alpha_reg(c, c_cutoff):
    c_max = max_value(c, c_cutoff)
    dc = c_max-c_cutoff
    return (alpha(c_max) - alpha(c_cutoff)
            + 0.5*alpha_cc(c_cutoff)*c**2
            - alpha_c(c_cutoff)*dc
            - 0.5*alpha_cc(c_cutoff)*dc**2, c_cutoff)


def alpha_prime_approx(ci, ci_1, EC_scheme, c_cutoff):
    if EC_scheme == "NL1":
        return (ci-ci_1)*(alpha(ci)-alpha(ci_1))/((ci-ci_1)**2+0.001)
    elif EC_scheme == "NL2":
        return alpha_c(ci)
    elif EC_scheme == "L1":
        return alpha_c(ci_1) + 0.5*alpha_cc(ci_1)*(ci-ci_1)
    elif EC_scheme == "L2":
        return alpha_c_reg(ci_1, c_cutoff) + 0.5*alpha_cc(c_cutoff)*(ci-ci_1)


def regulate(ci, ci_1, EC_scheme, c_cutoff):
    if EC_scheme in ["NL1", "NL2"]:
        return max_value(ci, 0.)
    if EC_scheme == ["L2"]:
        return max_value(ci_1, c_cutoff)
    else:
        return max_value(ci_1, 0.)


def setup_EC(w_EC, c, V, b, U, rho_e, grad_g_c, c_reg,
             dx, ds,
             dirichlet_bcs_EC, neumann_bcs, boundary_to_mark,
             c_1, u_1, K, veps,
             dt, z, q_rhs,
             enable_NS,
             rhs_source,
             use_iterative_solvers,
             nonlinear_EC,
             **namespace):
    """ Set up electrochemistry subproblem. """

    # Projected velocity
    u_star = u_1 - dt*sum([ci_1*grad_g_ci
                           for ci_1, grad_g_ci in zip(c_1, grad_g_c)])

    F_c = []
    for ci, ci_1, bi, Ki, grad_g_ci, qi in zip(c, c_1, b, K, grad_g_c, q_rhs):
        F_ci = (1./dt*(ci-ci_1)*bi*dx +
                Ki*c_reg*df.dot(grad_g_ci, df.grad(bi))*dx)
        if enable_NS:
            F_ci += df.dot(u_star, df.grad(ci_1))*bi*dx
        if qi is not None:
            F_ci += - qi*bi*dx
        F_c.append(F_ci)
    F_V = veps*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    F = sum(F_c) + F_V

    if nonlinear_EC:
        J = df.derivative(F, w_EC)
        problem = df.NonlinearVariationalProblem(F, w_EC, dirichlet_bcs_EC, J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-7
        if use_iterative_solvers:
            solver.parameters["newton_solver"]["linear_solver"] = "bicgstab"
            solver.parameters["newton_solver"]["preconditioner"] = "amg"
    else:
        a, L = df.lhs(F), df.rhs(F)
        problem = df.LinearVariationalProblem(a, L, w_EC, dirichlet_bcs_EC)
        solver = df.LinearVariationalSolver(problem)
        if use_iterative_solvers:
            solver.parameters["linear_solver"] = "bicgstab"
            solver.parameters["preconditioner"] = "amg"

    return solver


def solve(w_, t, q_rhs, solvers, enable_EC, enable_NS, **namespace):
    """ Solve equations. """
    if enable_EC:
        # Update the time-dependent source terms
        for qi in q_rhs:
            if qi is not None:
                qi.t = t

    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["EC", "NS"], [enable_EC, enable_NS]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            #if subproblem == "EC":
            #    print "Randomizing"
            #    w_["EC"].vector()[:] = 1+0.0*(np.random.rand(len(w_["EC"].vector().array()))-0.5)
            solvers[subproblem].solve()
            timer_inner.stop()

    timer_outer.stop()


def update(w_, w_1, t, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(["EC", "NS"], [enable_EC, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])
