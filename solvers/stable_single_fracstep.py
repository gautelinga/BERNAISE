"""This module implements the fractional step (Chorin) solver
described in the paper.

GL, 2017
"""
from stable_single import setup_EC, alpha_prime_approx, alpha_generalized, \
    regulate
import dolfin as df
from . import *
from . import __all__
import numpy as np


def get_subproblems(base_elements, solutes, enable_NS, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_NS:
        subproblems["NSu"] = ([dict(name="u", element="u")])
        subproblems["NSp"] = ([dict(name="p", element="p")])
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
          grav_dir,
          use_iterative_solvers,
          EC_scheme,
          c_cutoff,
          q_rhs,
          mesh,
          V_lagrange, p_lagrange,
          **namespace):
    """ Set up problem. """
    # Constant
    grav = df.Constant(tuple(grav_const*np.array(grav_dir)))
    nu = viscosity[0]
    veps = permittivity[0]
    rho = density[0]

    if EC_scheme in ["NL1", "NL2"]:
        nonlinear_EC = True
    else:
        nonlinear_EC = False

    p0 = q0 = p0_ = p0_1 = None
    # Navier-Stokes
    if enable_NS:
        u = trial_functions["NSu"]
        p = trial_functions["NSp"]
        v = test_functions["NSu"]
        q = test_functions["NSp"]

        u_ = w_["NSu"]
        p_ = w_["NSp"]
        u_1 = w_1["NSu"]
        p_1 = w_1["NSp"]
    else:
        u_ = p_ = None
        u_1 = p_1 = None

    # Electrochemistry
    c_ = V_ = c_1 = V_1 = V0 = V0_ = V0_1 = b = U = U0 = None
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

    if enable_EC:
        grad_g_c = []
        grad_g_c_ = []
        c_reg = []
        for ci, ci_, ci_1, zi in zip(c, c_, c_1, z):
            grad_g_c.append(df.grad(
                alpha_prime_approx(ci, ci_1, EC_scheme, c_cutoff) + zi*V))
            grad_g_c_.append(df.grad(
                alpha_prime_approx(ci_, ci_1, EC_scheme, c_cutoff) + zi*V_))
            c_reg.append(regulate(ci, ci_1, EC_scheme, c_cutoff))
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
        w_NSu = w_["NSu"]
        w_NSp = w_["NSp"]
        dirichlet_bcs_NSu = dirichlet_bcs["NSu"]
        dirichlet_bcs_NSp = dirichlet_bcs["NSp"]
        solvers["NSu"] = setup_NSu(**vars())
        solvers["NSp"] = setup_NSp(**vars())
    return dict(solvers=solvers)


def setup_NSu(w_NSu, u, v,
              dx, ds, normal,
              dirichlet_bcs_NSu, neumann_bcs, boundary_to_mark,
              u_, p_, u_1, p_1, nu, c_1, grad_g_c_,
              dt, grav,
              enable_EC,
              trial_functions,
              use_iterative_solvers,
              mesh,
              **namespace):
    """ Set up the Navier-Stokes velocity subproblem. """
    solvers = dict()

    F_predict = (1./dt * df.dot(u - u_1, v) * dx
                 + df.inner(df.grad(u), df.outer(u_1, v)) * dx
                 + nu*df.inner(df.grad(u), df.grad(v)) * dx
                 - p_1 * df.div(v) * dx
                 - df.dot(grav, v) * dx)

    for boundary_name, pressure in neumann_bcs["p"].iteritems():
        F_predict += pressure * df.inner(
            normal, v) * ds(boundary_to_mark[boundary_name])

    if enable_EC:
        F_predict += sum([ci_1*df.dot(grad_g_ci_, v)*dx
                          for ci_1, grad_g_ci_ in zip(c_1, grad_g_c_)])

    a_predict, L_predict = df.lhs(F_predict), df.rhs(F_predict)
    #    if not use_iterative_solvers:
    problem_predict = df.LinearVariationalProblem(
        a_predict, L_predict, w_NSu, dirichlet_bcs_NSu)
    solvers["predict"] = df.LinearVariationalSolver(problem_predict)
    if use_iterative_solvers:
        solvers["predict"].parameters["linear_solver"] = "bicgstab"
        solvers["predict"].parameters["preconditioner"] = "amg"

    F_correct = (
        df.inner(u - u_, v)*df.dx
        - dt * (p_ - p_1) * df.div(v) * df.dx
    )
    a_correct, L_correct = df.lhs(F_correct), df.rhs(F_correct)
    problem_correct = df.LinearVariationalProblem(
        a_correct, L_correct, w_NSu, dirichlet_bcs_NSu)
    solvers["correct"] = df.LinearVariationalSolver(problem_correct)

    if use_iterative_solvers:
        solvers["correct"].parameters["linear_solver"] = "bicgstab"
        solvers["correct"].parameters["preconditioner"] = "amg"
    #else:
    #    solver = df.LUSolver("mumps")
    #    # solver.set_operator(A)
    #    return solver, a, L, dirichlet_bcs_NS

    return solvers


def setup_NSp(w_NSp, p, q, dirichlet_bcs_NSp,
              dt, u_, p_1,
              use_iterative_solvers,
              **namespace):
    """ Set up Navier-Stokes pressure subproblem. """
    F = (
        df.dot(df.grad(p - p_1), df.grad(q)) * df.dx
        + 1./dt * df.div(u_) * q * df.dx
    )

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(
        a, L, w_NSp, dirichlet_bcs_NSp)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "bicgstab"
        solver.parameters["preconditioner"] = "amg"
    
    return solver


def solve(tstep, w_, w_1, solvers,
          enable_EC, enable_NS,
          **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    if enable_EC:
        timer_inner = df.Timer("Solve subproblem EC")
        df.mpi_comm_world().barrier()
        solvers["EC"].solve()
        timer_inner.stop()
    if enable_NS:
        # Step 1: Predict u
        timer = df.Timer("NS: Predict velocity.")
        solvers["NSu"]["predict"].solve()
        timer.stop()

        # Step 2: Pressure correction
        timer = df.Timer("NS: Pressure correction")
        solvers["NSp"].solve()
        timer.stop()

        # Step 3: Velocity correction
        timer = df.Timer("NS: Velocity correction")
        solvers["NSu"]["correct"].solve()
        timer.stop()

    timer_outer.stop()


def update(w_, w_1, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(
            ["EC", "NSu", "NSp"],
            [enable_EC, enable_NS, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def discrete_energy(x_, solutes, permittivity,
                    c_cutoff, EC_scheme, dt, **namespace):
    if x_ is None:
        return ["E_kin"] + ["E_{}".format(solute[0])
                            for solute in solutes] + ["E_V"] + ["E_p"]

    u = x_["u"]
    grad_V = df.grad(x_["V"])
    veps = permittivity[0]
    grad_p = df.grad(x_["p"])

    alpha_list = [alpha_generalized(x_[solute[0]], c_cutoff, EC_scheme)
                  for solute in solutes]

    return ([0.5*df.dot(u, u)]
            + alpha_list
            + [0.5*veps*df.dot(grad_V, grad_V)]
            + [0.5*dt**2*df.dot(grad_p, grad_p)])
