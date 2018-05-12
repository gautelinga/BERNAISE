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
          reactions,
          V_lagrange, p_lagrange,
          density_per_concentration,
          viscosity_per_concentration,
          **namespace):
    """ Set up problem. """
    # Constant
    grav = df.Constant(tuple(grav_const*np.array(grav_dir)))
    veps = df.Constant(permittivity[0])

    mu_0 = df.Constant(viscosity[0])
    rho_0 = df.Constant(density[0])

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

    rho = rho_0
    rho_ = rho_0
    rho_1 = rho_0

    if enable_EC and density_per_concentration is not None:
        for drhodci, ci, ci_, ci_1 in zip(
                density_per_concentration, c, c_, c_1):
            if drhodci > 0.:
                rho += drhodci*ci
                rho_ += drhodci*ci_
                rho_1 += drhodci*ci_1

    mu = mu_0
    mu_ = mu_0
    mu_1 = mu_0

    if enable_EC and viscosity_per_concentration is not None:
        for dmudci, ci, ci_, ci_1 in zip(
                viscosity_per_concentration, c, c_, c_1):
            if dmudci != 0.:
                mu += dmudci*ci
                mu_ += dmudci*ci_
                mu_1 += dmudci*ci_1

    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]
    beta = []

    if enable_EC:
        for solute in solutes:
            z.append(solute[1])
            K.append(solute[2])
            beta.append(solute[4])
    else:
        z = None
        K = None
        beta = None

    if enable_EC:
        rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial func.
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of curr. sol.
    else:
        rho_e_ = None

    if enable_EC:
        g_c = []
        g_c_ = []
        grad_g_c = []
        grad_g_c_ = []
        c_reg = []
        for ci, ci_, ci_1, zi, betai in zip(c, c_, c_1, z, beta):
            g_ci = alpha_prime_approx(
                ci, ci_1, EC_scheme, c_cutoff) + betai + zi*V
            g_ci_ = alpha_prime_approx(
                ci_, ci_1, EC_scheme, c_cutoff) + betai + zi*V_
            g_c.append(g_ci)
            g_c_.append(g_ci_)
            grad_g_c.append(df.grad(g_ci))
            grad_g_c_.append(df.grad(g_ci_))
            c_reg.append(regulate(ci, ci_1, EC_scheme, c_cutoff))
    else:
        g_c = None
        g_c_ = None
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
              u_, p_, u_1, p_1, rho_, rho_1, mu_, c_1, grad_g_c_,
              dt, grav,
              enable_EC,
              trial_functions,
              use_iterative_solvers,
              mesh,
              density_per_concentration,
              viscosity_per_concentration,
              K,
              **namespace):
    """ Set up the Navier-Stokes velocity subproblem. """
    solvers = dict()

    mom_1 = rho_1 * u_1
    if enable_EC and density_per_concentration is not None:
        for drhodci, ci_1, grad_g_ci_, Ki in zip(
                density_per_concentration, c_1, grad_g_c_, K):
            if drhodci > 0.:
                mom_1 += -drhodci*Ki*ci_1*grad_g_ci_

    F_predict = (1./dt * rho_1 * df.dot(u - u_1, v) * dx
                 + df.inner(df.nabla_grad(u), df.outer(mom_1, v)) * dx
                 + 2*mu_*df.inner(df.sym(df.nabla_grad(u)),
                                  df.sym(df.nabla_grad(v))) * dx
                 + 0.5*(
                     1./dt * (rho_ - rho_1) * df.dot(u, v)
                     - df.inner(mom_1, df.grad(df.dot(u, v)))) * dx
                 - p_1 * df.div(v) * dx
                 - rho_ * df.dot(grav, v) * dx)

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
        rho_ * df.inner(u - u_, v) * dx
        - dt * (p_ - p_1) * df.div(v) * dx
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
              dt, u_, p_1, rho_0,
              use_iterative_solvers,
              **namespace):
    """ Set up Navier-Stokes pressure subproblem. """
    F = (
        df.dot(df.nabla_grad(p - p_1), df.nabla_grad(q)) * df.dx
        + 1./dt * rho_0 * df.div(u_) * q * df.dx
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


def update(t, dt, w_, w_1,
           q_rhs, bcs,
           enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    # Update the time-dependent source terms
    for qi in q_rhs.values():
        qi.t = t+dt

    # Update the time-dependent boundary conditions
    for boundary_name, bcs_fields in bcs.iteritems():
        for field, bc in bcs_fields.iteritems():
            if isinstance(bc.value, df.Expression):
                bc.value.t = t+dt

    for subproblem, enable in zip(
            ["EC", "NSu", "NSp"],
            [enable_EC, enable_NS, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def discrete_energy(x_, solutes, density, permittivity,
                    c_cutoff, EC_scheme, dt,
                    density_per_concentration,
                    enable_NS, enable_EC,
                    **namespace):
    if x_ is None:
        E_list = []
        if enable_NS:
            E_list.append("E_kin")
        if enable_EC:
            E_list.extend(
                ["E_{}".format(solute[0])
                 for solute in solutes] + ["E_V"])
        if enable_NS:
            E_list.append("E_p")
        return E_list

    if enable_NS:
        rho_0 = density[0]
        u = x_["u"]
        grad_p = df.grad(x_["p"])

    if enable_EC:
        veps = permittivity[0]
        grad_V = df.grad(x_["V"])

        alpha_list = [alpha_generalized(x_[solute[0]], c_cutoff, EC_scheme)
                      + solute[4]*x_[solute[0]]
                      for solute in solutes]

    if enable_NS:
        rho = rho_0
        if enable_EC and density_per_concentration is not None:
            for drhodci, solute in zip(density_per_concentration, solutes):
                rho += drhodci*x_[solute[0]]

    E_list = []
    if enable_NS:
        E_list.append(0.5*rho*df.dot(u, u))
    if enable_EC:
        E_list.extend(
            alpha_list + [0.5*veps*df.dot(grad_V, grad_V)])
    if enable_NS:
        E_list.append(0.5*dt**2*df.dot(grad_p, grad_p)/rho_0)

    return E_list
