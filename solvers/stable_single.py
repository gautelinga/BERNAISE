"""This module implements the NS-coupled solver described in the paper.

GL, 2017

"""
import dolfin as df
from common.functions import max_value, alpha, alpha_c, alpha_cc, \
    alpha_reg, alpha_c_reg, absolute
from . import *
from . import __all__
import numpy as np


def get_subproblems(base_elements, solutes,
                    enable_NS, enable_EC,
                    V_lagrange, p_lagrange,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_NS:
        subproblems["NS"] = [dict(name="u", element="u"),
                             dict(name="p", element="p")]
        if p_lagrange:
            subproblems["NS"].append(dict(name="p0", element="p0"))
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
        if V_lagrange:
            subproblems["EC"].append(dict(name="V0", element="V0"))
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
    mu = df.Constant(viscosity[0])
    veps = df.Constant(permittivity[0])
    rho = df.Constant(density[0])

    if EC_scheme in ["NL1", "NL2"]:
        nonlinear_EC = True
    else:
        nonlinear_EC = False

    # Navier-Stokes
    u_ = p_ = None
    u_1 = p_1 = None
    p0 = q0 = p0_ = p0_1 = None
    if enable_NS:
        u, p = trial_functions["NS"][:2]
        v, q = test_functions["NS"][:2]

        up_ = df.split(w_["NS"])
        up_1 = df.split(w_1["NS"])
        u_, p_ = up_[:2]
        u_1, p_1 = up_1[:2]
        if p_lagrange:
            p0 = trial_functions["NS"][-1]
            q0 = test_functions["NS"][-1]
            p0_ = up_[-1]
            p0_1 = up_1[-1]

    # Electrochemistry
    c_ = V_ = c_1 = V_1 = V0 = V0_ = V0_1 = b = U = U0 = None
    if enable_EC:
        num_solutes = len(trial_functions["EC"])-1
        if V_lagrange:
            num_solutes -= 1
        assert(num_solutes == len(solutes))

        cV_ = df.split(w_["EC"])
        cV_1 = df.split(w_1["EC"])
        c_, V_ = cV_[:num_solutes], cV_[num_solutes]
        c_1, V_1 = cV_1[:num_solutes], cV_1[num_solutes]
        if V_lagrange:
            V0_ = cV_[-1]
            V0_1 = cV_1[-1]

        if not nonlinear_EC:
            c = trial_functions["EC"][:num_solutes]
            V = trial_functions["EC"][num_solutes]
            if V_lagrange:
                V0 = trial_functions["EC"][-1]
        else:
            c = c_
            V = V_
            if V_lagrange:
                V0 = V0_

        b = test_functions["EC"][:num_solutes]
        U = test_functions["EC"][num_solutes]
        if V_lagrange:
            U0 = test_functions["EC"][-1]

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
        w_NS = w_["NS"]
        dirichlet_bcs_NS = dirichlet_bcs["NS"]
        solvers["NS"] = setup_NS(**vars())
    return dict(solvers=solvers)


def setup_NS(w_NS, u, p, v, q, p0, q0,
             dx, ds, normal,
             dirichlet_bcs_NS, neumann_bcs, boundary_to_mark,
             u_1, rho, mu, c_1, grad_g_c_,
             dt, grav,
             enable_EC,
             trial_functions,
             use_iterative_solvers,
             p_lagrange,
             mesh,
             q_rhs,
             **namespace):
    """ Set up the Navier-Stokes subproblem. """
    F = (1./dt * rho * df.dot(u - u_1, v) * dx
         + rho * df.inner(df.nabla_grad(u), df.outer(u_1, v)) * dx
         + mu * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * dx
         - p * df.div(v) * dx
         - q * df.div(u) * dx
         - rho * df.dot(grav, v) * dx)

    for boundary_name, pressure in neumann_bcs["p"].iteritems():
        F += pressure * df.inner(
            normal, v) * ds(boundary_to_mark[boundary_name])

    if enable_EC:
        F += sum([ci_1*df.dot(grad_g_ci_, v)*dx
                  for ci_1, grad_g_ci_ in zip(c_1, grad_g_c_)])

    if p_lagrange:
        F += (p*q0 + q*p0)*dx

    if "u" in q_rhs:
        F += -df.dot(q_rhs["u"], v)*dx

    a, L = df.lhs(F), df.rhs(F)
    if not use_iterative_solvers:
        problem = df.LinearVariationalProblem(a, L, w_NS, dirichlet_bcs_NS)
        solver = df.LinearVariationalSolver(problem)

    else:
        solver = df.LUSolver("mumps")
        # solver.set_operator(A)
        return solver, a, L, dirichlet_bcs_NS

    return solver


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
    elif EC_scheme == ["L2"]:
        return max_value(ci_1, c_cutoff)
    else:
        return max_value(ci_1, 0.)


def setup_EC(w_EC, c, V, V0, b, U, U0,
             rho_e, grad_g_c, c_reg,
             dx, ds, normal,
             dirichlet_bcs_EC, neumann_bcs, boundary_to_mark,
             c_1, u_1, K, veps,
             rho,
             dt,
             enable_NS,
             solutes,
             use_iterative_solvers,
             nonlinear_EC,
             V_lagrange, p_lagrange,
             q_rhs,
             **namespace):
    """ Set up electrochemistry subproblem. """
    if enable_NS:
        # Projected velocity
        u_star = u_1 - dt/rho*sum([ci_1*grad_g_ci
                                   for ci_1, grad_g_ci in zip(c_1, grad_g_c)])

    F_c = []
    for ci, ci_1, bi, Ki, grad_g_ci, solute, ci_reg in zip(
            c, c_1, b, K, grad_g_c, solutes, c_reg):
        F_ci = (1./dt*(ci-ci_1)*bi*dx +
                Ki*ci_reg*df.dot(grad_g_ci, df.grad(bi))*dx)
        if enable_NS:
            F_ci += - ci_1*df.dot(u_star, df.grad(bi))*dx
        if solute[0] in q_rhs:
            F_ci += - q_rhs[solute[0]]*bi*dx
        if enable_NS:
            for boundary_name, value in neumann_bcs[solute[0]].iteritems():
                # F_ci += df.dot(u_1, normal)*bi*ci_1*ds(
                #     boundary_to_mark[boundary_name])
                pass
        F_c.append(F_ci)

    F_V = veps*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    if V_lagrange:
        F_V += veps*V0*U*dx + veps*V*U0*dx
    if "V" in q_rhs:
        F_V += q_rhs["V"]*U*dx

    F = sum(F_c) + F_V
    if nonlinear_EC:
        J = df.derivative(F, w_EC)
        problem = df.NonlinearVariationalProblem(F, w_EC, dirichlet_bcs_EC, J)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-7
        if use_iterative_solvers:
            solver.parameters["newton_solver"]["linear_solver"] = "bicgstab"
            if not V_lagrange:
                solver.parameters["newton_solver"]["preconditioner"] = "hypre_amg"
    else:
        a, L = df.lhs(F), df.rhs(F)
        problem = df.LinearVariationalProblem(a, L, w_EC, dirichlet_bcs_EC)
        solver = df.LinearVariationalSolver(problem)
        if use_iterative_solvers:
            solver.parameters["linear_solver"] = "bicgstab"
            solver.parameters["preconditioner"] = "hypre_amg"
            
    return solver


def solve(w_, t, dt, q_rhs, solvers, enable_EC, enable_NS,
          use_iterative_solvers, bcs,
          **namespace):
    """ Solve equations. """
    # Update the time-dependent source terms
    # Update the time-dependent source terms
    for qi in q_rhs.values():
        qi.t = t+dt
    # Update the time-dependent boundary conditions
    for boundary_name, bcs_fields in bcs.iteritems():
        for field, bc in bcs_fields.iteritems():
            if isinstance(bc.value, df.Expression):
                bc.value.t = t+dt

    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["EC", "NS"], [enable_EC, enable_NS]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            if subproblem == "NS" and use_iterative_solvers:
                solver, a, L, bcs = solvers[subproblem]
                A = df.assemble(a)
                b = df.assemble(L)
                for bc in bcs:
                    bc.apply(A)
                    bc.apply(b)
                solver.set_operator(A)
                solver.solve(w_["NS"].vector(), b)
            else:
                solvers[subproblem].solve()
            timer_inner.stop()

    timer_outer.stop()


def update(w_, w_1, t, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(["EC", "NS"], [enable_EC, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def alpha_generalized(c, c_cutoff, EC_scheme):
    if EC_scheme == "L2":
        return alpha_reg(c, c_cutoff)
    else:
        return alpha(c)


def discrete_energy(x_, solutes, density, permittivity,
                    c_cutoff, EC_scheme, **namespace):
    if x_ is None:
        return ["E_kin"] + ["E_{}".format(solute[0])
                            for solute in solutes] + ["E_V"]

    u = x_["u"]
    grad_V = df.grad(x_["V"])
    rho = density[0]
    veps = permittivity[0]

    alpha_list = [alpha_generalized(x_[solute[0]], c_cutoff, EC_scheme)
                  for solute in solutes]

    return ([0.5*rho*df.dot(u, u)]
            + alpha_list + [0.5*veps*df.dot(grad_V, grad_V)])


def equilibrium_EC(w_, test_functions,
                   solutes,
                   permittivity,
                   dx, ds, normal,
                   dirichlet_bcs, neumann_bcs, boundary_to_mark,
                   use_iterative_solvers,
                   V_lagrange,
                   **namespace):
    """ Electrochemistry equilibrium solver. Nonlinear! """
    num_solutes = len(solutes)

    cV = df.split(w_["EC"])
    c, V = cV[:num_solutes], cV[num_solutes]
    if V_lagrange:
        V0 = cV[-1]

    b = test_functions["EC"][:num_solutes]
    U = test_functions["EC"][num_solutes]
    if V_lagrange:
        U0 = test_functions["EC"][-1]

    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]

    for solute in solutes:
        z.append(solute[1])
        K.append(solute[2])

    rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])

    veps = permittivity[0]

    F_c = []
    for ci, bi, Ki, zi in zip(c, b, K, z):
        grad_g_ci = df.grad(alpha_c(ci) + zi*V)
        F_ci = Ki*max_value(ci, 0.)*df.dot(grad_g_ci, df.grad(bi))*dx
        F_c.append(F_ci)

    F_V = veps*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    if V_lagrange:
        F_V += veps*V0*U*dx + veps*V*U0*dx

    F = sum(F_c) + F_V
    J = df.derivative(F, w_["EC"])

    problem = df.NonlinearVariationalProblem(F, w_["EC"],
                                             dirichlet_bcs["EC"], J)
    solver = df.NonlinearVariationalSolver(problem)

    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-7
    if use_iterative_solvers:
        solver.parameters["newton_solver"]["linear_solver"] = "bicgstab"
        if not V_lagrange:
            solver.parameters["newton_solver"]["preconditioner"] = "hypre_amg"

    solver.solve()


def equilibrium_EC_PNP(w_, test_functions,
                       solutes,
                       permittivity,
                       dx, ds, normal,
                       dirichlet_bcs, neumann_bcs, boundary_to_mark,
                       use_iterative_solvers,
                       V_lagrange,
                       **namespace):
    """ Electrochemistry equilibrium solver. Nonlinear! """
    num_solutes = len(solutes)

    cV = df.split(w_["EC"])
    c, V = cV[:num_solutes], cV[num_solutes]
    if V_lagrange:
        V0 = cV[-1]

    b = test_functions["EC"][:num_solutes]
    U = test_functions["EC"][num_solutes]
    if V_lagrange:
        U0 = test_functions["EC"][-1]

    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]

    for solute in solutes:
        z.append(solute[1])
        K.append(solute[2])

    rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])

    veps_arr = np.array([100.**2, 50.**2, 20.**2,
                         10.**2, 5.**2, 2.**2, 1.**2])*permittivity[0]

    veps = df.Expression("val",
                         val=veps_arr[0],
                         degree=1)

    F_c = []
    for ci, bi, Ki, zi in zip(c, b, K, z):
        ci_grad_g_ci = df.grad(ci) + ci*zi*df.grad(V)
        F_ci = Ki*df.dot(ci_grad_g_ci, df.grad(bi))*dx
        F_c.append(F_ci)

    F_V = veps*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    if V_lagrange:
        F_V += veps*V0*U*dx + veps*V*U0*dx

    F = sum(F_c) + F_V
    J = df.derivative(F, w_["EC"])

    problem = df.NonlinearVariationalProblem(F, w_["EC"],
                                             dirichlet_bcs["EC"], J)
    solver = df.NonlinearVariationalSolver(problem)

    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-7
    if use_iterative_solvers:
        solver.parameters["newton_solver"]["linear_solver"] = "bicgstab"
        if not V_lagrange:
            solver.parameters["newton_solver"]["preconditioner"] = "hypre_amg"
    for val in veps_arr[1:]:
        veps.val = val
        solver.solve()
