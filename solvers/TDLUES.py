"""
An implementation of the scheme of
Shen and Yang, SIAM J. Numer. Anal., 2015.

Binary electrohydrodynamics solved using a partial splitting approach and
linearisation. 

The problem is split between the following subproblems.

* PF: Solved with stabilization and a modified phase field potential.

* EC: Improvised. (This is not covered by Shen and Yang)

* NSu: Velocity.

* NSp: Pressure.

GL, 2017-05-29

"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential
from common.cmd import info_red
from basic import unit_interval_filter  # GL: Move this to common.functions?
from . import *
from . import __all__
import numpy as np


def get_subproblems(base_elements, solutes,
                    enable_NS, enable_PF, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_NS:
        subproblems["NSu"] = [dict(name="u", element="u")]
        subproblems["NSp"] = [dict(name="p", element="p")]
    if enable_PF:
        subproblems["PF"] = [dict(name="phi", element="phi"),
                             dict(name="g", element="g")]
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
    return subproblems


def unpack_quantities(surface_tension, grav_const, pf_mobility_coeff,
                      pf_mobility, interface_thickness, density,
                      viscosity, permittivity, trial_functions,
                      test_functions, w_, w_1, solutes, dt, enable_EC,
                      enable_PF, enable_NS):
    """ """
    # Constant
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness
    chi = 0.5*min(density)

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

    # Phase field
    if enable_PF:
        phi, g = trial_functions["PF"]
        psi, h = test_functions["PF"]

        phi_, g_ = df.split(w_["PF"])
        phi_1, g_1 = df.split(w_1["PF"])
    else:
        # Defaults to phase 1 if phase field is disabled
        phi_ = phi_1 = 1.
        g_ = g_1 = None

    # Electrochemistry
    if enable_EC:
        num_solutes = len(trial_functions["EC"])-1
        assert(num_solutes == len(solutes))
        c = trial_functions["EC"][:num_solutes]
        V = trial_functions["EC"][num_solutes]
        b = test_functions["EC"][:num_solutes]
        U = test_functions["EC"][num_solutes]

        cV_ = df.split(w_["EC"])
        cV_1 = df.split(w_1["EC"])
        c_, V_ = cV_[:num_solutes], cV_[num_solutes]
        c_1, V_1 = cV_1[:num_solutes], cV_1[num_solutes]
    else:
        c = V = b = U = c_ = V_ = c_1 = V_1 = None

    # phi_flt_ = unit_interval_filter(phi_)
    # phi_flt_1 = unit_interval_filter(phi_1)
    phi_flt_ = phi_
    phi_flt_1 = phi_1

    M_ = pf_mobility(phi_flt_, gamma)
    M_1 = pf_mobility(phi_flt_1, gamma)
    nu_ = ramp(phi_flt_, viscosity)
    rho_ = ramp(phi_flt_, density)
    veps_ = ramp(phi_flt_, permittivity)

    rho_1 = ramp(phi_flt_1, density)
    nu_1 = ramp(phi_flt_1, viscosity)

    dveps = dramp(permittivity)
    drho = dramp(density)

    dbeta = []  # Diff. in beta
    z = []  # Charge z[species]
    K_ = []  # Diffusivity K[species]
    beta_ = []  # Conc. jump func. beta[species]

    for solute in solutes:
        z.append(solute[1])
        K_.append(ramp(phi_flt_, [solute[2], solute[3]]))
        beta_.append(ramp(phi_flt_, [solute[4], solute[5]]))
        dbeta.append(dramp([solute[4], solute[5]]))

    if enable_EC:
        rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial func.
        rho_e_ = sum([c_e_*z_e for c_e_, z_e in zip(c_, z)])  # Sum of curr. sol.
        rho_e_1 = sum([c_e_1*z_e for c_e_1, z_e in zip(c_1, z)])  # Sum of prev. sol.
    else:
        rho_e = rho_e_ = rho_e_1 = 0.

    return (sigma_bar, per_tau, grav, gamma, eps, chi, u, p, v, q, u_,
            p_, u_1, p_1, phi, g, psi, h, phi_, g_, phi_1, g_1, c, V, b, U,
            c_, V_, c_1, V_1, phi_flt_, phi_flt_1, M_, M_1, nu_, rho_, veps_,
            rho_1, nu_1, dveps, drho, dbeta, z, K_, beta_, rho_e, rho_e_,
            rho_e_1)


def setup(tstep, test_functions, trial_functions, w_, w_1,
          ds, dx,
          dirichlet_bcs, neumann_bcs, boundary_to_mark,
          permittivity, density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility, pf_mobility_coeff,
          use_iterative_solvers,
          **namespace):
    """ Set up problem. """

    (sigma_bar, per_tau, grav, gamma, eps, chi, u, p, v, q, u_, p_,
     u_1, p_1, phi, g, psi, h, phi_, g_, phi_1, g_1, c, V, b, U, c_,
     V_, c_1, V_1, phi_flt_, phi_flt_1, M_, M_1, nu_, rho_, veps_,
     rho_1, nu_1, dveps, drho, dbeta, z, K_, beta_,
     rho_e, rho_e_, rho_e_1) = unpack_quantities(
         surface_tension, grav_const,
         pf_mobility_coeff,
         pf_mobility,
         interface_thickness, density,
         viscosity, permittivity,
         trial_functions,
         test_functions, w_, w_1,
         solutes, dt, enable_EC,
         enable_PF, enable_NS)

    if tstep == 0 and enable_NS and False:
        solve_initial_pressure(w_["NSp"], p, q, u, v,
                               dx, ds,
                               dirichlet_bcs["NSp"],
                               neumann_bcs, boundary_to_mark,
                               M_, g_, phi_flt_, rho_, rho_e_, V_,
                               drho, sigma_bar, eps, grav, dveps,
                               enable_PF, enable_EC)

    solvers = dict()
    if enable_PF:
        solvers["PF"] = setup_PF(w_["PF"], phi, g, psi, h,
                                 dx, ds,
                                 dirichlet_bcs["PF"],
                                 neumann_bcs, boundary_to_mark,
                                 phi_1, u_1, M_1, c_1, V_1, rho_1,
                                 dt, sigma_bar, eps,
                                 dbeta, dveps,
                                 enable_NS, enable_EC,
                                 use_iterative_solvers)

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e,
                                 dx, ds,
                                 dirichlet_bcs["EC"],
                                 neumann_bcs, boundary_to_mark,
                                 c_1,
                                 u_1, K_, veps_, phi_flt_, rho_1,
                                 dt, z, dbeta,
                                 enable_NS, enable_PF,
                                 use_iterative_solvers)

    if enable_NS:
        solvers["NSu"] = setup_NSu(
            w_["NSu"], u, v,
            dx, ds,
            dirichlet_bcs["NSu"], neumann_bcs, boundary_to_mark,
            u_1, p_1, phi_, phi_1, rho_, rho_1, g_, g_1, c_, c_1,
            M_, nu_, nu_1, rho_e_, rho_e_1, V_,
            dt, drho, sigma_bar, eps, dveps, grav, dbeta, z,
            enable_PF, enable_EC,
            use_iterative_solvers
        )
        solvers["NSp"] = setup_NSp(w_["NSp"], p, q,
                                   dx, ds,
                                   dirichlet_bcs["NSp"],
                                   neumann_bcs, boundary_to_mark,
                                   u_, u_1, p_, p_1, rho_, dt, chi,
                                   use_iterative_solvers)

    return dict(solvers=solvers)


def setup_PF(w_PF, phi, g, psi, h,
             dx, ds,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             phi_1, u_1, M_1, c_1, V_1, rho_1,
             dt, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC,
             use_iterative_solvers):
    """ Set up phase field subproblem. """
    # Projected velocity (for energy stability)
    u_proj = u_1 - dt*phi_1*df.grad(g)/rho_1

    F_phi = (1./dt*(phi - phi_1)*psi*dx
             + M_1*df.dot(df.grad(g), df.grad(psi))*dx)
    if enable_NS:
        F_phi += df.dot(u_proj, df.grad(phi_1))*psi*dx
    F_g = (g*h*dx
           - sigma_bar/eps * (phi - phi_1) * h * dx  # Damping term (makes the system elliptic)
           - sigma_bar*eps * df.dot(df.grad(phi), df.grad(h))*dx
           - sigma_bar/eps * diff_pf_potential(phi_1)*h*dx)
    if enable_EC:
        F_g += (-sum([dbeta_i*ci_1*h*dx
                      for dbeta_i, ci_1 in zip(dbeta, c_1)])
                + 0.5*dveps*df.dot(df.grad(V_1), df.grad(V_1))*h*dx)
    F = F_phi + F_g
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_PF)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def setup_EC(w_EC, c, V, b, U, rho_e,
             dx, ds,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             c_1, u_1, K_, veps_, phi_, rho_1,
             dt, z, dbeta,
             enable_NS, enable_PF,
             use_iterative_solvers):
    """ Set up electrochemistry subproblem. """

    F_c = []
    for ci, ci_1, bi, Ki_, zi, dbetai in zip(c, c_1, b, K_, z, dbeta):
        u_proj_i = u_1 - dt/rho_1*df.grad(ci)

        F_ci = (1./dt*(ci-ci_1)*bi*dx +
                Ki_*df.dot(df.grad(ci), df.grad(bi))*dx)
        if zi != 0:
            F_ci += Ki_*zi*ci_1*df.dot(df.grad(V), df.grad(bi))*dx
            u_proj_i += -dt/rho_1 * zi * ci_1 * df.grad(V)
        if enable_PF:
            F_ci += Ki_*ci*dbetai*df.dot(df.grad(phi_), df.grad(bi))*dx
            u_proj_i += -dt/rho_1 * ci_1 * dbetai*df.grad(phi_)
        if enable_NS:
            F_ci += df.dot(u_proj_i, df.grad(ci_1))*bi*dx
        F_c.append(F_ci)
    F_V = veps_*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    F = sum(F_c) + F_V
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_EC, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def setup_NSu(w_NSu, u, v,
              dx, ds,
              dirichlet_bcs, neumann_bcs, boundary_to_mark,
              u_1, p_1, phi_, phi_1, rho_, rho_1, g_, g_1, c_, c_1,
              M_, nu_, nu_1, rho_e_, rho_e_1, V_,
              dt, drho, sigma_bar, eps, dveps, grav, dbeta, z,
              enable_PF, enable_EC,
              use_iterative_solvers):
    """ Set up the Navier-Stokes velocity subproblem. """
    mom_1 = rho_1*u_1
    if enable_PF:
        mom_1 += -drho * df.grad(g_1)
    F = (
        1./dt * rho_1 * df.dot(u - u_1, v) * dx
        + 2*nu_1*df.inner(df.sym(df.nabla_grad(u)),
                          df.sym(df.nabla_grad(v))) * dx
        - p_1 * df.div(v) * dx
        + df.inner(df.grad(u), df.outer(mom_1, v)) * dx
        + 0.5 * (1./dt * (rho_ - rho_1) + df.div(mom_1)) * df.dot(u, v) * dx
        - rho_*df.dot(grav, v) * dx
    )
    if enable_PF:
        F += phi_1 * df.dot(df.grad(g_), v) * dx
    if enable_EC and rho_e_ != 0:
        F += rho_e_1 * df.dot(df.grad(V_), v) * dx
    if enable_PF and enable_EC:
        # Not clear how to discretize this term!
        # F += dveps * df.dot(df.grad(phi_), v)*df.dot(df.grad(V_),
        #                                              df.grad(V_))*dx
        F_c = []
        for ci_, ci_1, zi, dbetai in zip(c_, c_1, z, dbeta):
            F_ci = ci_1*dbetai*df.dot(df.grad(phi_), v)*dx
            F_c.append(F_ci)
        F += sum(F_c)

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NSu, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def setup_NSp(w_NSp, p, q,
              dx, ds,
              dirichlet_bcs, neumann_bcs, boundary_to_mark,
              u_, u_1, p_, p_1, rho_, dt, chi,
              use_iterative_solvers):
    F = (
        df.dot(df.grad(q), df.grad(p - p_1)) * dx
        + chi/dt * q * df.div(2*u_ - u_1) * dx
    )
    a, L = df.lhs(F), df.rhs(F)
    problem = df.LinearVariationalProblem(a, L, w_NSp, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def solve(tstep, w_, w_1, w_tmp, solvers,
          enable_PF, enable_EC, enable_NS,
          **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(
            ["PF", "EC", "NSu", "NSp"],
            [enable_PF, enable_EC, enable_NS, enable_NS]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            solvers[subproblem].solve()
            timer_inner.stop()
    timer_outer.stop()


def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(
            ["PF", "EC", "NSu", "NSp"],
            [enable_PF, enable_EC, enable_NS, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def epsilon(u):
    return df.sym(df.nabla_grad(u))


def stress(u, p, mu):
    return 2*mu*epsilon(u) - p*df.Identity(len(u))


def solve_initial_pressure(w_NSp, p, q, u, v, dx, ds, dirichlet_bcs,
                           M_, g_, phi_, rho_, rho_e_, V_,
                           drho, sigma_bar, eps, grav, dveps,
                           enable_PF, enable_EC):
    V = u.function_space()
    grad_p = df.TrialFunction(V)
    grad_p_out = df.Function(V)
    F_grad_p = (
        df.dot(grad_p, v) * dx
        - rho_*df.dot(grav, v) * dx
    )
    if enable_PF:
        F_grad_p += - drho*M_*df.inner(df.grad(u),
                                       df.outer(df.grad(g_), v))*dx
        F_grad_p += - sigma_bar*eps*df.inner(df.outer(df.grad(phi_),
                                                      df.grad(phi_)),
                                             df.grad(v))*dx
    if enable_EC and rho_e_ != 0:
        F_grad_p += rho_e_ * df.dot(df.grad(V_), v)*dx
    if enable_PF and enable_EC:
        F_grad_p += 0.5 * dveps * df.dot(df.grad(
            phi_), v)*df.dot(df.grad(V_),
                             df.grad(V_))*dx

    info_red("Solving initial grad_p...")
    df.solve(df.lhs(F_grad_p) == df.rhs(F_grad_p), grad_p_out)

    F_p = (
        df.dot(df.grad(q), df.grad(p))*dx
        - df.dot(df.grad(q), grad_p_out)*dx
    )
    info_red("Solving initial p...")
    df.solve(df.lhs(F_p) == df.rhs(F_p), w_NSp, dirichlet_bcs)

    info_red("Done with the initials.")
