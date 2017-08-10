"""This module defines the basic solver.

Binary electrohydrodynamics solved using a partial splitting approach and
linearisation. The problem is split between the following subproblems.

* PF: The phase-field equation is solved simultaneously with the phase-field
  chemical potential (considered as a separate field), with a linearised
  double-well potential to make the problem linear.

* EC: Solute concentrations are solved simultaneously as the electric
  potential, with a linearization of the c \grad V term, to make the whole
  problem linear.

* NS: The Navier-Stokes equations are solved simultaneously for the
  velocity and pressure fields, where the intertial term is linearised
  to make the whole subproblem linear.

GL, 2017-05-29

"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential_linearised, \
    unit_interval_filter
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


def setup(test_functions, trial_functions, w_, w_1,
          ds, dx, normal,
          dirichlet_bcs, neumann_bcs, boundary_to_mark,
          permittivity, density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility, pf_mobility_coeff,
          use_iterative_solvers, use_pressure_stabilization,
          **namespace):
    """ Set up problem. """
    # Constant
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness

    # Navier-Stokes
    if enable_NS:
        u, p = trial_functions["NS"]
        v, q = test_functions["NS"]

        u_, p_ = df.split(w_["NS"])
        u_1, p_1 = df.split(w_1["NS"])
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
        c_ = V_ = c_1 = V_1 = None

    phi_flt_ = unit_interval_filter(phi_)
    phi_flt_1 = unit_interval_filter(phi_1)

    M_ = pf_mobility(phi_flt_, gamma)
    M_1 = pf_mobility(phi_flt_1, gamma)
    nu_ = ramp(phi_flt_, viscosity)
    rho_ = ramp(phi_flt_, density)
    rho_1 = ramp(phi_flt_1, density)
    veps_ = ramp(phi_flt_, permittivity)

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

    if enable_EC:
        rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial func.
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of curr. sol.
    else:
        rho_e_ = None

    solvers = dict()
    if enable_PF:
        solvers["PF"] = setup_PF(w_["PF"], phi, g, psi, h,
                                 dx, ds,
                                 dirichlet_bcs["PF"], neumann_bcs,
                                 boundary_to_mark,
                                 phi_1, u_1, M_1, c_1, V_1,
                                 per_tau, sigma_bar, eps, dbeta, dveps,
                                 enable_NS, enable_EC,
                                 use_iterative_solvers)

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e,
                                 dx, ds,
                                 dirichlet_bcs["EC"], neumann_bcs,
                                 boundary_to_mark,
                                 c_1, u_1, K_, veps_, phi_flt_,
                                 per_tau, z, dbeta,
                                 enable_NS, enable_PF,
                                 use_iterative_solvers)

    if enable_NS:
        solvers["NS"] = setup_NS(w_["NS"], u, p, v, q,
                                 dx, ds, normal,
                                 dirichlet_bcs["NS"], neumann_bcs,
                                 boundary_to_mark,
                                 u_1, phi_flt_,
                                 rho_, rho_1, g_, M_, nu_, rho_e_, V_,
                                 per_tau, drho, sigma_bar, eps, dveps, grav,
                                 enable_PF, enable_EC,
                                 use_iterative_solvers,
                                 use_pressure_stabilization)
    return dict(solvers=solvers)


def setup_NS(w_NS, u, p, v, q,
             dx, ds, normal,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             u_1, phi_, rho_, rho_1, g_, M_, nu_, rho_e_, V_,
             per_tau, drho, sigma_bar, eps, dveps, grav,
             enable_PF, enable_EC,
             use_iterative_solvers, use_pressure_stabilization):
    """ Set up the Navier-Stokes subproblem. """
    # F = (
    #     per_tau * rho_ * df.dot(u - u_1, v)*dx
    #     + rho_*df.inner(df.grad(u), df.outer(u_1, v))*dx
    #     + 2*nu_*df.inner(df.sym(df.grad(u)), df.grad(v))*dx
    #     - p * df.div(v)*dx
    #     + df.div(u)*q*dx
    #     - df.dot(rho_*grav, v)*dx
    # )
    mom_1 = rho_1*u_1
    if enable_PF:
        mom_1 += -M_*drho * df.grad(g_)
    F = (
        per_tau * rho_1 * df.dot(u - u_1, v) * dx
        + 2*nu_*df.inner(df.sym(df.nabla_grad(u)),
                         df.sym(df.nabla_grad(v))) * dx
        - p * df.div(v) * dx
        - q * df.div(u) * dx
        + df.inner(df.grad(u), df.outer(mom_1, v)) * dx
        + 0.5 * (per_tau * (rho_ - rho_1) + df.div(mom_1)) * df.dot(u, v) * dx
        - rho_*df.dot(grav, v) * dx
    )
    for boundary_name, pressure in neumann_bcs["p"].iteritems():
        F += pressure * df.inner(
            normal, v) * ds(boundary_to_mark[boundary_name])
    if use_pressure_stabilization:
        mesh = w_NS.function_space().mesh()
        cellsize = df.CellSize(mesh)
        beta = 0.0008
        delta = beta*cellsize*cellsize

        F += (
            delta*df.inner(df.grad(q), df.grad(p))*dx
            - delta*df.dot(rho_*grav, df.grad(q))*dx
        )

    if enable_PF:
        # F += - drho*M_*df.inner(df.grad(u), df.outer(df.grad(g_), v))*dx
        # GL: Wasn't filter applied outside?
        F += - sigma_bar*eps*df.inner(df.outer(
            df.grad(unit_interval_filter(phi_)),
            df.grad(unit_interval_filter(phi_))),
                                      df.grad(v))*dx
    if enable_EC and rho_e_ != 0:
        F += rho_e_*df.dot(df.grad(V_), v)*dx
    if enable_PF and enable_EC:
        F += dveps * df.dot(df.grad(
            unit_interval_filter(phi_)), v)*df.dot(df.grad(V_),
                                                   df.grad(V_))*dx

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NS, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers and use_pressure_stabilization:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "ilu"

    return solver


def setup_PF(w_PF, phi, g, psi, h,
             dx, ds,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             phi_1, u_1, M_1, c_1, V_1,
             per_tau, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC,
             use_iterative_solvers):
    """ Set up phase field subproblem. """

    F_phi = (per_tau*(phi-unit_interval_filter(phi_1))*psi*dx +
             M_1*df.dot(df.grad(g), df.grad(psi))*dx)
    if enable_NS:
        F_phi += df.dot(u_1, df.grad(phi))*psi*dx
    F_g = (g*h*dx
           - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*dx
           - sigma_bar/eps*(
               diff_pf_potential_linearised(phi,
                                            unit_interval_filter(
                                                phi_1))*h*dx))
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
             c_1, u_1, K_, veps_, phi_,
             per_tau, z, dbeta,
             enable_NS, enable_PF,
             use_iterative_solvers):
    """ Set up electrochemistry subproblem. """
    F_c = []
    for ci, ci_1, bi, Ki_, zi, dbetai in zip(c, c_1, b, K_, z, dbeta):
        F_ci = (per_tau*(ci-ci_1)*bi*dx +
                Ki_*df.dot(df.grad(ci), df.grad(bi))*dx)
        if zi != 0:
            F_ci += Ki_*zi*ci_1*df.dot(df.grad(V), df.grad(bi))*dx
            #for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
            #    F_ci += -Ki_*zi*ci_1*sigma_e/veps_*bi*ds(boundary_to_mark[boundary_name])
        if enable_PF:
            F_ci += Ki_*ci*dbetai*df.dot(df.grad(phi_), df.grad(bi))*dx
            #for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
            #     F_ci += -Ki_*ci*dbetai*df.dot(df.grad(phi_), normal)*bi*ds(boundary_to_mark[boundary_name])
        if enable_NS:
            F_ci += df.dot(u_1, df.grad(ci))*bi*dx
        F_c.append(F_ci)
    F_V = veps_*df.dot(df.grad(V), df.grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].iteritems():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
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


def solve(w_, solvers, enable_PF, enable_EC, enable_NS, **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_PF, enable_EC, enable_NS]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            solvers[subproblem].solve()
            timer_inner.stop()

    timer_outer.stop()


def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_PF, enable_EC, enable_NS]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])
