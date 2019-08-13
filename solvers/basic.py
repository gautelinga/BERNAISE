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
    unit_interval_filter, diff_pf_contact_linearised, pf_potential, alpha
from common.io import mpi_barrier, info_red
import numpy as np
from . import *
from . import __all__


def get_subproblems(base_elements, solutes,
                    p_lagrange,
                    enable_NS, enable_PF, enable_EC,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_NS:
        subproblems["NS"] = [dict(name="u", element="u"),
                             dict(name="p", element="p")]
        if p_lagrange:
            subproblems["NS"].append(dict(name="p0", element="p0"))
    if enable_PF:
        subproblems["PF"] = [dict(name="phi", element="phi"),
                             dict(name="g", element="g")]
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
    return subproblems


def setup(mesh, test_functions, trial_functions,
          w_, w_1,
          ds, dx, normal,
          dirichlet_bcs, neumann_bcs, boundary_to_mark,
          permittivity, density, viscosity,
          solutes,
          enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const,
          grav_dir,
          friction_coeff,
          pf_mobility,
          pf_mobility_coeff,
          use_iterative_solvers, use_pressure_stabilization,
          comoving_velocity,
          p_lagrange,
          q_rhs,
          **namespace):
    """ Set up problem. """
    # Constant
    dim = mesh.geometry().dim()
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant(tuple(grav_const*np.array(grav_dir[:dim])))
    gamma = pf_mobility_coeff
    eps = interface_thickness
    fric = df.Constant(friction_coeff)
    u_comoving = df.Constant(tuple(comoving_velocity[:dim]))
    
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
    mu_ = ramp(phi_flt_, viscosity)
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
                                 dx, ds, normal,
                                 dirichlet_bcs["PF"], neumann_bcs,
                                 boundary_to_mark,
                                 phi_1, u_1, M_1, c_1, V_1,
                                 per_tau, sigma_bar, eps, dbeta, dveps,
                                 enable_NS, enable_EC,
                                 use_iterative_solvers, q_rhs)

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e,
                                 dx, ds, normal,
                                 dirichlet_bcs["EC"], neumann_bcs,
                                 boundary_to_mark,
                                 c_1, u_1, K_, veps_, phi_flt_,
                                 solutes,
                                 per_tau, z, dbeta,
                                 enable_NS, enable_PF,
                                 use_iterative_solvers,
                                 q_rhs)

    if enable_NS:
        solvers["NS"] = setup_NS(w_["NS"], u, p, v, q, p0, q0,
                                 dx, ds, normal,
                                 dirichlet_bcs["NS"], neumann_bcs,
                                 boundary_to_mark,
                                 u_1, phi_flt_,
                                 rho_, rho_1, g_, M_, mu_, rho_e_,
                                 c_, V_,
                                 c_1, V_1,
                                 dbeta, solutes,
                                 per_tau, drho, sigma_bar, eps, dveps,
                                 grav, fric,
                                 u_comoving,
                                 enable_PF, enable_EC,
                                 use_iterative_solvers,
                                 use_pressure_stabilization,
                                 p_lagrange,
                                 q_rhs)
    return dict(solvers=solvers)


def setup_NS(w_NS, u, p, v, q, p0, q0,
             dx, ds, normal,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             u_1, phi_, rho_, rho_1, g_, M_, mu_, rho_e_,
             c_, V_,
             c_1, V_1,
             dbeta, solutes,
             per_tau, drho, sigma_bar, eps, dveps, grav, fric,
             u_comoving,
             enable_PF, enable_EC,
             use_iterative_solvers, use_pressure_stabilization,
             p_lagrange,
             q_rhs):
    """ Set up the Navier-Stokes subproblem. """
    # F = (
    #     per_tau * rho_ * df.dot(u - u_1, v)*dx
    #     + rho_*df.inner(df.grad(u), df.outer(u_1, v))*dx
    #     + 2*mu_*df.inner(df.sym(df.grad(u)), df.grad(v))*dx
    #     - p * df.div(v)*dx
    #     + df.div(u)*q*dx
    #     - df.dot(rho_*grav, v)*dx
    # )
    mom_1 = rho_1*(u_1 + u_comoving)
    if enable_PF:
        mom_1 += -M_*drho * df.nabla_grad(g_)

    F = (
        per_tau * rho_1 * df.dot(u - u_1, v) * dx
        + fric*mu_*df.dot(u + u_comoving, v) * dx
        + 2*mu_*df.inner(df.sym(df.nabla_grad(u)),
                         df.sym(df.nabla_grad(v))) * dx
        - p * df.div(v) * dx
        + q * df.div(u) * dx
        + df.inner(df.nabla_grad(u), df.outer(mom_1, v)) * dx
        + 0.5 * (per_tau * (rho_ - rho_1) * df.dot(u, v)
                 - df.dot(mom_1, df.nabla_grad(df.dot(u, v)))) * dx
        - rho_*df.dot(grav, v) * dx
        - mu_ * df.dot(df.nabla_grad(u)*normal, v) * df.ds
    )
    for boundary_name, slip_length in neumann_bcs["u"].items():
        F += 1./slip_length * \
             df.dot(u, v) * ds(boundary_to_mark[boundary_name])

    for boundary_name, pressure in neumann_bcs["p"].items():
        F += pressure * df.dot(normal, v) * ds(boundary_to_mark[boundary_name])
    #          - 2*mu_*df.dot(df.dot(df.sym(df.nabla_grad(u)), v),
    #                         normal)) * ds(boundary_to_mark[boundary_name])

    if enable_PF:
        F += phi_*df.dot(df.nabla_grad(g_), v)*dx

    if enable_EC:
        for ci_, ci_1, dbetai, solute in zip(c_, c_1, dbeta, solutes):
            zi = solute[1]
            F += df.dot(df.grad(ci_), v)*dx \
                + zi*ci_1*df.dot(df.grad(V_), v)*dx
            if enable_PF:
                F += ci_*dbetai*df.dot(df.grad(phi_), v)*dx

    if p_lagrange:
        F += (p*q0 + q*p0)*dx

    if "u" in q_rhs:
        F += -df.dot(q_rhs["u"], v)*dx

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NS, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers and use_pressure_stabilization:
        solver.parameters["linear_solver"] = "gmres"
        #solver.parameters["preconditioner"] = "jacobi"
        #solver.parameters["preconditioner"] = "ilu"

    return solver


def setup_PF(w_PF, phi, g, psi, h,
             dx, ds, normal,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             phi_1, u_1, M_1, c_1, V_1,
             per_tau, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC,
             use_iterative_solvers,
             q_rhs):
    """ Set up phase field subproblem. """

    F_phi = (per_tau*(phi-unit_interval_filter(phi_1))*psi*dx +
             M_1*df.dot(df.grad(g), df.grad(psi))*dx)
    if enable_NS:
        F_phi += - phi*df.dot(u_1, df.grad(psi))*dx \
                 + phi*psi*df.dot(u_1, normal)*df.ds
        # F_phi += df.div(phi*u_1)*psi*dx
    F_g = (g*h*dx
           - sigma_bar*eps*df.dot(df.nabla_grad(phi), df.nabla_grad(h))*dx
           - sigma_bar/eps*(
               diff_pf_potential_linearised(phi,
                                            unit_interval_filter(
                                                phi_1))*h*dx))
    if enable_EC:
        F_g += (-sum([dbeta_i*ci_1*h*dx
                      for dbeta_i, ci_1 in zip(dbeta, c_1)])
                + 0.5*dveps*df.dot(df.nabla_grad(V_1), df.nabla_grad(V_1))*h*dx)

    for boundary_name, costheta in neumann_bcs["phi"].items():
        fw_prime = diff_pf_contact_linearised(phi, unit_interval_filter(phi_1))
        # Should be just surface tension!
        F_g += sigma_bar*costheta*fw_prime*h*ds(boundary_to_mark[boundary_name])

    if "phi" in q_rhs:        
        F_phi += -q_rhs["phi"]*psi*dx
    
    F = F_phi + F_g
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_PF, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def setup_EC(w_EC, c, V, b, U, rho_e,
             dx, ds, normal,
             dirichlet_bcs, neumann_bcs, boundary_to_mark,
             c_1, u_1, K_, veps_, phi_,
             solutes,
             per_tau, z, dbeta,
             enable_NS, enable_PF,
             use_iterative_solvers,
             q_rhs):
    """ Set up electrochemistry subproblem. """
    F_c = []
    for ci, ci_1, bi, Ki_, zi, dbetai, solute in zip(
            c, c_1, b, K_, z, dbeta, solutes):
        F_ci = (per_tau*(ci-ci_1)*bi*dx +
                Ki_*df.dot(df.nabla_grad(ci), df.nabla_grad(bi))*dx)
        if zi != 0:
            F_ci += Ki_*zi*ci_1*df.dot(df.nabla_grad(V), df.nabla_grad(bi))*dx

        if enable_PF:
            F_ci += Ki_*ci*dbetai*df.dot(df.nabla_grad(phi_),
                                         df.nabla_grad(bi))*dx

        if enable_NS:
            # F_ci += df.div(ci*u_1)*bi*dx
            F_ci += - ci*df.dot(u_1, df.grad(bi))*dx \
                    + ci*bi*df.dot(u_1, normal)*df.ds

        if solute[0] in q_rhs:
            F_ci += - q_rhs[solute[0]]*bi*dx

        F_c.append(F_ci)
    F_V = veps_*df.dot(df.nabla_grad(V), df.nabla_grad(U))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].items():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    if "V" in q_rhs:
        F_V += q_rhs["V"]*U*dx

    F = sum(F_c) + F_V
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_EC, dirichlet_bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def solve(w_, solvers, enable_PF, enable_EC, enable_NS,
          freeze_NSPF, **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_PF and not freeze_NSPF,
                                   enable_EC,
                                   enable_NS and not freeze_NSPF]):
        if enable:
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            mpi_barrier()
            solvers[subproblem].solve()
            timer_inner.stop()

    timer_outer.stop()


def update(t, dt, w_, w_1, bcs, bcs_pointwise,
           enable_PF, enable_EC, enable_NS, q_rhs,
           freeze_NSPF, **namespace):
    """ Update work variables at end of timestep. """
    # Update the time-dependent source terms
    for qi in q_rhs.values():
        qi.t = t+dt
    # Update the time-dependent boundary conditions
    for boundary_name, bcs_fields in bcs.items():
        for field, bc in bcs_fields.items():
            if isinstance(bc.value, df.Expression):
                bc.value.t = t+dt

    # Update fields
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_PF and not freeze_NSPF,
                                   enable_EC,
                                   enable_NS and not freeze_NSPF]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def equilibrium_EC(w_, x_, test_functions,
                   solutes,
                   permittivity,
                   mesh, dx, ds, normal,
                   dirichlet_bcs, neumann_bcs, boundary_to_mark,
                   use_iterative_solvers,
                   c_lagrange, V_lagrange,
                   **namespace):
    """ Electrochemistry equilibrium solver. Nonlinear! """
    num_solutes = len(solutes)

    cV = df.split(w_["EC"])
    c, V = cV[:num_solutes], cV[num_solutes]
    if c_lagrange:
        c0, V0 = cV[num_solutes+1:2*num_solutes+1], cV[2*num_solutes+1]
    if V_lagrange:
        V0 = cV[-1]

    b = test_functions["EC"][:num_solutes]
    U = test_functions["EC"][num_solutes]
    if c_lagrange:
        b0, U0 = cV[num_solutes+1:2*num_solutes+1], cV[2*num_solutes+1]
    if V_lagrange:
        U0 = test_functions["EC"][-1]

    phi = x_["phi"]

    q = []
    sum_zx = sum([solute[1]*xj for solute, xj in zip(solutes, composition)])
    for solute, xj in zip(solutes, composition):
        q.append(-xj*Q/(area*sum_zx))

    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]
    beta = []
    for solute in solutes:
        z.append(solute[1])
        K.append(ramp(phi, solute[2:4]))
        beta.append(ramp(phi, solute[4:6]))

    rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])
    veps = ramp(phi, permittivity)

    F_c = []
    for ci, bi, c0i, b0i, solute, qi, betai, Ki in zip(
            c, b, c0, b0, solutes, q, beta, K):
        zi = solute[1]
        F_ci = Ki*(df.dot(
            df.nabla_grad(bi),
            df.nabla_grad(ci) + df.nabla_grad(betai) + zi*ci*df.nabla_grad(V)))*dx
        if c_lagrange:
            F_ci += b0i*(ci-df.Constant(qi))*dx + c0i*bi*dx

    F_V = veps*df.dot(df.nabla_grad(U), df.nabla_grad(V))*dx
    for boundary_name, sigma_e in neumann_bcs["V"].items():
        F_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])
    if rho_e != 0:
        F_V += -rho_e*U*dx
    if V_lagrange:
        F_V += V0*U*dx + V*U0*dx

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


def discrete_energy(x_, solutes, density, permittivity,
                    c_cutoff, EC_scheme, dt,
                    density_per_concentration,
                    surface_tension, interface_thickness,
                    enable_NS, enable_PF, enable_EC,
                    **namespace):
    if x_ is None:
        E_list = []
        if enable_NS:
            E_list.append("E_kin")
        if enable_PF:
            E_list.append("E_phi")
        if enable_EC:
            E_list.extend(
                ["E_{}".format(solute[0])
                 for solute in solutes] + ["E_V"])
        return E_list

    if enable_NS:
        rho = density[0]
        veps = permittivity[0]
        u = x_["u"]
        # grad_p = df.grad(x_["p"])

    if enable_PF:
        sigma_bar = surface_tension*3./(2*math.sqrt(2))
        eps = interface_thickness
        phi = x_["phi"]
        rho = ramp(phi, density)
        veps = ramp(phi, permittivity)

    if enable_EC:
        grad_V = df.nabla_grad(x_["V"])
        M_list = []
        for solute in solutes:
            ci = x_[solute[0]]
            if enable_PF:
                betai = ramp(phi, solute[4:6])
            else:
                betai = solute[4]
            M_list.append(alpha(ci) + betai*ci)

    E_list = []
    if enable_NS:
        E_list.append(0.5*rho*df.dot(u, u))
    if enable_PF:
        E_list.append(sigma_bar/eps*pf_potential(phi)
                      + 0.5*sigma_bar*eps*df.dot(df.nabla_grad(phi),
                                                 df.nabla_grad(phi)))
    if enable_EC:
        E_list.extend(
            M_list + [0.5*veps*df.dot(grad_V, grad_V)])

    return E_list
