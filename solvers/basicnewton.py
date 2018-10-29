"""
This module defines the basic Newton solver.

Binary electrohydrodynamics solved using monolithic Newton solver 
and with a implicit Euler time integartion. 

AB, 2017-06-1 (based on basic.py)
GL, 2018-03 
"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential, diff_pf_contact,\
    unit_interval_filter, max_value
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


def setup(test_functions, trial_functions, w_, w_1,
          dx, ds, normal,
          dirichlet_bcs, neumann_bcs,
          boundary_to_mark,
          permittivity,
          density, viscosity,
          solutes, enable_PF, enable_EC, enable_NS,
          surface_tension, dt, interface_thickness,
          grav_const, pf_mobility_coeff, pf_mobility,
          q_rhs,
          use_iterative_solvers,
          p_lagrange,
          **namespace):
    """ Set up problem. """
    # Constants
    sigma_bar = surface_tension*3./(2*math.sqrt(2))
    per_tau = df.Constant(1./dt)
    grav = df.Constant((0., -grav_const))
    gamma = pf_mobility_coeff
    eps = interface_thickness

    # Set up the fields
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

        if p_lagrange:
            q0 = test_functions["NSPFEC"][field_number]
            p0_ = funs_[field_number]
            p0_1 = funs_1[field_number]
            field_number += 1
        else:
            q0 = p0_ = p0_1 = None
    else:
        v = u_ = u_1 = q = q0 = p0_ = p0_1 = None
    if enable_PF:
        psi = test_functions["NSPFEC"][field_number]
        phi_ = funs_[field_number]
        phi_1 = funs_1[field_number]
        field_number += 1
        h = test_functions["NSPFEC"][field_number]
        g_ = funs_[field_number]
        g_1 = funs_1[field_number]
        field_number += 1
    else:
        psi = phi_ = phi_1 = h = g_ = g_1 = 1
    if enable_EC:
        num_solutes = len(test_functions["NSPFEC"])-field_number-1
        b = test_functions["NSPFEC"][field_number:(num_solutes+field_number)]
        c_ = funs_[field_number:(num_solutes+field_number)]
        c_1 = funs_1[field_number:(num_solutes+field_number)]
        U = test_functions["NSPFEC"][num_solutes+field_number]
        V_ = funs_[num_solutes+field_number]
        V_1 = funs_1[num_solutes+field_number]
    else:
        b = c_ = c_1 = U = V_ = V_1 = rho_e_ = 0

    M_ = pf_mobility(phi_, gamma)
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

    if enable_EC:
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of current sol.
        rho_e_1 = sum([c_e*z_e for c_e, z_e in zip(c_1, z)])  # Sum of current sol.

    solver = dict()
    solver["NSPFEC"] = setup_NSPFEC(w_["NSPFEC"], w_1["NSPFEC"],
                                    dirichlet_bcs["NSPFEC"],
                                    neumann_bcs,
                                    boundary_to_mark,
                                    dx, ds, normal,
                                    v, q, q0, psi, h, b, U,
                                    u_, p_, p0_, phi_, g_, c_, V_,
                                    u_1, p_1, p0_1, phi_1, g_1, c_1, V_1,
                                    M_, nu_, veps_, rho_, K_, beta_, rho_e_,
                                    dbeta, dveps, drho,
                                    per_tau, sigma_bar, eps, grav, z,
                                    solutes,
                                    enable_NS, enable_PF, enable_EC,
                                    use_iterative_solvers,
                                    p_lagrange,
                                    q_rhs)
    return dict(solvers=solver)


def setup_NSPFEC(w_NSPFEC, w_1NSPFEC,
                 dirichlet_bcs_NSPFEC,
                 neumann_bcs,
                 boundary_to_mark,
                 dx, ds, normal,
                 v, q, q0, psi, h, b, U,
                 u_, p_, p0_, phi_, g_, c_, V_,
                 u_1, p_1, p0_1, phi_1, g_1, c_1, V_1,
                 M_, nu_, veps_, rho_, K_, beta_, rho_e_,
                 dbeta, dveps, drho,
                 per_tau, sigma_bar, eps, grav, z,
                 solutes,
                 enable_NS, enable_PF, enable_EC,
                 use_iterative_solvers,
                 p_lagrange,
                 q_rhs):
    """ The full problem of electrohydrodynamics in two phases.
    Note that it is possible to turn off the different parts at will.
    """
    # Setup of the Navier-Stokes part of F
    mom_ = rho_*u_
    if enable_PF:
        mom_ += -M_*drho * df.nabla_grad(g_)
    
    F = []
    if enable_NS:
        F_NS = (per_tau * rho_ * df.dot(u_ - u_1, v) * dx
                + df.inner(df.nabla_grad(u_), df.outer(mom_, v)) * dx
                + 2*nu_*df.inner(df.sym(df.nabla_grad(u_)),
                                 df.sym(df.nabla_grad(v))) * dx
                - p_ * df.div(v) * dx
                - df.div(u_) * q * dx
                - df.dot(rho_ * grav, v) * dx)
        # if enable_PF:
        #     F_NS += - sigma_bar*eps*df.inner(
        #         df.outer(df.grad(phi_),
        #                  df.grad(phi_)), df.grad(v)) * dx
        # if enable_EC and rho_e_ != 0:
        #     F_NS += rho_e_*df.dot(df.grad(V_), v) * dx
        # if enable_PF and enable_EC:
        #     F_NS += dveps*df.dot(
        #         df.grad(phi_), v)*df.dot(df.grad(V_),
        #                                  df.grad(V_)) * dx

        if enable_PF:
            F_NS += phi_*df.dot(df.grad(g_), v) * dx

        if enable_EC:
            for ci_, dbetai, solute in zip(c_, dbeta, solutes):
                zi = solute[1]
                F_NS += df.dot(df.grad(ci_), v) * dx \
                        + ci_*dbetai*df.dot(df.grad(phi_), v) * dx \
                        + zi*ci_*df.dot(df.grad(V_), v) * dx

        # Slip boundary condition
        for boundary_name, slip_length in neumann_bcs["u"].items():
            F_NS += 1./slip_length * \
                    df.dot(u_, v) * ds(boundary_to_mark[boundary_name])

        # Pressure boundary condition
        for boundary_name, pressure in neumann_bcs["p"].items():
            F_NS += pressure * df.inner(
                normal, v) * ds(boundary_to_mark[boundary_name])

        # Lagrange pressure
        if p_lagrange:
            F_NS += (p_*q0 + q*p0_)*dx

        # RHS source terms
        if "u" in q_rhs:
            F_NS += -df.dot(q_rhs["u"], v)*dx

        F.append(F_NS)

    # Setup of the phase-field equations
    if enable_PF:
        phi_1_flt = unit_interval_filter(phi_1)
        F_PF_phi = (per_tau*(phi_-phi_1_flt)*psi*df.dx +
                    M_*df.dot(df.grad(g_), df.grad(psi)) * dx)
        if enable_NS:
            F_PF_phi += df.dot(u_, df.grad(phi_)) * psi * dx

        F_PF_g = (g_ * h * dx
                  - sigma_bar*eps*df.dot(df.grad(phi_), df.grad(h)) * dx
                  - sigma_bar/eps*diff_pf_potential(phi_) * h * dx)
        if enable_EC:
            F_PF_g += (-sum([dbeta_i * ci_ * h * dx
                             for dbeta_i, ci_ in zip(dbeta, c_)])
                       + dveps * df.dot(df.grad(V_), df.grad(V_)) * h * dx)

        # Contact angle boundary condtions
        for boundary_name, costheta in neumann_bcs["phi"].items():
            fw_prime = diff_pf_contact(phi_)
            F_PF_g += sigma_bar * costheta * fw_prime * h * ds(
                boundary_to_mark[boundary_name])

        # RHS source terms
        if "phi" in q_rhs:
            F_PF_phi += -q_rhs["phi"]*psi*dx

        F_PF = F_PF_phi + F_PF_g
        F.append(F_PF)

    # Setup of the electrochemistry
    if enable_EC:
        F_E_c = []
        for ci_, ci_1, bi, Ki_, zi, dbetai, solute in zip(
                c_, c_1, b, K_, z, dbeta, solutes):
            ci_1_flt = max_value(ci_1, 0.)
            F_E_ci = (per_tau*(ci_-ci_1_flt)*bi*df.dx
                       + Ki_*df.dot(df.grad(ci_), df.grad(bi))*df.dx)
            if zi != 0:
                F_E_ci += Ki_*zi*ci_*df.dot(df.grad(V_),
                                             df.grad(bi))*df.dx
            if enable_NS:
                F_E_ci += df.dot(u_, df.grad(ci_))*bi*df.dx

            if enable_PF:
                F_E_ci += Ki_*ci_*dbetai*df.dot(df.grad(phi_),
                                                df.grad(bi)) * dx

            if solute[0] in q_rhs:
                F_E_ci += - q_rhs[solute[0]] * bi * dx

            F_E_c.append(F_E_ci)

        F_E_V = veps_*df.dot(df.grad(V_), df.grad(U))*df.dx

        if rho_e_ != 0:
            F_E_V += -rho_e_*U*df.dx

        # Surface charge boundary condition
        for boundary_name, sigma_e in neumann_bcs["V"].items():
            F_E_V += -sigma_e*U*ds(boundary_to_mark[boundary_name])

        # RHS source terms
        if "V" in q_rhs:
            F_E_V += q_rhs["V"]*U*dx

        F_E = sum(F_E_c) + F_E_V

        F.append(F_E)

    F = sum(F)

    J = df.derivative(F, w_NSPFEC)
    problem_NSPFEC = df.NonlinearVariationalProblem(F, w_NSPFEC, dirichlet_bcs_NSPFEC, J)
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

