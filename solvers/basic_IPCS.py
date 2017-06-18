"""This module defines the basic solver with incremental pressure correction.

Binary electrohydrodynamics solved using a partial splitting approach and
linearisation. The problem is split between the following subproblems.

* PF: Same as basic

* EC: Same as basic

* NSu: Velocity. 

* NSp: Pressure.

GL, 2017-05-29

"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential_linearised
from common.cmd import info_red
from basic import setup_PF, setup_EC, unit_interval_filter
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


def setup(tstep, test_functions, trial_functions, w_, w_1, bcs, permittivity,
          density, viscosity,
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
        u = trial_functions["NSu"]
        p = trial_functions["NSp"]
        v = test_functions["NSu"]
        q = test_functions["NSp"]

        u_ = w_["NSu"]
        p_ = w_["NSp"]
        u_1 = w_1["NSu"]
        p_1 = w_1["NSp"]

    # Phase field
    if enable_PF:
        phi, g = trial_functions["PF"]
        psi, h = test_functions["PF"]

        phi_, g_ = df.split(w_["PF"])
        phi_1, g_1 = df.split(w_1["PF"])
    else:
        # Defaults to phase 1 if phase field is disabled
        phi_ = phi_1 = 1.

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

    M_ = pf_mobility(unit_interval_filter(phi_), gamma)
    M_1 = pf_mobility(unit_interval_filter(phi_1), gamma)
    nu_ = ramp(unit_interval_filter(phi_), viscosity)
    rho_ = ramp(unit_interval_filter(phi_), density)
    veps_ = ramp(unit_interval_filter(phi_), permittivity)

    rho_1 = ramp(unit_interval_filter(phi_1), density)

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

    if tstep == 0 and enable_NS:
        solve_initial_pressure(w_["NSp"], p, q, u, v, bcs["NSp"],
                               M_, g_, phi_, rho_, rho_e_, V_,
                               drho, sigma_bar, eps, grav, dveps,
                               enable_PF, enable_EC)

    solvers = dict()
    if enable_PF:
        solvers["PF"] = setup_PF(w_["PF"], phi, g, psi, h, bcs["PF"],
                                 phi_1, u_1, M_1, c_1, V_1,
                                 per_tau, sigma_bar, eps, dbeta, dveps,
                                 enable_NS, enable_EC,
                                 use_iterative_solvers)

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e, bcs["EC"], c_1,
                                 u_1, K_, veps_, phi_, per_tau, z, dbeta,
                                 enable_NS, enable_PF,
                                 use_iterative_solvers)

    if enable_NS:
        solvers["NSu"] = setup_NSu(
            u, v, u_, p_, bcs["NSu"],
            u_1, p_1, phi_, rho_, rho_1, g_, M_, nu_, rho_e_, V_,
            dt, drho, sigma_bar, eps, dveps, grav,
            enable_PF, enable_EC)
        solvers["NSp"] = setup_NSp(p, q, bcs["NSp"], u_, p_, p_1, rho_, dt)

    return dict(solvers=solvers)


def setup_NSu(u, v, u_, p_, bcs_NSu,
              u_1, p_1, phi_, rho_, rho_1, g_, M_, nu_, rho_e_, V_,
              dt, drho, sigma_bar, eps, dveps, grav,
              enable_PF, enable_EC):
    """ Set up the Navier-Stokes subproblem. """
    # Crank-Nicolson velocity
    # u_CN = 0.5*(u_1 + u)

    F_predict = (
        1./dt * df.sqrt(rho_) * df.dot(df.sqrt(rho_)*u - df.sqrt(rho_1)*u_1, v)*df.dx
        # + rho_*df.inner(df.grad(u), df.outer(u_1, v))*df.dx
        # + 2*nu_*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
        # - p_1 * df.div(v)*df.dx
        # + df.div(u)*q*df.dx
        + rho_*df.dot(df.dot(u_1, df.nabla_grad(u)), v)*df.dx
        + 2*nu_*df.inner(df.sym(df.grad(u)), df.sym(df.grad(v)))*df.dx
        - p_1*df.div(v)*df.dx
        - df.dot(rho_*grav, v)*df.dx
    )

    phi_filtered = unit_interval_filter(phi_)
    if enable_PF:
        F_predict += - drho*M_*df.dot(df.dot(df.nabla_grad(g_),
                                             df.nabla_grad(u)), v)*df.dx
        F_predict += - sigma_bar*eps*df.inner(df.outer(df.grad(phi_filtered),
                                                       df.grad(phi_filtered)),
                                              df.grad(v))*df.dx
    if enable_EC and rho_e_ != 0:
        F_predict += rho_e_ * df.dot(df.grad(V_), v)*df.dx
    if enable_PF and enable_EC:
        F_predict += dveps * df.dot(df.grad(
            phi_filtered), v)*df.dot(df.grad(V_),
                                     df.grad(V_))*df.dx

    # a1, L1 = df.lhs(F_predict), df.rhs(F_predict)

    F_correct = (
        df.inner(u - u_, v)*df.dx
        + dt/rho_ * df.inner(df.grad(p_ - p_1), v)*df.dx
    )
    # a3 = df.dot(u, v)*df.dx
    # L3 = df.dot(u_, v)*df.dx - dt*df.dot(df.grad(p_), v)*df.dx
    # a3, L3 = df.lhs(F_correct), df.rhs(F_correct)

    solver = dict()
    # solver["a1"] = a1
    # solver["L1"] = L1
    solver["Fu"] = F_predict
    solver["Fu_corr"] = F_correct
    # solver["a3"] = a3
    # solver["L3"] = L3
    solver["bcs"] = bcs_NSu
    return solver


def setup_NSp(p, q, bcs_NSp, u_, p_, p_1, rho_, dt):
    F_correct = (
        1./rho_ * df.dot(df.grad(p - p_1), df.grad(q)) * df.dx
        + 1./dt * df.div(u_) * q * df.dx
    )
    # a2 = df.dot(df.grad(p), df.grad(q))*df.dx
    # L2 = (df.dot(df.grad(p_1), df.grad(q))*df.dx
    #       - (1./dt)*df.div(rho_*u_)*q*df.dx)
    # L2 = -(1./dt)*df.div(u_)*q*df.dx
    # a2, L2 = df.lhs(F_correct), df.rhs(F_correct)

    solver = dict()
    # solver["a2"] = a2
    # solver["L2"] = L2
    solver["Fp"] = F_correct
    solver["bcs"] = bcs_NSp
    return solver


def solve(tstep, w_, w_1, w_tmp, solvers,
          enable_PF, enable_EC, enable_NS,
          **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["PF", "EC"], [enable_PF, enable_EC]):
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            solvers[subproblem].solve()
            timer_inner.stop()
    if enable_NS:
        # timer = df.Timer("NS: Assemble matrices")
        # A1 = df.assemble(solvers["NSu"]["a1"])
        # A2 = df.assemble(solvers["NSp"]["a2"])
        # A3 = df.assemble(solvers["NSu"]["a3"])
        # timer.stop()

        # timer = df.Timer("NS: Apply BCs 1")
        # [bc.apply(A1) for bc in solvers["NSu"]["bcs"]]
        # [bc.apply(A2) for bc in solvers["NSp"]["bcs"]]
        # timer.stop()
        du = np.array([1e9])
        tol = 1e-6
        max_num_iterations = 1
        i_iter = 0

        Fu = solvers["NSu"]["Fu"]
        bcs_u = solvers["NSu"]["bcs"]

        while du > tol and i_iter < max_num_iterations:
            print du[0]
            i_iter += 1
            du[0] = 0.
            # Step 1: Tentative velocity
            timer = df.Timer("NS: Tentative velocity")
            # b1 = df.assemble(solvers["NSu"]["L1"])
            # [bc.apply(b1) for bc in solvers["NSu"]["bcs"]]
            # df.solve(A1, w_["NSu"].vector(), b1)

            w_tmp["NSu"].vector()[:] = w_["NSu"].vector()

            # A, L = df.system(Fu)
            # df.solve(A == L, w_["NSu"], bcs_u)
            df.solve(df.lhs(Fu) == df.rhs(Fu), w_["NSu"], bcs_u)

            du[0] += df.norm(w_["NSu"].vector()-w_tmp["NSu"].vector())
            timer.stop()

            # Step 2: Pressure correction
            timer = df.Timer("NS: Pressure correction")
            # b2 = df.assemble(solvers["NSp"]["L2"])
            # [bc.apply(b2) for bc in solvers["NSp"]["bcs"]]
            # df.solve(A2, w_["NSp"].vector(), b2)

            Fp = solvers["NSp"]["Fp"]
            bcs_p = solvers["NSp"]["bcs"]
            df.solve(df.lhs(Fp) == df.rhs(Fp), w_["NSp"], bcs_p)

            w_1["NSp"].assign(w_["NSp"])

            timer.stop()

        # Step 3: Velocity correction
        timer = df.Timer("NS: Velocity correction")
        # b3 = df.assemble(solvers["NSu"]["L3"])
        # df.solve(A3, w_["NSu"].vector(), b3)
        Fu_corr = solvers["NSu"]["Fu_corr"]

        df.solve(df.lhs(Fu_corr) == df.rhs(Fu_corr), w_["NSu"], bcs_u)

        timer.stop()

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


def solve_initial_pressure(w_NSp, p, q, u, v, bcs_NSp,
                           M_, g_, phi_, rho_, rho_e_, V_,
                           drho, sigma_bar, eps, grav, dveps,
                           enable_PF, enable_EC):
    V = u.function_space()
    grad_p = df.TrialFunction(V)
    grad_p_out = df.Function(V)
    F_grad_p = (
        df.dot(grad_p, v) * df.dx
        - rho_*df.dot(grav, v)*df.dx
    )
    if enable_PF:
        F_grad_p += - drho*M_*df.inner(df.grad(u),
                                       df.outer(df.grad(g_), v))*df.dx
        F_grad_p += - sigma_bar*eps*df.inner(df.outer(df.grad(phi_),
                                                      df.grad(phi_)),
                                             df.grad(v))*df.dx
    if enable_EC and rho_e_ != 0:
        F_grad_p += rho_e_ * df.dot(df.grad(V_), v)*df.dx
    if enable_PF and enable_EC:
        F_grad_p += dveps * df.dot(df.grad(
            phi_), v)*df.dot(df.grad(V_),
                             df.grad(V_))*df.dx


    info_red("Solving initial grad_p...")
    df.solve(df.lhs(F_grad_p) == df.rhs(F_grad_p), grad_p_out)

    F_p = (
        df.dot(df.grad(q), df.grad(p))*df.dx
        - df.dot(df.grad(q), grad_p_out)*df.dx
    )
    info_red("Solving initial p...")
    df.solve(df.lhs(F_p) == df.rhs(F_p), w_NSp, bcs_NSp)
