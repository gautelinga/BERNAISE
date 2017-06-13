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
from basic import setup_PF, setup_EC, unit_interval_filter
from . import *
from . import __all__


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


def setup(test_functions, trial_functions, w_, w_1, bcs, permittivity,
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
            u_1, p_1, phi_, rho_, g_, M_, nu_, rho_e_, V_,
            dt, drho, sigma_bar, eps, dveps, grav,
            enable_PF, enable_EC)
        solvers["NSp"] = setup_NSp(p, q, bcs["NSp"], u_, p_, p_1, rho_, dt)

    return dict(solvers=solvers)


def setup_NSu(u, v, u_, p_, bcs_NSu,
              u_1, p_1, phi_, rho_, g_, M_, nu_, rho_e_, V_,
              dt, drho, sigma_bar, eps, dveps, grav,
              enable_PF, enable_EC):
    """ Set up the Navier-Stokes subproblem. """
    # Crank-Nicolson velocity
    u_CN = 0.5*(u_1 + u)
    
    F1 = (
        1./dt * rho_ * df.dot(u - u_1, v)*df.dx
        #+ rho_*df.inner(df.grad(u), df.outer(u_1, v))*df.dx
        #+ 2*nu_*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
        #- p_1 * df.div(v)*df.dx
        # + df.div(u)*q*df.dx
        + rho_*df.dot(df.dot(u_1, df.nabla_grad(u_1)), v)*df.dx
        + df.inner(stress(u_CN, p_1, nu_), epsilon(v))*df.dx
        - df.dot(rho_*grav, v)*df.dx
    )

    if enable_PF:
        F1 += - drho*M_*df.inner(df.grad(u), df.outer(df.grad(g_), v))*df.dx
        F1 += - sigma_bar*eps*df.inner(df.outer(
            df.grad(unit_interval_filter(phi_)),
            df.grad(unit_interval_filter(phi_))),
                                      df.grad(v))*df.dx
    if enable_EC and rho_e_ != 0:
        F1 += rho_e_*df.dot(df.grad(V_), v)*df.dx
    if enable_PF and enable_EC:
        F1 += dveps * df.dot(df.grad(
            unit_interval_filter(phi_)), v)*df.dot(df.grad(V_),
                                                   df.grad(V_))*df.dx

    a1, L1 = df.lhs(F1), df.rhs(F1)

    a3 = df.dot(u, v)*df.dx
    # L3 = rho_*df.dot(u_, v)*df.dx - dt*df.dot(df.grad(p_ - p_1), v)*df.dx
    L3 = df.dot(u_, v)*df.dx - dt*df.dot(df.grad(p_), v)*df.dx

    solver = dict()
    solver["a1"] = a1
    solver["L1"] = L1
    solver["a3"] = a3
    solver["L3"] = L3
    solver["bcs"] = bcs_NSu
    return solver


def setup_NSp(p, q, bcs_NSp, u_, p_, p_1, rho_, dt):

    a2 = df.dot(df.grad(p), df.grad(q))*df.dx
    # L2 = (df.dot(df.grad(p_1), df.grad(q))*df.dx
    #       - (1./dt)*df.div(rho_*u_)*q*df.dx)
    L2 = -(1./dt)*df.div(u_)*q*df.dx

    solver = dict()
    solver["a2"] = a2
    solver["L2"] = L2
    solver["bcs"] = bcs_NSp
    return solver


def solve(w_, solvers, enable_PF, enable_EC, enable_NS, **namespace):
    """ Solve equations. """
    timer_outer = df.Timer("Solve system")
    for subproblem, enable in zip(["PF", "EC"], [enable_PF, enable_EC]):
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            solvers[subproblem].solve()
            timer_inner.stop()
    if enable_NS:
        timer = df.Timer("NS: Assemble matrices")
        A1 = df.assemble(solvers["NSu"]["a1"])
        A2 = df.assemble(solvers["NSp"]["a2"])
        A3 = df.assemble(solvers["NSu"]["a3"])
        timer.stop()

        timer = df.Timer("NS: Apply BCs 1")
        [bc.apply(A1) for bc in solvers["NSu"]["bcs"]]
        [bc.apply(A2) for bc in solvers["NSp"]["bcs"]]
        timer.stop()

        # Step 1: Tentative velocity
        timer = df.Timer("NS: Tentative velocity")
        b1 = df.assemble(solvers["NSu"]["L1"])
        [bc.apply(b1) for bc in solvers["NSu"]["bcs"]]
        df.solve(A1, w_["NSu"].vector(), b1)
        timer.stop()

        # Step 2: Pressure correction
        timer = df.Timer("NS: Pressure correction")
        b2 = df.assemble(solvers["NSp"]["L2"])
        [bc.apply(b2) for bc in solvers["NSp"]["bcs"]]
        df.solve(A2, w_["NSp"].vector(), b2)
        timer.stop()

        # Step 3: Velocity correction
        timer = df.Timer("NS: Velocity correction")
        b3 = df.assemble(solvers["NSu"]["L3"])
        df.solve(A3, w_["NSu"].vector(), b3)
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
