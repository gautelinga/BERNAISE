"""This module defines the basic solver.

Binary electrohydrodynamics solved using a partial splitting approach and
linearisation. The problem is split between the following subproblems.

* PF: The phase-field equation is solved simultaneously with the phase-field
  chemical potential (considered as a separate field), with a linearised
  double-well potential to make the problem linear.

* EC: Solute concentrations are solved simultaneously as the electric
  potential, with a linearization of the c \grad V term, to make the
  problem linear.

* NS: The Navier-Stokes equations are solved simultaneously for the
  velocity and pressure fields, where the intertial term is linearised
  to make the whole subproblem linear.

GL, 2017-05-29

"""
import dolfin as df
import math
from common.functions import ramp, dramp, diff_pf_potential_linearised
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
        u, p = trial_functions["NS"]
        v, q = test_functions["NS"]

        u_, p_ = df.split(w_["NS"])
        u_1, p_1 = df.split(w_1["NS"])

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
                                 u_1, K_, veps_, per_tau, z, enable_NS,
                                 use_iterative_solvers)

    if enable_NS:
        solvers["NS"] = setup_NS(w_["NS"], u, p, v, q, bcs["NS"], u_1, phi_,
                                 rho_, g_, M_, nu_, rho_e_, V_,
                                 per_tau, drho, sigma_bar, eps, dveps, grav,
                                 enable_PF, enable_EC,
                                 use_iterative_solvers,
                                 use_pressure_stabilization)
    return dict(solvers=solvers)


class BernaiseKrylovSolver:
    """ Define class to make it more easy...
    Doesn't work now.
    """
    def __init__(self, a, L, bcs, b):
        self.a = a
        self.L = L
        self.bcs = bcs
        self.b = b
        self.solver = df.KrylovSolver("gmres", "amg")
        self.A = df.Matrix()
        self.bb = df.Vector()
        self.P = df.Matrix()
        self.btmp = df.Vector()

    def solve(self, w_spec):
        df.assemble_system(self.a, self.L, self.bcs,
                           A_tensor=self.A, b_tensor=self.bb)
        df.assemble_system(self.b, self.L, self.bcs,
                           A_tensor=self.P, b_tensor=self.btmp)
        self.solver.set_operators(self.A, self.P)
        self.solver.solve(w_spec.vector(), self.bb)


def setup_NS(w_NS, u, p, v, q, bcs,
             u_1, phi_, rho_, g_, M_, nu_, rho_e_, V_,
             per_tau, drho, sigma_bar, eps, dveps, grav,
             enable_PF, enable_EC,
             use_iterative_solvers, use_pressure_stabilization):
    """ Set up the Navier-Stokes subproblem. """
    F = (
        per_tau * rho_ * df.dot(u - u_1, v)*df.dx
        + rho_*df.inner(df.grad(u), df.outer(u_1, v))*df.dx
        + 2*nu_*df.inner(df.sym(df.grad(u)), df.grad(v))*df.dx
        - p * df.div(v)*df.dx
        + df.div(u)*q*df.dx
        - df.dot(rho_*grav, v)*df.dx
    )
    if use_pressure_stabilization:
        mesh = w_NS.function_space().mesh()
        cellsize = df.CellSize(mesh)
        beta = 0.0008
        delta = beta*cellsize*cellsize

        F += (
            delta*df.inner(df.grad(q), df.grad(p))*df.dx
            - delta*df.dot(rho_*grav, df.grad(q))*df.dx
        )

    if enable_PF:
        F += - drho*M_*df.inner(df.grad(u), df.outer(df.grad(g_), v))*df.dx
        F += - sigma_bar*eps*df.inner(df.outer(
            df.grad(unit_interval_filter(phi_)),
            df.grad(unit_interval_filter(phi_))),
                                      df.grad(v))*df.dx
    if enable_EC:
        F += rho_e_*df.dot(df.grad(V_), v)*df.dx
    if enable_PF and enable_EC:
        F += dveps * df.dot(df.grad(
            unit_interval_filter(phi_)), v)*df.dot(df.grad(V_),
                                                   df.grad(V_))*df.dx

    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_NS, bcs)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers and use_pressure_stabilization:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "ilu"

    return solver


def setup_PF(w_PF, phi, g, psi, h, bcs,
             phi_1, u_1, M_1, c_1, V_1,
             per_tau, sigma_bar, eps,
             dbeta, dveps,
             enable_NS, enable_EC,
             use_iterative_solvers):
    """ Set up phase field subproblem. """

    F_phi = (per_tau*(phi-unit_interval_filter(phi_1))*psi*df.dx +
             M_1*df.dot(df.grad(g), df.grad(psi))*df.dx)
    if enable_NS:
        F_phi += df.dot(u_1, df.grad(phi))*psi*df.dx
    F_g = (g*h*df.dx
           - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*df.dx
           - sigma_bar/eps*(
               diff_pf_potential_linearised(phi,
                                            unit_interval_filter(
                                                phi_1))*h*df.dx))
    if enable_EC:
        F_g += (-sum([dbeta_i*ci_1*h*df.dx
                      for dbeta_i, ci_1 in zip(dbeta, c_1)])
                + dveps*df.dot(df.grad(V_1), df.grad(V_1))*h*df.dx)
    F = F_phi + F_g
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_PF)
    solver = df.LinearVariationalSolver(problem)

    if use_iterative_solvers:
        solver.parameters["linear_solver"] = "gmres"
        # solver.parameters["preconditioner"] = "hypre_euclid"

    return solver


def setup_EC(w_EC, c, V, b, U, rho_e, bcs,
             c_1, u_1, K_, veps_,
             per_tau, z,
             enable_NS,
             use_iterative_solvers):
    """ Set up electrochemistry subproblem. """
    F_c = []
    for ci, ci_1, bi, Ki_, zi in zip(c, c_1, b, K_, z):
        F_ci = (per_tau*(ci-ci_1)*bi*df.dx
                + Ki_*df.dot(df.grad(ci), df.grad(bi))*df.dx
                + zi*ci_1*df.dot(df.grad(V), df.grad(bi))*df.dx)
        if enable_NS:
            F_ci += df.dot(u_1, df.grad(ci))*bi*df.dx
        F_c.append(F_ci)
    F_V = (veps_*df.dot(df.grad(V), df.grad(U))*df.dx + rho_e*U*df.dx)
    F = sum(F_c) + F_V
    a, L = df.lhs(F), df.rhs(F)

    problem = df.LinearVariationalProblem(a, L, w_EC, bcs)
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
            timer_inner = df.Timer("Solve subproblem " + subproblem)
            df.mpi_comm_world().barrier()
            solvers[subproblem].solve()
            timer_inner.stop()

    timer_outer.stop()


def update(w_, w_1, enable_PF, enable_EC, enable_NS, **namespace):
    """ Update work variables at end of timestep. """
    for subproblem, enable in zip(["PF", "EC", "NS"],
                                  [enable_NS, enable_EC, enable_PF]):
        if enable:
            w_1[subproblem].assign(w_[subproblem])


def max_value(a, b):
    return 0.5*(a+b+df.sign(a-b)*(a-b))


def min_value(a, b):
    return 0.5*(a+b-df.sign(a-b)*(a-b))


def unit_interval_filter(phi):
    return min_value(max_value(phi, -1.), 1.)
