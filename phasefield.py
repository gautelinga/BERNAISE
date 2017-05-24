__author__ = "Gaute Linga <gaute.linga@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"
"""Prototype of the phase field solver and basic problem.

"""
import dolfin as df
import numpy as np


# Form compiler options
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "quadrature"


class PhaseFieldEquation(df.NonlinearProblem):
    def __init__(self, a, L, bcs):
        df.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def F(self, b, x):
        df.assemble(self.L, tensor=b)

    def J(self, A, x):
        df.assemble(self.a, tensor=A)


def make_mesh():
    """ Returns simple channel """
    return df.RectangleMesh(df.Point(0., 0.),
                            df.Point(5., 1.),
                            200, 40)


def contact_function(psi):
    """ Function to enter into boundary function """
    return 0.
    

def initial_velocity(V):
    """ Generate uniform initial velocity """
    u_expr = df.Expression((".01", "0."), degree=1)
    u = df.interpolate(u_expr, V)
    return u


def initial_phasefield(x0, width, psi_space):
    """Generate initial phase field with interface with thickness W
    located at x=x0"""
    psi_expr = df.Expression("tanh(2*(x[0]-x0)/width)", degree=1,
                             x0=x0, width=width)
    psi = df.interpolate(psi_expr, psi_space)
    return psi


def approx_chemical_potential(psi, A, K, g_space):
    """Chemical potential g corresponding to the phase field psi
    according to
         g[psi] = 4*A*psi*(psi**2-1) - K laplacian psi .
    To be used during initialization. """
    # Test and trial functions
    h = df.TestFunction(g_space)
    g = df.TrialFunction(g_space)
    # Set up LHS and RHS
    dx = df.dx
    a = h*g*dx
    L = (4.*A*psi*(psi**2-1.)*h*dx  # - K*h*contact_function(psi)*df.ds
         + K*df.dot(df.nabla_grad(h), df.nabla_grad(psi))*dx)
    w = df.Function(g_space)
    df.solve(a == L, w)
    return w


def problem():
    mesh = make_mesh()
    
    # Define spaces
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    S_psi_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    S_g_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, S_psi_el*S_g_el)
    # S = df.FunctionSpace(mesh, S_el)

    dt = 1e-3
    t0 = 0.
    T = 1.
    sigma = 1.
    delta = 0.3
    A = 3*sigma/(4*delta)
    K = 3*sigma*delta/8.
    M = 1.
    print "A =", A
    print "K =", K
    
    theta = 1.0  # time-stepping family

    # Trial and test functions
    dw = df.TrialFunction(W)  # Is this really necessary?
    phi, h = df.TestFunctions(W)
    
    # Mixed functions
    w = df.Function(W)  # current solution
    w0 = df.Function(W)  # solution from previous step

    # Split mixed functions
    dpsi, dg = df.split(dw)  # Necessary?
    psi, g = df.split(w)
    psi0, g0 = df.split(w)

    # Initial conditions
    u0 = initial_velocity(V)  # Initial velocity
    u = df.Function(V)
    u.interpolate(u0)
    
    psi_init = initial_phasefield(2.5, delta, W.sub(0).collapse())
    g_init = approx_chemical_potential(psi_init, A, K,
                                       W.sub(1).collapse())
    w_init = df.project(df.as_vector((psi_init, g_init)), W)
    w.interpolate(w_init)
    w0.interpolate(w_init)

    psi_mid = theta*psi + (1.-theta)*psi0
    g_mid = theta*g + (1.-theta)*g0

    dx = df.dx
    ds = df.ds

    # Variational forms
    L0 = (phi*psi*dx - phi*psi0*dx
          + dt*phi*df.dot(u, df.grad(psi0))*dx
          + dt*M*df.dot(df.grad(phi), df.grad(g))*dx)
    L1 = (h*g*dx - 4*A*h*psi_mid*(psi_mid**2-1.)*dx
          - K*df.dot(df.grad(h), df.grad(psi_mid))*dx)
          # + K*h*contact_function(psi)*ds)
    L = L0 + L1
    a = df.derivative(L, w, dw)
    bc_psi_left = df.DirichletBC(W.sub(0), df.Constant(-1.),
                             "x[0] < DOLFIN_EPS && on_boundary")
    bc_psi_right = df.DirichletBC(W.sub(0), df.Constant(1.),
                              "x[0] > 4.9-DOLFIN_EPS && on_boundary")
    bc_g_left = df.DirichletBC(W.sub(1), df.Constant(0.),
                               "x[0] < DOLFIN_EPS && on_boundary")
    bc_g_right = df.DirichletBC(W.sub(1), df.Constant(0.),
                                "x[0] > 4.9-DOLFIN_EPS && on_boundary")
    bcs = [bc_psi_left, bc_g_left]

    # problem = PhaseFieldEquation(a, L, bcs)
    # solver = df.NewtonSolver()
    problem = df.NonlinearVariationalProblem(L, w, J=a, bcs=bcs)
    solver = df.NonlinearVariationalSolver(problem)
    # solver.parameters["linear_solver"] = "lu"
    # solver.parameters["convergence_criterion"] = "incremental"
    # solver.parameters["relative_tolerance"] = 1e-6

    t = t0
    
    xdmff = df.XDMFFile(mesh.mpi_comm(), "pf.xdmf")
    while t < 1*dt:
        xdmff.write(w, t)
        t += dt
        w0.vector()[:] = w.vector()

        # solver.solve(problem, w.vector())
        solver.solve()
        
    if True:
        df.plot(psi, title="psi")
        df.plot(g, title="g")
        df.plot(df.dot(u, df.grad(psi)), title="grad_psi")
        df.interactive()


if __name__ == "__main__":
    problem()
