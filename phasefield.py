__author__ = "Gaute Linga <gaute.linga@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"
"""Prototype of the phase field solver and basic problem.

"""
import dolfin as df


def make_mesh():
    """ Returns simple channel """
    return df.RectangleMesh(df.Point(0., 0.),
                            df.Point(5., 1),
                            50, 10)


def contact_function(psi):
    """ Function to enter into boundary function """
    return 0.
    

def initial_velocity(V):
    """ Generate uniform initial velocity """
    u_expr = df.Expression(("1.", "0."), degree=1)
    u = df.interpolate(u_expr, V)
    return u


def initial_phasefield(x0, width, psi_space):
    """Generate initial phase field with interface with thickness W
    located at x=x0"""
    psi_expr = df.Expression("tanh((x[0]-x0)/width)", degree=1,
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
    L = (4.*A*psi*(psi**2-1.)*h*dx #- K*h*contact_function(psi)*df.ds
        + K*df.dot(df.nabla_grad(h), df.nabla_grad(psi))*dx)
    w = df.Function(g_space)
    df.solve(a == L, w)
    return w


def problem():
    mesh = make_mesh()

    # Define spaces
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    S_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, S_el*S_el)
    # S = df.FunctionSpace(mesh, S_el)

    dt = 0.1
    T = 1.
    theta = 1.  # time-stepping family

    # Trial and test functions
    dw = df.TrialFunction(W)  # Is this really necessary?
    phi, h = df.TestFunctions(W)
    
    # Mixed functions
    w = df.Function(W)  # current solution
    w0 = df.Function(W)  # solution from previous step

    # Split mixed functions
    psi, g = df.split(w)
    psi0, g0 = df.split(w)

    # Initial conditions
    u = initial_velocity(V)  # Initial velocity
    psi_init = initial_phasefield(2.5, 0.2, W.sub(0).collapse())
    g_init = approx_chemical_potential(psi_init, 1., 1.,
                                       W.sub(1).collapse())
    w_init = df.project(df.as_vector((psi_init, g_init)), W)
    w.interpolate(w_init)
    w0.interpolate(w_init)

    

    
    df.plot(psi, title="psi")
    df.plot(g, title="g")
    df.interactive()


if __name__ == "__main__":
    problem()
