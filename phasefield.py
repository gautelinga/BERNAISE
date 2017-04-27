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


def initial_phasefield(S, x0, width):
    """Generate initial phase field with interface with thickness W
    located at x=x0"""
    psi_expr = df.Expression("tanh((x[0]-x0)/width)", degree=1,
                             x0=x0, width=width)


def approx_chemical_potential(phi, a, K):
    """Chemical potential g of the phase field according to
         g[psi] = 4*a*psi*(psi**2-1) - K laplacian psi .
    To be used during initialization. """
    
    return 4*a*psi - K*df.div(df.grad()


def problem():
    mesh = make_mesh()

    # Define spaces
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    S_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, S_el*S_el)

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
    psi_init = initial_phasefield(S, 2.5, 0.5)
    psi.interpolate(psi_init)
    psi0.interpolate(psi_init)
