import dolfin as df
import math


df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["representation"] = "quadrature"
df.parameters["linear_algebra_backend"] = "PETSc"
df.parameters["std_out_all_processes"] = False
df.parameters["krylov_solver"]["nonzero_initial_guess"] = True


def get_mesh(Lx, Ly, h):
    return df.RectangleMesh(df.Point(0., 0.),
                            df.Point(Lx, Ly),
                            int(Lx/h), int(Ly/h))


def density(phi, rho_1, rho_2):
    return rho_1*0.5*(1.+phi) + rho_2*0.5*(1.-phi)


def viscosity(phi, nu_1, nu_2):
    return nu_1*0.5*(1.+phi) + nu_2*0.5*(1.-phi)


def pf_mobility(phi, gamma):
    return gamma * (phi**2-1.)**2


def pf_potential(phi):
    return 0.25*(phi**2-1.)**2


def diff_pf_potential_linearised(phi, phi0):
    return phi0**3-phi0+(3*phi0**2-1.)*(phi-phi0)


def diff_pf_potential(phi):
    return phi*(phi**2-1.)


def initial_phasefield(x0, y0, rad, eps, S, shape="circle"):
    if shape == "flat":
        expr_str = "tanh((x[1]-y0)/(sqrt(2)*eps))"
    elif shape == "circle":
        expr_str = "tanh(sqrt(2)*(sqrt(pow(x[0]-x0, 2) + pow(x[1]-y0, 2)) - rad)/eps)"
    else:
        exit("Unrecognized shape: " + shape)
    phi_init_expr = df.Expression(expr_str, x0=x0, y0=y0, rad=rad, eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, S)
    return phi_init


def filter_pf(phi):
    return phi  # 0.5*phi*(3.-phi**2)


def approx_chem_pot(phi, sigma_bar, eps):
    g_space = phi.function_space()
    h = df.TestFunction(g_space)
    g = df.TrialFunction(g_space)
    dx = df.dx
    a = h*g*dx
    L = sigma_bar * (1./eps * diff_pf_potential(phi)*h*dx
                     # - eps*h*contact_function(psi)*df.ds
                     + eps * df.dot(df.nabla_grad(h), df.nabla_grad(phi))*dx)
    g_out = df.Function(g_space)
    df.solve(a == L, g_out)
    return g_out


def problem():
    factor = 1./2.
    h_int = factor * 1./16
    eps = factor * 0.040
    dt = factor * 0.008
    gamma = factor * 0.000040

    Lx, Ly = 1., 2.
    rad_init = 0.25
    t0 = 0.
    T = 6.

    rho_1 = 1000.
    nu_1 = 10.
    rho_2 = 100.
    nu_2 = 1.

    sigma = 24.5
    grav_const = 0.98

    mesh = get_mesh(Lx, Ly, h_int)
    sigma_bar = sigma*3./(2*math.sqrt(2))
    grav = df.Constant((0., -grav_const))
    tau = df.Constant(dt)

    u_el = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    p_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    phi_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    g_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    c_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V_el = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    
    W_NS = df.FunctionSpace(mesh, u_el * p_el)
    W_PF = df.FunctionSpace(mesh, phi_el * g_el)

    phi_init = initial_phasefield(Lx/2., Lx/2., rad_init, eps,
                                  W_PF.sub(0).collapse())
    g_init = approx_chem_pot(phi_init, sigma_bar, eps)
    w_PF_init = df.project(df.as_vector((phi_init, g_init)), W_PF)

    u_init = df.Function(W_NS.sub(0).collapse())
    u_init.vector()[:] = 0.
    ux_init, uy_init = u_init.split()
    p_init = df.Function(W_NS.sub(1).collapse())
    p_init.vector()[:] = 0.
    w_NS_init = df.project(df.as_vector((ux_init, uy_init, p_init)), W_NS)
    # w_NS_init = df.project(
    
    # Velocity
    w1_NS = df.Function(W_NS)  # Current timestep
    w0_NS = df.Function(W_NS)  # previous timestep
    u0, p0 = df.split(w0_NS)
    u1, p1 = df.split(w1_NS)
    w0_NS.interpolate(w_NS_init)

    u, p = df.TrialFunctions(W_NS)
    v, q = df.TestFunctions(W_NS)

    w1_PF = df.Function(W_PF)  # Current timestep
    w0_PF = df.Function(W_PF)  # Previous timestep
    phi0, g0 = df.split(w0_PF)
    phi1, g1 = df.split(w1_PF)
    w0_PF.interpolate(w_PF_init)

    phi, g = df.TrialFunctions(W_PF)
    psi, h = df.TestFunctions(W_PF)

    rho1 = density(phi1, rho_1, rho_2)
    rho0 = density(filter_pf(phi0), rho_1, rho_2)
    nu1 = viscosity(phi1, nu_1, nu_2)
    nu0 = viscosity(filter_pf(phi0), nu_1, nu_2)
    M1 = pf_mobility(phi1, gamma)
    M0 = pf_mobility(phi0, gamma)
    
    dx = df.dx

    # phi, g = df.split(w1)
    F_PF = ((phi-filter_pf(phi0))*psi*dx + tau*df.dot(u0, df.grad(phi))*psi*dx +
            tau*M0*df.dot(df.grad(g), df.grad(psi))*dx +
            g*h*dx - sigma_bar*eps*df.dot(df.grad(phi), df.grad(h))*dx -
            sigma_bar/eps * diff_pf_potential_linearised(phi, filter_pf(phi0))*h*dx)
    a_PF, L_PF = df.lhs(F_PF), df.rhs(F_PF)
    # A_PF = df.assemble(a_PF)
    problem_PF = df.LinearVariationalProblem(a_PF, L_PF, w1_PF)
    solver_PF = df.LinearVariationalSolver(problem_PF)
   
    F_NS = (1./tau * df.dot(rho1*u - rho0*u0, v)*dx
            + 2*nu1*df.inner(df.sym(df.grad(u)), df.grad(v))*dx
            - rho1*df.inner(df.outer(u0, u), df.grad(v))*dx
            - df.dot(rho1*grav + g1*df.grad(phi1), v)*dx
            - p * df.div(v)*dx
            - 0.5*(rho_1-rho_2)*M1*df.inner(df.outer(u, df.grad(g1)),
                                            df.grad(v))*dx
            + df.div(u)*q*dx)
    a_NS, L_NS = df.lhs(F_NS), df.rhs(F_NS)
    freeslip = df.DirichletBC(
        W_NS.sub(0).sub(0), df.Constant(0.),
        "on_boundary && (x[0] < DOLFIN_EPS || x[0] > {Lx}-DOLFIN_EPS)".format(Lx=Lx))
    noslip = df.DirichletBC(
        W_NS.sub(0), df.Constant((0., 0.)),
        "on_boundary && (x[1] < DOLFIN_EPS || x[1] > {Ly}-DOLFIN_EPS)".format(Ly=Ly))
    bcs_NS = [noslip, freeslip]

    problem_NS = df.LinearVariationalProblem(a_NS, L_NS, w1_NS, bcs_NS)
    solver_NS = df.LinearVariationalSolver(problem_NS)

    # solver_PF.parameters["linear_solver"] = "gmres"
    # solver_PF.parameters["preconditioner"] = "default"
    # solver_NS.parameters["linear_solver"] = "gmres"
    # solver_NS.parameters["preconditioner"] = "amg"
    
    field_names = ["u", "p", "phi", "g"]
    tstepfiles = dict()
    for field_name in field_names:
        tstepfiles[field_name] = df.XDMFFile(mesh.mpi_comm(),
                                             "droplet_" + field_name + ".xdmf")
        tstepfiles[field_name].parameters["rewrite_function_mesh"] = False
        tstepfiles[field_name].parameters["flush_output"] = True

    t = t0
    tstep = 0
    dump_intv = 10

    while t < T:
        df.info("tstep " + str(tstep) + " | time " + str(t))
        t += dt
        tstep += 1

        solver_PF.solve()
        solver_NS.solve()

        if tstep % dump_intv == 0:
            phi_out, g_out = w0_PF.split()
            u_out, p_out = w0_NS.split()
            fields = [u_out, p_out, phi_out, g_out]
            for field, field_name in zip(fields, field_names):
                field.rename(field_name, "tmp")
                tstepfiles[field_name].write(field, t)

        # Finally, update
        w0_PF.assign(w1_PF)
        w0_NS.assign(w1_NS)

    df.list_timings(df.TimingClear_clear, [df.TimingType_wall])

    do_plot = False
    if do_plot:
        df.plot(phi0, title="Phase field")
        df.plot(g0, title="Chem.pot")
        df.plot(u0, title="Velocity")
        df.plot(p0, title="Pressure")
        df.interactive()
        # df.plot(mesh, interactive=True)


if __name__ == "__main__":
    problem()
