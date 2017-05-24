__author__ = "Asger J. S Bolet <asgerbolet@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"

"""
Prototype of the timedependet stokes solver and a basic problem.
"""

import dolfin as df
import mshr as mshr
from mpi4py import MPI
from utilities.meshload import LoadMesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Boundary marks numbers and alias 
mark = {"bulk": 0,
        "charged_wall": 1,
        "neutral_wall": 2,
        "inlet": 3,
        "outlet": 4,
        "openside": 5,}   

class StarightCapilarWall(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary   
#
class StarightCapilarInlet(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[1], 0)
#
class StarightCapilarOutlet(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[1], 5)
#

def starightcapilarboundary(mesh):
    if rank == 0:
        print 'Marking boundareis...'
    
    boundarymarkers = df.FacetFunction('size_t', mesh)
    boundarymarkers.set_all(mark["bulk"])
    # mark the boundareis
    capilarwall = StarightCapilarWall()
    capilarwall.mark(boundarymarkers,mark["charged_wall"])
    capilarinlet = StarightCapilarInlet()
    capilarinlet.mark(boundarymarkers,mark["inlet"])
    capilaroutlet = StarightCapilarOutlet()
    capilaroutlet.mark(boundarymarkers,mark["outlet"])
    if rank == 0:
        print 'Done'
    return boundarymarkers
    
def problem(meshname='defulat'):
    '''
    Defination of the Finit Element Method (FEM) version
    of the time dependen Stokes equation.
    '''
    if meshname == 'defulat':
        mesh = LoadMesh()
        boundarymarkers = starightcapilarboundary(mesh)
    else:
        mesh = LoadMesh(meshname)
    
    n = df.FacetNormal(mesh) 
    # Redefine the boundary intergration meausre 
    ds = df.Measure('ds', domain=mesh, subdomain_data= boundarymarkers)

    tetha = 0.5 # time-stepping family. tetha = 0.5 Crank-Nicolson, tetha = 1 forward Euler, tetha = 0 backward Euler.
    dt = 0.1
    eta_0 = 1. # The Kenetic viscosity
    rho_0 = 1. # The Density
    lambdaD = 1. # The Debye lengt, see: https://en.wikipedia.org/wiki/Debye_length
    Sc = 1. # The Schmidt number, see: https://en.wikipedia.org/wiki/Schmidt_number

    #V = df.VectorFunctionSpace(mesh, "CG", 2)
    V = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    N = df.FunctionSpace(mesh, R)
    Fc = df.FunctionSpace(mesh, V)
    W = df.FunctionSpace(mesh, V*P)
    dx = df.dx
    ds = df.ds

    # Trial and test functions
    u, p = df.TrialFunctions(W)  # Is this really necessary?
    v, q = df.TestFunctions(W)

    # Mixed functions
    w = df.Function(W) # current solution
    w0 = df.Function(W) # solution from previous step
    rho = df.Function(N) # the psi dependent density
    eta = df.Function(N) # the psi dependent vicosity
    rhoe = df.Function(N) # the charge density
    phi = df.Function(N) # the electric potential
    f = df.Function(Fc)

    rho.interpolate(df.Constant(rho_0))
    eta.interpolate(df.Constant(eta_0))
    rhoe.interpolate(df.Constant(0))

    # Specify Dirichlet Boundary Conditions
    pressure_inlet = df.Constant(1.000001)
    pressure_outlet = df.Constant(1.0)
    bc_1 = df.DirichletBC(W.sub(0),df.Constant((0.0,0.0)),boundarymarkers,mark["charged_wall"])
    bc_2 = df.DirichletBC(W.sub(1),pressure_inlet,boundarymarkers,mark["inlet"])
    bc_3 = df.DirichletBC(W.sub(1),pressure_outlet,boundarymarkers,mark["outlet"])
    bcs = [bc_1,bc_2,bc_3]
    

    # Split mixed functions
    u, p = df.split(w)
    w.sub(1).interpolate(df.Constant(1.))
    u0, p0 = df.split(w0)
    p0.sub(1).interpolate(df.Constant(1.))
    # The time integration method 
    umid = (1.-tetha)*u + tetha*u0
    pmid = (1.-tetha)*p + tetha*p0
    
    a = 2/Sc*rho*df.inner(v,(u-u0))*dx  -dt*eta*df.inner(df.grad(v),df.grad(umid))*dx -dt*pmid*df.div(v)*dx + dt*q*df.div(umid)*dx +dt*df.inner(v,f)*dx -dt*pressure_inlet*df.inner(n, v)*ds(mark["inlet"]) -dt*pressure_outlet*df.inner(n, v)*ds(mark["outlet"])
    ##a = -df.inner(df.grad(v),df.grad(umid))*dx -pmid*df.div(v)*dx +q*df.div(umid)*dx
    ##L = df.inner(v,f)*dx -pressure_inlet*df.inner(n, v)*ds(mark["inlet"]) -pressure_outlet*df.inner(n, v)*ds(mark["outlet"])
    
    ## dt*1/(2*lambdaD*lambdaD)*rhoe*df.inner(v,df.grad(phi))*dx
    #As = df.assemble(a)
    #[df.bc.apply(As) for bc in bcs]
    #bs = df.assemble(L)
    #[df.bc.apply(bs) for bc in bcs]
    #df.solve(As,W,bs)
    t =0
    while (t<10): 
        df.solve(a==0,w,bcs)
        w0.assign(w)
        t += dt
    u, p = df.split(w)
    df.plot(u,title="vel")
    df.plot(p,title="press")
    df.interactive()


def main():
     problem()
     #mesh = LoadMesh()
     #boundarymarkers = starightcapilarboundary(mesh)
     #df.plot(boundarymarkers,title="boundarymarkers")
     #df.plot(mesh,title="mesh")
     #df.interactive()


if __name__ == "__main__":
    main()