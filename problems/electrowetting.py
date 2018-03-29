import dolfin as df
import os
import numpy as np
from . import *
from common.io import mpi_is_root
from common.bcs import Fixed, NoSlip, FreeSlip, Slip, ContactAngle
__author__ = "Gaute Linga and Asger Bolet"


'''
Control paremetors in BERNAISE units

Potential drop over the electric double layer in units of the termal voltage (sometimes \zeta-potential):
\phi_0 = V_top - V_bottem  
Capacitance  for the droplet interface c_d and c_s is defined by concentration_init_d and concentration_init_s:
C = \sqrt(\epsilon_s\sqrt(\epsilon_d/(2z^2 c_d))/(\epsilon_s\sqrt(\epsilon_s)/(2z^2 c_d)))

b that is the capacitance over the surface tension as is given by: 
1/\sigma\sqrt(\sqrt(1/(2 z^2 c_d \epsilon_d))\sqrt(1/(2 z^2 c_s \epsilon_s)))

For simulation it would be nice to scan in \phi_0 for valus of at least [0-2] but as hight as 10 would be interesting for some aplication (glass can have phi_0 ~ 4).
This scan should give eq (1) in C.W. Monroe et al.

the parameter C one should consider to be  0.5 and 2 so one are in two regions of FIG 3. C.W. Monroe et al.  

and then the b should be 0.005.  
'''

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[1] < df.DOLFIN_EPS and on_boundary)

    
class Top(df.SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        df.SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return bool(x[1] > self.Ly-df.DOLFIN_EPS and on_boundary)

    
class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < df.DOLFIN_EPS and on_boundary)

    
class Right(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(x[0] > self.Lx-df.DOLFIN_EPS and on_boundary)


def problem():
    info_cyan("Charged droplet on electrode.")

    # Define solutes
    # Format: name, valency, diffusivity in phase 1, diffusivity in phase
    #         2, beta in phase 1, beta in phase 2
    solutes = [
        ["c_sp",  1, 1., .001, 0., 4.],
        ["c_sm", -1, 1., .001, 0., 4.],
        ["c_dp",  1, .001, 1., 4., 0.],
        ["c_dm", -1, .001, 1., 4., 0.]
    ]

    # Default parameters to be loaded unless starting from checkpoint.
    parameters = dict(
        solver="basic",
        folder="results_electrowetting",
        restart_folder=False,
        enable_NS=True,
        enable_PF=True,
        enable_EC=True,
        save_intv=5,
        stats_intv=5,
        checkpoint_intv=50,
        tstep=0,
        dt=0.08,  # 0.02,
        t_0=0.,
        T=50.,
        grid_spacing=1./4,  # 1./64,
        interface_thickness=0.05,  # 0.02,
        solutes=solutes,
        base_elements=base_elements,
        Lx=3.,
        Ly=3.,
        rad_init=1.5,  # 1.0,
        #
        V_top=1.,
        V_bottom=0.,
        surface_tension=5.,  # 24.5,
        grav_const=0.,
        concentration_init_s=10.,  # 10.,
        concentration_init_d=0.,  # 10.,
        contact_angle=np.pi/4.,
        #
        pf_mobility_coeff=0.000002,  # 0.000010,
        density=[10., 10.],
        viscosity=[10., 10.],
        permittivity=[.1, .2],  # [.1, .2],
        #
        use_iterative_solvers=True,
        use_pressure_stabilization=False
    )
    return parameters


def mesh(Lx=1, Ly=5, grid_spacing=1./16, rad_init=0.75, **namespace):
    m = df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                         int(Lx/(1*grid_spacing)),
                         int(Ly/(1*grid_spacing)))

    for k in range(3):
        cell_markers = df.MeshFunction("bool", m, 2)
        origin = df.Point(0.0, 0.0)
        for cell in df.cells(m):
            p = cell.midpoint()
            x = p.x()
            y = p.y()

            k_p = 1.6-0.2*k
            k_m = 0.4+0.2*k
            rad_x = 0.75*rad_init
            rad_y = 1.25*rad_init
            
            if (bool(p.distance(origin) < k_p*rad_init and
                     p.distance(origin) > k_m*rad_init)
                or bool((x/rad_x)**2 + (y/rad_y)**2 < k_p**2 and
                        (x/rad_x)**2 + (y/rad_y)**2 > k_m**2)
                or bool((x/rad_y)**2 + (y/rad_x)**2 < k_p**2 and
                        (x/rad_y)**2 + (y/rad_x)**2 > k_m**2)
                or p.y() < 0.5 - k*0.2):
                cell_markers[cell] = True
            else:
                cell_markers[cell] = False
        m = df.refine(m, cell_markers)
    return m


def initialize(Lx, Ly, rad_init,
               interface_thickness, solutes,
               concentration_init_d,
               concentration_init_s,
               contact_angle,
               restart_folder,
               field_to_subspace,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ Create the initial state. """
    rad0 = rad_init*np.sqrt(np.pi/(2*contact_angle - np.sin(2*contact_angle)))
    x0 = 0.
    y0 = -rad0*np.cos(contact_angle)
    
    w_init_field = dict()
    if not restart_folder:
        # Phase field
        if enable_PF:
            w_init_field["phi"] = initial_pf(
                x0, y0, rad0, interface_thickness,
                field_to_subspace["phi"].collapse())

        # Electrochemistry
        if enable_EC:
            for solute in solutes[:2]:
                c_init = initial_pf(x0, y0, rad0, interface_thickness,
                                    field_to_subspace[solute[0]].collapse())
                c_init.vector()[:] = ((0.5*(1+(c_init.vector().get_local())))**10
                                      *concentration_init_s)
                w_init_field[solute[0]] = c_init
            for solute in solutes[2:]:
                c_init = initial_pf(x0, y0, rad0, interface_thickness,
                                    field_to_subspace[solute[0]].collapse())
                c_init.vector()[:] = ((0.5*(1-(c_init.vector().get_local())))**10
                                      *concentration_init_d)
                w_init_field[solute[0]] = c_init
            V_init_expr = df.Expression("0.", degree=1)
            w_init_field["V"] = df.interpolate(
                V_init_expr, field_to_subspace["V"].collapse())

    return w_init_field


def create_bcs(field_to_subspace, Lx, Ly,
               solutes,
               concentration_init_s,
               V_top, V_bottom,
               contact_angle,
               enable_NS, enable_PF, enable_EC,
               **namespace):
    """ The boundary conditions are defined in terms of field. """

    boundaries = dict(
        top=[Top(Ly)],
        bottom=[Bottom(Ly)], 
        left=[Left()],
        right=[Right(Lx)]
    )

    noslip = NoSlip()
    freeslip_y = FreeSlip(0., 0)
    freeslip_x = FreeSlip(0., 1)
    slip_x = Slip(0.005, 1)

    bcs = dict()
    bcs_pointwise = dict()

    bcs["top"] = dict()
    bcs["bottom"] = dict()   
    bcs["left"] = dict()
    bcs["right"] = dict()

    if enable_NS:
        bcs["top"]["u"] = slip_x
        bcs["bottom"]["u"] = noslip        
        bcs["left"]["u"] = freeslip_y
        bcs["right"]["u"] = freeslip_y
        bcs_pointwise["p"] = (
            0.,
            "x[0] > {Lx}- DOLFIN_EPS && x[1] > {Ly} - DOLFIN_EPS".format(Lx=Lx, Ly=Ly))

    if enable_EC:
        for solute in solutes[:2]:
            bcs["top"][solute[0]] = Fixed(concentration_init_s)
        for solute in solutes[2:]:
            bcs["top"][solute[0]] = Fixed(0.)
        bcs["top"]["V"] = Fixed(V_top)
        bcs["bottom"]["V"] = Fixed(V_bottom)

    if enable_PF:
        bcs["bottom"]["phi"] = ContactAngle(np.pi-contact_angle)

    return boundaries, bcs, bcs_pointwise


def initial_pf(x0, y0, rad0, eps, function_space):
    expr_str = ("tanh(1./sqrt(2)*(sqrt(pow(x[0]-x0, 2)"
                "+pow(x[1]-y0, 2))-rad0)/eps)")
    phi_init_expr = df.Expression(
        expr_str,
        x0=x0, y0=y0, rad0=rad0, eps=eps, degree=2)
    phi_init = df.interpolate(phi_init_expr, function_space)
    return phi_init


def tstep_hook(t, tstep, **namespace):
    info_blue("Timestep = {}".format(tstep))


def pf_mobility(phi, gamma):
    """ Phase field mobility function. """
    # return gamma * (phi**2-1.)**2
    func = 1.-phi**2
    return 0.75 * gamma * 0.5 * (1. + df.sign(func)) * func
    #return gamma


def start_hook(newfolder, **namespace):
    statsfile = os.path.join(newfolder, "Statistics/stats.dat")
    return dict(statsfile=statsfile)
