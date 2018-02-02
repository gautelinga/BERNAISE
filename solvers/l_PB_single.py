"""This module implements the a linearized-Poisson-Boltzmann-solver for one phase for the ideal gas chemical potential and two ion-spicese. 
- Planed to add two phase surport 
- Planed to genrelize to N ion-spicese if possibol for linearized version

AB, 2018
"""
import dolfin as df
from common.functions import max_value, alpha, alpha_c, alpha_cc, \
    alpha_reg, alpha_c_reg, absolute
from . import *
from . import __all__
import numpy as np

def get_subproblems(base_elements, solutes, enable_EC,
                    V_lagrange, p_lagrange,
                    **namespace):
    """ Returns dict of subproblems the solver splits the problem into. """
    subproblems = dict()
    if enable_EC:
        subproblems["EC"] = ([dict(name=solute[0], element="c")
                              for solute in solutes]
                             + [dict(name="V", element="V")])
        if V_lagrange:
            subproblems["EC"].append(dict(name="V0", element="V0"))
    return subproblems

def setup(test_functions, trial_functions, w_, w_1,
          ds, dx, normal,
          dirichlet_bcs, neumann_bcs, boundary_to_mark,
          permittivity, density, viscosity,
          solutes, enable_EC, enable_PF,
          dt,
          use_iterative_solvers,
          mesh,
          V_lagrange, p_lagrange,
          **namespace):
    """ Set up problem. """
    # Constant
    veps = df.Constant(permittivity[0])
    
    # If Phase Field 
    if enable_PF:
        phi, g = trial_functions["PF"]
        psi, h = test_functions["PF"]

        phi_, g_ = df.split(w_["PF"])
        phi_1, g_1 = df.split(w_1["PF"])
    else:
        # Defaults to phase 1 if phase field is disabled
        phi_ = phi_1 = 1.
        g_ = g_1 = None

    # Electrochemistry
    c_ = V_ = c_1 = V_1 = V0 = V0_ = V0_1 = b = U = U0 = None
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
    else:
        c_ = V_ = c_1 = V_1 = None

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
        rho_e = sum([c_e*z_e for c_e, z_e in zip(c, z)])  # Sum of trial func.
        rho_e_ = sum([c_e*z_e for c_e, z_e in zip(c_, z)])  # Sum of curr. sol.
    else:
        rho_e_ = None

    if enable_EC:
        solvers["EC"] = setup_EC(w_["EC"], c, V, b, U, rho_e,
                                 dx, ds,
                                 dirichlet_bcs["EC"], neumann_bcs,
                                 boundary_to_mark,
                                 c_1, K_, veps,
                                 z, dbeta,
                                 enable_PF,
                                 use_iterative_solvers)


    return dict(solvers=solvers)
