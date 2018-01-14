""" flux_in_time script """
from common import info, info_cyan
from postprocess import get_steps, rank
import numpy as np
import dolfin as df
import os
from common.functions import ramp, dramp, diff_pf_potential_linearised, \
    unit_interval_filter


def description(ts, **kwargs):
    info("Plot flux in time.")


def method(ts, dt=0, **kwargs):
    """ Plot flux in time. """
    info_cyan("Plot flux in time.")

    params = ts.get_parameters()
    steps = get_steps(ts, dt)

    problem = params["problem"]
    info("Problem: {}".format(problem))

    exec("from problems.{} import constrained_domain, create_bcs".format(problem))

    pbc = constrained_domain(**params)
    boundaries, _, _ = create_bcs(**params)

    boundaries = boundaries.items()
    boundaries.insert(0, ("periodic", [pbc]))

    subdomains = df.FacetFunction("size_t", ts.mesh)
    subdomains.set_all(0)

    boundary_to_mark = dict()
    for i, (name, subdomain_list) in enumerate(boundaries):
        for subdomain in subdomain_list:
            subdomain.mark(subdomains, i+1)
        boundary_to_mark[name] = i+1
        print name

    ds = df.Measure("ds", domain=ts.mesh, subdomain_data=subdomains)

    #solver = params["solver"]
    #info("Solver:  {}".format(solver))
    #exec("from solvers.{} import discrete_energy".format(solver))

    x_ = ts.functions()

    if params["enable_NS"]:
        u = x_["u"]
    else:
        u = df.Constant(0.)

    if params["enable_PF"]:
        phi = x_["phi"]
        g = x_["g"]
        exec("from problems.{} import pf_mobility".format(problem))
        M = pf_mobility(phi, params["gamma"])
    else:
        phi = 1.
        g = df.Constant(0.)
        M = df.Constant(0.)

    solutes = params["solutes"]
    c = []
    c_grad_g_c = []
    if params["enable_EC"]:
        V = x_["V"]
    else:
        V = df.Constant(0.)

    dbeta = []  # Diff. in beta
    z = []  # Charge z[species]
    K = []  # Diffusivity K[species]
    beta = []  # Conc. jump func. beta[species]

    for solute in solutes:
        ci = x_[solute[0]]
        dbetai = dramp([solute[4], solute[5]])
        c.append(ci)
        z.append(solute[1])
        K.append(ramp(phi, [solute[2], solute[3]]))
        beta.append(ramp(phi, [solute[4], solute[5]]))
        dbeta.append(dbetai)
        # THIS HAS NOT BEEN GENERALIZED!
        c_grad_g_ci = df.grad(ci) + solute[1]*ci*df.grad(V)
        if params["enable_PF"]:
            c_grad_g_ci += dbetai*df.grad(phi)
        c_grad_g_c.append(c_grad_g_ci)

    nu = ramp(phi, params["viscosity"])
    veps = ramp(phi, params["permittivity"])
    rho = ramp(phi, params["density"])

    dveps = dramp(params["permittivity"])
    drho = dramp(params["density"])

    t = np.zeros(len(steps))

    # Define the fluxes
    fluxes = dict()
    fluxes["Velocity"] = u
    fluxes["Phase"] = phi*u
    fluxes["Mass"] = rho*x_["u"]
    if params["enable_PF"]:
        fluxes["Phase"] += -M*df.grad(g)
        fluxes["Mass"] += -drho*M*df.grad(g)

    if params["enable_EC"]:
        for i, solute in enumerate(solutes):
            fluxes["Solute {}".format(solute[0])] = K[i]*c_grad_g_c[i]
        fluxes["E-field"] = -df.grad(V)

    data = dict()
    for boundary_name in boundary_to_mark:
        data[boundary_name] = dict()
        for flux_name in fluxes:
            data[boundary_name][flux_name] = np.zeros(len(steps))

    n = df.FacetNormal(ts.mesh)

    for i, step in enumerate(steps):
        info("Step {} of {}".format(step, len(ts)))

        for field in x_:
            ts.update(x_[field], field, step)

        for boundary_name, mark in boundary_to_mark.iteritems():
            for flux_name, flux in fluxes.items():
                data[boundary_name][flux_name][i] = df.assemble(
                    df.dot(flux, n)*ds(mark))

        t[i] = ts.times[step]

    savedata = dict()
    flux_keys = fluxes.keys()
    for boundary_name in boundary_to_mark:
        savedata[boundary_name] = np.array(
            zip(steps, t, *[data[boundary_name][flux_name]
                            for flux_name in flux_keys]))

    if rank == 0:
        header = "Step\tTime\t"+"\t".join(flux_keys)
        for boundary_name in boundary_to_mark:
            filename = os.path.join(ts.analysis_folder,
                                    "flux_in_time_{}.dat".format(boundary_name))
            np.savetxt(filename, savedata[boundary_name], header=header)
