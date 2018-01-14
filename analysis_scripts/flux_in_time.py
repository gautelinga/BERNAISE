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


class CrossSection(df.SubDomain):
    def __init__(self, x0, dim):
        self.x0 = x0
        self.dim = dim
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return df.near(x[self.dim], self.x0) and on_boundary


def method(ts, dt=0, extra_boundaries="", **kwargs):
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

    extra_boundaries_keys = [s.lower() for s in extra_boundaries.split(",")]
    extra_boundaries = []
    if "left" in extra_boundaries_keys:
        extra_boundaries.append(
            ("extra_left", [CrossSection(ts.nodes[:, 0].min(), 0)]))
    if "right" in extra_boundaries_keys:
        extra_boundaries.append(
            ("extra_right", [CrossSection(ts.nodes[:, 0].max(), 0)]))
    if "top" in extra_boundaries_keys:
        extra_boundaries.append(
            ("extra_top", [CrossSection(ts.nodes[:, 1].max(), 1)]))
    if "bottom" in extra_boundaries_keys:
        extra_boundaries.append(
            ("extra_bottom", [CrossSection(ts.nodes[:, 1].min(), 1)]))
    boundaries_list = [boundaries]
    if len(extra_boundaries) > 0:
        boundaries_list.append(extra_boundaries)

    print boundaries_list

    subdomains = [df.FacetFunction("size_t", ts.mesh)
                  for b in boundaries_list]

    for subdomain in subdomains:
        subdomain.set_all(0)

    ds = [df.Measure("ds", domain=ts.mesh, subdomain_data=subdomain)
          for subdomain in subdomains]

    boundary_to_mark = dict()
    for k, subdomain in enumerate(subdomains):
        for i, (name, boundary_list) in enumerate(boundaries_list[k]):
            for boundary in boundary_list:
                boundary.mark(subdomain, i+1)
            boundary_to_mark[name] = (i+1, k)
            print "Boundary:", name

    x_ = ts.functions()

    if params["enable_NS"]:
        u = x_["u"]
    else:
        u = df.Constant(0.)

    if params["enable_PF"]:
        phi = x_["phi"]
        g = x_["g"]
        exec("from problems.{} import pf_mobility".format(problem))
        M = pf_mobility(phi, params["pf_mobility_coeff"])
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

        for boundary_name, (mark, k) in boundary_to_mark.iteritems():
            for flux_name, flux in fluxes.items():
                data[boundary_name][flux_name][i] = df.assemble(
                    df.dot(flux, n)*ds[k](mark))

        t[i] = ts.times[step]

    savedata = dict()
    flux_keys = sorted(fluxes.keys())
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
