'''
This converts the simulation parmatetor into dimensionless quantiteis and 
give is the physical uints that the feilds should be scaled whit.
'''
# import external packs 
import dolfin as df
import h5py
import os
import glob
import numpy as np
import sys

# import internal BERNAISE packs
# Find path to the BERNAISE root folder
bernaise_path = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-2])
# ...and append it to sys.path to get functionality from BERNAISE
sys.path.append(bernaise_path)
from utilities.generate_mesh import numpy_to_dolfin
from mpi4py.MPI import COMM_WORLD
from utilities.plot import plot_contour, plot_edges, plot_quiver, plot_faces,\
    zero_level_set, plot_probes, plot_fancy
from utilities.generate_mesh import line_points
from common import load_parameters, info, parse_command_line, makedirs_safe, \
    info_blue, info_cyan, info_split, info_on_red, info_red, info_yellow, \
    parse_xdmf, info_warning


__author__ = "Asger Bolet"

__methods__ = ["dimensionless_numbers"]

__all__ = [] + __methods__


class LoadSettings:
    """ Class for loading timeseries """
    def __init__(self, folder):
        self.folder = folder
        self.settings_folder = os.path.join(folder, "Settings")
        self.params_prefix = os.path.join(self.settings_folder,
                                          "parameters_from_tstep_")
        self.params_suffix = ".dat"

        self.parameters = dict()
        self._get_model_parameters()

    def _get_model_parameters(self):
            for params_file in glob.glob(
                self.params_prefix + "*" + self.params_suffix):
                parameters = dict()
                load_parameters(parameters, params_file)
            self.parameters = parameters

            # The active parts of the solvers
            self.enable_NS = self.parameters[enable_NS]
            self.enable_PF = self.parameters[enable_PF]
            self.enable_EC = self.parameters[enable_EC]

            # Extracting important parameters
            self.Lx = self.parameters["Lx"]
            self.Ly = self.parameters["Ly"]

            self.temperatur = 1
            self.k_b = 1
            self.varcum_permittivity = 1
            self.q_e = 1

            if self.enable_PF:
                self.interface_thickness = self.parameters["interface_thickness"]
                self.pf_mobility_coeff = self.parameters["pf_mobility_coeff"]
                self.surface_tension = self.parameters["surface_tension"]    

            if self.enable_EC:
                self.permittivity = self.parameters["permittivity"]
                self.solutes = self.parameters["solutes"]

            if self.enable_NS:
                self.density = self.parameters["density"]
                self.viscosity = self.parameters["viscosity"]

            '''


            interface_thickness
            solutes=solutes
            Lx=1.,
            Ly=2.,
            surface_tension
            pf_mobility_coeff=factor*0.000010,
            density=[10., 10.],
            viscosity=[1., 1.],
            permittivity=[1., 1.],
        
            model_k_b = 1
            model_T = 1
            model_Vacuum permittivity = 1
            model_electron_charge = 1
            '''
    #SimulationParameters


def main():
    info_yellow("BERNAISE: Unit-conversion tool")
    info_warning("Work in progress!")
    cmd_kwargs = parse_command_line()

    # Get help if it was called for.
    if cmd_kwargs.get("help", False):
        get_help()

    folder = cmd_kwargs.get("folder", False)

    if folder:
        LoadSettings(folder)

    # Call the specified method
    #if method in __methods__:
    #    globals()[method](ts, **cmd_kwargs)
    #else:
    #    info_on_red("The specified conversion method doesn't exist.")



if __name__ == "__main__":
    main()
