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

    
    def _get_model_parameters():
            for params_file in glob.glob(
                self.params_prefix + "*" + self.params_suffix):
                parameters = dict()
                load_parameters(parameters, params_file)
            self.parameters = parameters
            

    
    #SimulationParameters

def main():
    info_yellow("BERNAISE: Unit-conversion tool")
    cmd_kwargs = parse_command_line()

    # Get help if it was called for.
    if cmd_kwargs.get("help", False):
        get_help()

    folder = cmd_kwargs.get("folder", False)

    LoadSettings(folder)

    # Call the specified method
    #if method in __methods__:
    #    globals()[method](ts, **cmd_kwargs)
    #else:
    #    info_on_red("The specified conversion method doesn't exist.")



if __name__ == "__main__":
    main()