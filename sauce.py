__author__ = "Gaute Linga"

"""
This is the main module for running the BERNAISE code.
More specific info will follow in a later commit.
"""

from common import *

cmd_kwargs = parse_command_line()

# Import problem
default_problem = "simple"
exec("from problems.{} import *".format(cmd_kwargs.get("problem", default_problem)))

# Internalize cmd arguments and mesh
vars().update(import_problem_hook(**vars()))

print parameters

# Import solver functionality
exec("from solvers.{} import *".format(solver))

# Gather all fields
dim = mesh.geometry().dim()
fields = ["u", "p", "phi", "g", "cp", "cm", "V"]  # Make this part more flexible

# Create initial folders for storing results
newfolder, tstepfiles = create_initial_folders(folder, restart_folder,
                                               fields, tstep, parameters)
