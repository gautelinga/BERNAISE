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

# print parameters

# Import solver functionality
exec("from solvers.{} import *".format(solver))

# Gather all fields
# dim = mesh.geometry().dim()

# Create initial folders for storing results
newfolder, tstepfiles = create_initial_folders(folder, restart_folder,
                                                fields, tstep, parameters)

# Declare finite elements
elements = dict()
for field, (family, degree, is_vector) in base_elements.iteritems():
    if is_vector:
        elements[field] = df.VectorElement(family, mesh.ufl_cell(), degree)
    else:
        elements[field] = df.FiniteElement(family, mesh.ufl_cell(), degree)

# Declare function spaces
spaces = dict()
for subproblem, base_elements in subproblems.iteritems():
    spaces[subproblem] = df.FunctionSpace(
        mesh, df.MixedElement([elements[el] for el in base_elements]),
        constrained_domain=constrained_domain)


test_functions = dict()
trial_functions = dict()
for subproblem in subproblems:
    test_functions[subproblem] = df.TestFunctions(spaces[subproblem])
    trial_functions[subproblem] = df.TrialFunctions(spaces[subproblem])

w_ = dict((subproblem, df.Function(space, name=subproblem))
            for subproblem, space in spaces.iteritems())
w_1 = dict((subproblem, df.Function(space, name=subproblem+"_1"))
            for subproblem, space in spaces.iteritems())

# If continuing from previously, restart from checkpoint
load_checkpoint(restart_folder, w_, w_1)

bcs = create_bcs(**vars())

# Initialize solutions
initialize(**vars())

# Setup problem
vars().update(setup(**vars()))

# Problem-specific hook before time loop
vars().update(start_hook(**vars()))

stop = False
t = t_0
while t < T and not stop:
    t += dt
    tstep += 1

    for subproblem in subproblems_order:
        print subproblem