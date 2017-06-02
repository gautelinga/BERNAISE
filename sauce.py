__author__ = "Gaute Linga"

"""
This is the main module for running the BERNAISE code.
More specific info will follow in a later commit.
"""

from common import *

cmd_kwargs = parse_command_line()

# Check if user has called for help
if cmd_kwargs.get("help", False):
    info_yellow("BERNAISE (Binary ElectRohydrodyNAmIc SolvEr)")
    info_red("You called for help.")
    exit()

# Import problem and default parameters
default_problem = "simple"
exec("from problems.{} import *".format(
    cmd_kwargs.get("problem", default_problem)))

# Internalize cmd arguments and mesh
vars().update(import_problem_hook(**vars()))

# If loading from checkpoint, update parameters from file, and then
# again from command line arguments.
if restart_folder:
    info_red("Loading parameters from checkpoint.")
    load_parameters(parameters, os.path.join(
        restart_folder, "parameters.dat"))
    internalize_cmd_kwargs(parameters, cmd_kwargs)
    vars().update(parameters)

# Import solver functionality
exec("from solvers.{} import *".format(solver))

# Get subproblems
subproblems = get_subproblems(**vars())

# Declare finite elements
elements = dict()
for name, (family, degree, is_vector) in base_elements.iteritems():
    if is_vector:
        elements[name] = df.VectorElement(family, mesh.ufl_cell(), degree)
    else:
        elements[name] = df.FiniteElement(family, mesh.ufl_cell(), degree)

# Declare function spaces
spaces = dict()
for name, subproblem in subproblems.iteritems():
    spaces[name] = df.FunctionSpace(
        mesh, df.MixedElement([elements[s["element"]] for s in subproblem]),
        constrained_domain=constrained_domain)

# dim = mesh.geometry().dim()  # In case the velocity fields should be
#                              # segregated at some point
fields = []
field_to_subspace = dict()
for name, subproblem in subproblems.iteritems():
    for i, s in enumerate(subproblem):
        field = s["name"]
        fields.append(field)
        field_to_subspace[field] = spaces[name].sub(i)

# Create initial folders for storing results
newfolder, tstepfiles = create_initial_folders(folder, restart_folder,
                                               fields, tstep, parameters)

# Create overarching test and trial functions
# GL: A nonlinear solver doesn't require trial function?
test_functions = dict((subproblem, df.TestFunctions(spaces[subproblem]))
                      for subproblem in subproblems)
trial_functions = dict((subproblem, df.TrialFunctions(spaces[subproblem]))
                       for subproblem in subproblems)

# Create work dictionaries for all subproblems
w_ = dict((subproblem, df.Function(space, name=subproblem))
           for subproblem, space in spaces.iteritems())
w_1 = dict((subproblem, df.Function(space, name=subproblem+"_1"))
            for subproblem, space in spaces.iteritems())

# If continuing from previously, restart from checkpoint
load_checkpoint(restart_folder, w_, w_1)

# Get boundary conditions, from fields to subproblems
bcs_fields = create_bcs(**vars())
bcs = dict()
for name, subproblem in subproblems.iteritems():
    bcs[name] = []
    for s in subproblem:
        field = s["name"]
        bcs[name] += bcs_fields.get(field, [])

# Initialize solutions
initialize(**vars())

# Setup problem
vars().update(setup(**vars()))

# Problem-specific hook before time loop
vars().update(start_hook(**vars()))

stop = False
t = t_0
df.tic()
while t < T and not stop:
    t += dt
    tstep += 1

    tstep_hook(**vars())

    solve(**vars())

    stop = save_solution(**vars())

    update(**vars())

    if tstep % info_intv == 0:
        info_green("Time = {0:f}, timestep = {1:d}".format(t, tstep))
        info_cyan("Computing time for previous {0:d}"
                  " timesteps: {1:f} seconds".format(info_intv, df.toc()))
        df.list_timings(df.TimingClear_clear, [df.TimingType_wall])
        df.tic()

end_hook(**vars())
