"""
This is the main module for running the BERNAISE code.
More specific info will follow in a later commit.
"""
import dolfin as df
from common.cmd import parse_command_line, help_menu
from common.io import create_initial_folders, load_checkpoint, save_solution, \
    load_parameters, load_mesh

__author__ = "Gaute Linga"

cmd_kwargs = parse_command_line()

# Check if user has called for help
if cmd_kwargs.get("help", False):
    help_menu()
    exit()

# Import problem and default parameters
default_problem = "simple"
exec("from problems.{} import *".format(
    cmd_kwargs.get("problem", default_problem)))

# Problem specific parameters
parameters.update(problem())

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

    info_red("Loading mesh from checkpoint.")
    mesh = load_mesh(os.path.join(restart_folder, "fields.h5"),
                     use_partition_from_file=True)

# Import solver functionality
exec("from solvers.{} import *".format(solver))

# Get subproblems
subproblems = get_subproblems(**vars())

# Declare finite elements
elements = dict()
for name, (family, degree, is_vector) in base_elements.items():
    if is_vector:
        elements[name] = df.VectorElement(family, mesh.ufl_cell(), degree)
    else:
        elements[name] = df.FiniteElement(family, mesh.ufl_cell(), degree)

# Declare function spaces
spaces = dict()
for name, subproblem in subproblems.items():
    if len(subproblem) > 1:
        spaces[name] = df.FunctionSpace(
            mesh, df.MixedElement(
                [elements[s["element"]] for s in subproblem]),
            constrained_domain=constrained_domain(**vars()))
    # If there is only one field in the subproblem, don't bother with
    # the MixedElement.
    elif len(subproblem) == 1:
        spaces[name] = df.FunctionSpace(
            mesh, elements[subproblem[0]["element"]],
            constrained_domain=constrained_domain(**vars()))
    else:
        info_on_red("Something went wrong here!")
        exit("")

# dim = mesh.topology().dim()  # In case the velocity fields should be
#                              # segregated at some point
fields = []
field_to_subspace = dict()
field_to_subproblem = dict()
for name, subproblem in subproblems.items():
    if len(subproblem) > 1:
        for i, s in enumerate(subproblem):
            field = s["name"]
            fields.append(field)
            field_to_subspace[field] = spaces[name].sub(i)
            field_to_subproblem[field] = (name, i)
    else:
        field = subproblem[0]["name"]
        fields.append(field)
        field_to_subspace[field] = spaces[name]
        field_to_subproblem[field] = (name, -1)


# Create initial folders for storing results
newfolder, tstepfiles = create_initial_folders(folder, restart_folder,
                                               fields, tstep, parameters)

# Create overarching test and trial functions
test_functions = dict()
trial_functions = dict()
for name, subproblem in subproblems.items():
    if len(subproblem) > 1:
        test_functions[name] = df.TestFunctions(spaces[name])
        trial_functions[name] = df.TrialFunctions(spaces[name])
    else:
        test_functions[name] = df.TestFunction(spaces[name])
        trial_functions[name] = df.TrialFunction(spaces[name])

# Create work dictionaries for all subproblems
w_ = dict((subproblem, df.Function(space, name=subproblem))
           for subproblem, space in spaces.items())
w_1 = dict((subproblem, df.Function(space, name=subproblem+"_1"))
            for subproblem, space in spaces.items())
w_tmp = dict((subproblem, df.Function(space, name=subproblem+"_tmp"))
             for subproblem, space in spaces.items())

# Shortcuts to the fields
x_ = dict()
for name, subproblem in subproblems.items():
    if len(subproblem) > 1:
        w_loc = df.split(w_[name])
        for i, field in enumerate(subproblem):
            x_[field["name"]] = w_loc[i]
    else:
        x_[subproblem[0]["name"]] = w_[name]

# If continuing from previously, restart from checkpoint
load_checkpoint(restart_folder, w_, w_1)

# Get boundary conditions, from fields to subproblems
bcs_tuple = create_bcs(**vars())
if len(bcs_tuple) == 3:
    boundaries, bcs, bcs_pointwise = bcs_tuple
elif len(bcs_tuple) == 2:
    boundaries, bcs = bcs_tuple
    bcs_pointwise = None
else:
    info_on_red("Wrong implementation of create_bcs.")
    exit()

# Set up subdomains
subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
subdomains.set_all(0)
boundary_to_mark = dict()
mark_to_boundary = dict()
for i, (boundary_name, markers) in enumerate(boundaries.items()):
    for marker in markers:
        marker.mark(subdomains, i+1)
    boundary_to_mark[boundary_name] = i+1
    mark_to_boundary[i] = boundary_name

if dump_subdomains:
    subdomains_xdmf = df.XDMFFile("subdomains_dump.xdmf")
    subdomains_xdmf.write(subdomains)

# Set up dirichlet part of bcs
dirichlet_bcs = dict()
for subproblem_name in subproblems.keys():
    dirichlet_bcs[subproblem_name] = []

# Neumann BCs (per field)
neumann_bcs = dict()
for field in fields:
    neumann_bcs[field] = dict()

for boundary_name, bcs_fields in bcs.items():
    for field, bc in bcs_fields.items():
        subproblem_name = field_to_subproblem[field][0]
        subspace = field_to_subspace[field]
        mark = boundary_to_mark[boundary_name]
        if bc.is_dbc():
            dirichlet_bcs[subproblem_name].append(
                bc.dbc(subspace, subdomains, mark))
        if bc.is_nbc():
            neumann_bcs[field][boundary_name] = bc.nbc()

# Pointwise dirichlet bcs
for field, (value, c_code) in bcs_pointwise.items():
    subproblem_name = field_to_subproblem[field][0]
    subspace = field_to_subspace[field]
    if not isinstance(value, df.Expression):
        value = df.Constant(value)
    dirichlet_bcs[subproblem_name].append(
        df.DirichletBC(subspace, value, c_code, "pointwise"))

# Compute some mesh related stuff
dx = df.dx
ds = df.Measure("ds", domain=mesh, subdomain_data=subdomains)
normal = df.FacetNormal(mesh)

# Initialize solutions
w_init_fields = initialize(**vars())
if w_init_fields:
    for name, subproblem in subproblems.items():
        w_init_vector = []
        if len(subproblem) > 1:
            for i, s in enumerate(subproblem):
                field = s["name"]
                # Only change initial state if it is given in w_init_fields.
                if field in w_init_fields:
                    w_init_field = w_init_fields[field]
                else:
                    # Otherwise take the default value of that field.
                    w_init_field = w_[name].sub(i)
                # Use df.project(df.as_vector(...)) with care...
                num_subspaces = w_init_field.function_space().num_sub_spaces()
                if num_subspaces == 0:
                    w_init_vector.append(w_init_field)
                else:
                    for j in range(num_subspaces):
                        w_init_vector.append(w_init_field.sub(j))
            # assert len(w_init_vector) == w_[name].value_size()  
            w_init = df.project(
                df.as_vector(tuple(w_init_vector)), w_[name].function_space(),
                solver_type="gmres", preconditioner_type="default")
        else:
            field = subproblem[0]["name"]
            if field in w_init_fields:
                w_init_field = w_init_fields[field]
            else:
                # Take default value...
                w_init_field = w_[name]
            w_init = df.project(w_init_field, w_[name].function_space(),
                                solver_type="gmres",
                                preconditioner_type="default")
        w_[name].interpolate(w_init)
        w_1[name].interpolate(w_init)

# Get rhs source terms (if any)
q_rhs = rhs_source(t=t_0, **vars())

# Setup problem
vars().update(setup(**vars()))

# Problem-specific hook before time loop
vars().update(start_hook(**vars()))

stop = False
t = t_0

# Initial state to XDMF
stop = save_solution(**vars())

total_computing_time = 0.
total_num_tsteps = 0

tstep_0 = tstep

timer = df.Timer("Simulation loop")
timer.start()

while not stop:

    tstep_hook(**vars())

    solve(**vars())

    update(**vars())

    t += dt
    tstep += 1

    stop = save_solution(**vars())

    if tstep % info_intv == 0 or stop:
        info_green("Time = {0:f}, timestep = {1:d}".format(t, tstep))
        split_computing_time = timer.stop()
        split_num_tsteps = tstep-tstep_0
        timer.start()
        tstep_0 = tstep
        total_computing_time += split_computing_time
        total_num_tsteps += split_num_tsteps
        info_cyan("Computing time for previous {0:d}"
                  " timesteps: {1:f} seconds"
                  " ({2:f} seconds/timestep)".format(
                      split_num_tsteps, split_computing_time,
                      split_computing_time/split_num_tsteps))
        df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

if total_num_tsteps > 0:
    info_cyan("Total computing time for all {0:d}"
              " timesteps: {1:f} seconds"
              " ({2:f} seconds/timestep)".format(
                  total_num_tsteps, total_computing_time,
                  total_computing_time/total_num_tsteps))

end_hook(**vars())
