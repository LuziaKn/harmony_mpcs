import os
import harmony_mpcs.mpc_solver_generation.control_modules as control_modules
import harmony_mpcs.mpc_solver_generation.helpers as helpers
from harmony_mpcs.config.config import EnvConfig

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)

config = EnvConfig(parent_dir + "/config/config.yaml")
N = config.MPCConfig.FORCES_N               # MPC Horizon
integrator_stepsize = config.DT  # Timestep of the integrator
n_discs = 1
n_agents_mpc = config.MPCConfig.N_AGENTS_MPC
n_other_agents = n_agents_mpc-1

use_sqp_solver = False      # Note: SQP for scenario has max_it = 1
enable_repulsive = False    # Enable repulsive fields in the objective?

# --- Constraint Selection --- #
use_scenario_constraints = False      # Scenario approach
use_ellipsoid_constraints = False    # Ellipsoid obstacle models
use_gaussian_constraints = False
use_linear_constraints = False

config = config
robot_config = config.RobotConfig


module_selection = 1 # 1: GO-MPC

if module_selection == 1:
    use_linear_constraints = True
    use_ellipsoid_constraints = False
    use_combined_dynamics = False
    use_interactive_linear_constraints = False
    use_fixed_linear_constraints = False
    use_interactive_ellipsoid_constraints = False
    gaussian = False
    use_static_linear_constraints  = False

# --- Interface Selection --- #
interfaces = ['JackalSimulator', 'Jackal']  # Define the interfaces that this solver can run with
# @Note: With more than one interface, the IDE cannot automatically resolve the interface, giving errors

# --- Ellipsoid Settings --- #
if use_ellipsoid_constraints or use_gaussian_constraints or use_interactive_linear_constraints or use_fixed_linear_constraints or use_interactive_ellipsoid_constraints:
    modes = 1
        # Maximum dynamic obstacles to evade with the planner
    max_obstacles = n_other_agents * modes  # To account for the number of modes, we need an ellipsoid per mode!


if use_linear_constraints:
    n_static_constraints = 1
else: n_static_constraints = 0

# --- SQP Settings --- #
if use_sqp_solver:
    print_init_bfgs = True

# === Constraint Definition === #
# ! DO NOT CHANGE THE ORDER ANY PARTS BELOW ! (i.e., parameters first, then inequalities!)
params = helpers.ParameterStructure()
modules = control_modules.ModuleManager(params)


modules.add_module(control_modules.FixedMPCModule(params, robot_config, config))  # Track a reference path

# All added weights must also be in the .cfg file to function with the lmpcc_controller.cpp code!

weight_list = list()

weights = helpers.WeightStructure(params, weight_list)

# --- Inequality Constraints --- #
# Parameters for the vehicle collision region

for disc_id in range(n_discs): #todo consider disc?

    for obs_id in range(n_agents_mpc):
        params.add_parameter("agents_pos_r_" + str(obs_id), 6)


if use_static_linear_constraints:
    modules.add_module(
        control_modules.StaticLinearConstraintModule(params, n_discs=n_discs, max_obstacles=max_obstacles))

if use_interactive_linear_constraints:
    modules.add_module(
        control_modules.InteractiveLinearConstraintModule(params, n_discs=n_discs, max_obstacles=max_obstacles, num_constraints = n_static_constraints))

if use_fixed_linear_constraints:
    modules.add_module(
        control_modules.FixedLinearConstraintModule(params, n_discs=n_discs, max_obstacles=max_obstacles))

if use_ellipsoid_constraints and not gaussian:
    modules.add_module(control_modules.EllipsoidalConstraintModule(params, n_discs=n_discs, max_obstacles=max_obstacles, num_constraints=n_static_constraints))

if use_ellipsoid_constraints and gaussian:
    modules.add_module(control_modules.GaussianEllipsoidalConstraintModule(params, n_discs=n_discs, max_obstacles=max_obstacles,
                                                                   num_constraints=n_static_constraints))


if use_linear_constraints:
    modules.add_module(control_modules.LinearConstraintModule(params, n_discs=n_discs, num_constraints=n_static_constraints, horizon_length = N+2))







# === Collect Constraint Data === #
print(params)
npar = params.n_par()
nh = modules.number_of_constraints()

print(' ')