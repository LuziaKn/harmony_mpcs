import os
#from harmony_mpcs.mpc_solver_generation.fixed_mpc_solver import generate_solver
import harmony_mpcs.mpc_solver_generation.helpers as helpers
from harmony_mpcs.mpc_solver_generation.solver_settings import SolverSettings
from harmony_mpcs.mpc_solver_generation.solver_generator import SolverGenerator

current_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(current_dir, 'config')

solver_settings = SolverSettings(config_dir, 'dingo_config.yaml')
solver_generator = SolverGenerator(solver_settings, current_dir, '/solvers/')
solver_generator.generate_solver()