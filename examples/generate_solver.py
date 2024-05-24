import os
import argparse
import importlib
from harmony_mpcs.mpc_solver_generation.solver_generator import SolverGenerator

def generate_solver(config_file_name = 'dingo_config.yaml', solver_settings_file_str = 'solver_settings_interactive_mpc'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, 'config')

    module_name = 'harmony_mpcs.mpc_solver_generation.' + solver_settings_file_str
    prefix = 'solver_settings_'
    suffix = solver_settings_file_str[len(prefix):]
    print(suffix)
    solver_settings_module = importlib.import_module(module_name)
    SolverSettings = getattr(solver_settings_module, 'SolverSettings' + suffix)

    solver_settings = SolverSettings(config_dir, config_file_name)
    solver_settings.set_constr_obj()
    solver_generator = SolverGenerator(solver_settings, current_dir, '/solvers/')
    solver_generator.generate_solver()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate solver.')
    parser.add_argument('config_file_name', type=str, help='Name of the config file')


    args = parser.parse_args()
    print(args.config_file_name)

    generate_solver(args.config_file_name)