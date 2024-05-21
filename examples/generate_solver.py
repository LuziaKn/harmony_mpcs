import os
import argparse
from harmony_mpcs.mpc_solver_generation.solver_settings import SolverSettings
from harmony_mpcs.mpc_solver_generation.solver_generator import SolverGenerator

def generate_solver(config_file_name = 'dingo_config.yaml'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, 'config')

    solver_settings = SolverSettings(config_dir, config_file_name)
    solver_generator = SolverGenerator(solver_settings, current_dir, '/solvers/')
    solver_generator.generate_solver()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate solver.')
    parser.add_argument('config_file_name', type=str, help='Name of the config file')


    args = parser.parse_args()
    print(args.config_file_name)

    generate_solver(args.config_file_name)