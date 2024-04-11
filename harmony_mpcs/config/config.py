import numpy as np
import yaml
import os
from copy import deepcopy

# get path of file
dir_path = os.path.dirname(os.path.realpath(__file__))


class MPCConfig:

    def __init__(self, config):
        self.use_linear_constraints = config['MPCConfig']['use_linear_constraints']
        self.use_ellipsoid_constraints = config['MPCConfig']['use_ellipsoid_constraints']
        self.use_combined_dynamics = config['MPCConfig']['use_combined_dynamics']
        self.use_interactive_linear_constraints = config['MPCConfig']['use_interactive_linear_constraints']
        self.use_interactive_ellipsoid_constraints = config['MPCConfig']['use_interactive_ellipsoid_constraints']
        self.use_fixed_linear_constraints = config['MPCConfig']['use_fixed_linear_constraints']
        self.n_fixed_linear_constraints = config['MPCConfig']['n_fixed_linear_constraints']
        self.gaussian = config['MPCConfig']['gaussian']

        self.M = 4
        self.FORCES_N = config['MPCConfig']['N']
        self.FORCES_N_bar = self.FORCES_N + 2
        self.FORCES_NU = config['MPCConfig']['nu']
        self.FORCES_NX = config['MPCConfig']['nx']
        self.FORCES_TOTAL_V = self.FORCES_NX + self.FORCES_NU
        self.FORCES_NPAR = 56
        self.N_AGENTS_MPC = config['MPCConfig']['n_agents_mpc']

        self.goal_error_weight = config['MPCConfig']['goal_error_weight']
        self.goal_error_others_weight = config['MPCConfig']['goal_error_others_weight']

        self.egoforceFactorDesired = 1.0
        self.egoforceFactorSocial = 1.0
        self.egoforceFactorObstacle = 0.0
        self.egoforceSigmaObstacle = 0
        self.egopref_speed = 1.2

        self.include_fixed_sfm = True

   


class RobotConfig:

    def __init__(self, n_agents_mpc):
        self.name = 'Jackal'

        self.length = 0.65
        self.width = 0.65
        self.com_to_back = 0.325

        self.lower_bound = dict()
        self.upper_bound = dict()

        self.lower_bound['x'] = -10.0
        self.upper_bound['x'] = 10.0

        self.lower_bound['x0'] = -10.0
        self.upper_bound['x0'] = 10.0

        self.lower_bound['y'] = -10.0
        self.upper_bound['y'] = 10.0

        self.lower_bound['y0'] = -10.0
        self.upper_bound['y0'] = 10.0

        self.lower_bound['psi'] = -2 * np.pi
        self.upper_bound['psi'] = +2 * np.pi

        self.lower_bound['psi0'] = -2 * np.pi
        self.upper_bound['psi0'] = +2 * np.pi

        self.lower_bound['v'] = 0.0
        self.upper_bound['v'] = 1.5

        self.lower_bound['v0'] = 0.0
        self.upper_bound['v0'] = 1.5

        self.lower_bound['v_x0'] = -1.4
        self.upper_bound['v_x0'] = 1.4

        self.lower_bound['v_y0'] = -1.4
        self.upper_bound['v_y0'] = 1.4

        self.lower_bound['w'] = -2.0
        self.upper_bound['w'] = 2.0

        self.lower_bound['w0'] = -2.0
        self.upper_bound['w0'] = 2.0

        self.lower_bound['alpha'] = -2.0  # Not correct!
        self.upper_bound['alpha'] = 2.0

        self.lower_bound['alpha0'] = -2.0  # Not correct!
        self.upper_bound['alpha0'] = 2.0

        self.lower_bound['a'] = -1.5
        self.upper_bound['a'] = 1.5

        self.lower_bound['a0'] = -1.0
        self.upper_bound['a0'] = 1.0

        for i in range(1, n_agents_mpc):

            self.lower_bound[f'x{i}'] = -10.0
            self.upper_bound[f'x{i}'] = 10.0

            self.lower_bound[f'y{i}'] = -10.0
            self.upper_bound[f'y{i}'] = 10.0

            self.lower_bound[f'psi{i}'] = -2 * np.pi
            self.upper_bound[f'psi{i}'] = +2 * np.pi

            self.lower_bound[f'v_x{i}'] = -1.2
            self.upper_bound[f'v_x{i}'] = 1.2

            self.lower_bound[f'v_y{i}'] = -1.2
            self.upper_bound[f'v_y{i}'] = 1.2


        self.lower_bound['spline'] = -1.0
        self.upper_bound['spline'] = 10000.0



        self.lower_bound['a_x'] = -2.0
        self.upper_bound['a_x'] = 2.0

        self.lower_bound['a_y'] = -2.0
        self.upper_bound['a_y'] = 2.0

        self.lower_bound['slack'] = 0.0
        self.upper_bound['slack'] = 0.2

        self.lower_bound['slack0'] = 0.0
        self.upper_bound['slack0'] = 1

        self.lower_bound['side_pref'] = -1
        self.upper_bound['side_pref'] = 1



class EnvConfig:
    def __init__(self, config_path):
        print(config_path)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.DT = 0.1

        self.MPCConfig = MPCConfig(config)

        self.RobotConfig = RobotConfig(self.MPCConfig.N_AGENTS_MPC)







