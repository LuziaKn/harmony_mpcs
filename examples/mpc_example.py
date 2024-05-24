import yaml
import os
import numpy as np

from harmony_mpcs.mpc_planner.mpcPlanner import MPCPlanner

class MPCExample(object):
    def __init__(self, config_file_name):

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir = os.path.join(current_dir, 'config/')


        # load config
        with open(config_dir + config_file_name) as f:
            config = yaml.safe_load(f)
        
        self._config = config['mpc']
        self._robot_config = config['robot']
        self._ped_config = config['pedestrians']
        self._env_config = config['env']
        

        self._solver_directory = current_dir + "/solvers/"
        self._solver_name = self._config['solver_name']
        self._robot_type = self._config['model_name']
     
        self._planner = MPCPlanner(solverDir = self._solver_directory,
                                   solverName =self._solver_name,
                                   config = self._config,
                                   robot_config = self._robot_config,
                                   ped_config = self._ped_config,)
        self._planner.reset()

        self.output = np.zeros((self._planner._N, self._planner._nu + self._planner._nx))

        self._lidar_pc = 1000 * np.ones((3, 1))
        self._trans = [0., 0., 0.]
        
        
    def reset(self):
        self._initial_step = True