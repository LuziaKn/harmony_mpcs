import numpy as np
from typing import List, Union, Tuple

from harmony_mpcs.utils.free_space_decomposition import FreeSpaceDecomposition

class MPCPreprocessor(object):

    def __init__(self, config, N, nu, nx):
        # collision avoidance
        self._linear_constr_fixed_over_horizon = config['linear_constr_fixed_over_horizon']
        self._n_obstacles = config['n_static_obst']
        self._max_radius = 4
        self._min_radius = 0.4
        self._N = N
        self._nu = nu
        self._nx = nx
        self._fsd = FreeSpaceDecomposition(number_constraints=self._n_obstacles,
                                           max_radius=self._max_radius,
                                           min_radius=self._min_radius)
        self

    def define_position(self, robot_state: np.ndarray, trans_lidar: np.ndarray):

        angle = robot_state[2]
        rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
        ])

        pos_lidar_2d = np.dot(rot_matrix, np.array([trans_lidar[0], trans_lidar[1]])) + robot_state[0:2]
        pos_lidar_3d = np.array([pos_lidar_2d[0], pos_lidar_2d[1], trans_lidar[2]])
        
        return pos_lidar_3d
        
    def compute_constraints(self, robot_state: np.ndarray, point_cloud: np.ndarray, trans_lidar) -> Tuple[List, List]:
        """
        Computes linear constraints given a pointcloud as numpy array.
        The seed point is the robot_state.
        """


        #print(pos_lidar_2d)
        #print(trans_lidar)
        #print(robot_state)
        pos_lidar_3d = self.define_position(robot_state, trans_lidar)
        self._fsd.set_position(pos_lidar_3d)
        self._fsd.compute_constraints(point_cloud)
        self._lidar_pc = self._fsd.lidar_pc()

        return list(self._fsd.asdict().values()), self._fsd.constraints(), self._fsd.points()    
    
    def preprocess(self, obs, info, previous_plan=None):
        self._robot_radius = info['robot_radius']

        self._goal_position = obs['goal']['position']
        self._goal_orientation = obs['goal']['orientation']
        self._initial_pose = obs['x']


        x_ref = obs['x']
        point_cloud = obs['lidar_point_cloud']
        trans_lidar = obs['trans_lidar']
        
        pos_lidar_3d = self.define_position(x_ref, trans_lidar)
        self._fsd.set_position(pos_lidar_3d)
        point_cloud = self._fsd.filter_point_cloud(point_cloud)
      
        self._linear_constraints = []
        halfplanes = []

        for j in range(self._N):
            if not self._linear_constr_fixed_over_horizon:
                 x_ref = previous_plan[j,self._nu:self._nu+3]

            self._linear_constraints_j, halfplanes_j, points_j = self.compute_constraints(x_ref, point_cloud, trans_lidar)

            self._linear_constraints.append(self._linear_constraints_j)
            halfplanes.append(halfplanes_j)
            self._closest_points = points_j

    def get_params_dict(self):
        params_dict = {
            "robot_radius": self._robot_radius,
            "goal_position": self._goal_position,
            "goal_orientation": self._goal_orientation,
            "initial_pose": self._initial_pose,
            "linear_constraints": self._linear_constraints,
    
        }
        return params_dict
           


