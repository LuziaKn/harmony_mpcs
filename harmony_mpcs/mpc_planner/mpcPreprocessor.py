import numpy as np
from typing import List, Union, Tuple

from harmony_mpcs.utils.free_space_decomposition import FreeSpaceDecomposition

class MPCPreprocessor(object):

    def __init__(self, config, N):
        # collision avoidance
        self._n_obstacles = 1
        self._max_radius = 5
        self._N = N
        self._fsd = FreeSpaceDecomposition(number_constraints=self._n_obstacles, max_radius=self._max_radius)

    def compute_constraints(self, robot_state: np.ndarray, point_cloud: np.ndarray, trans_lidar) -> Tuple[List, List]:
        """
        Computes linear constraints given a pointcloud as numpy array.
        The seed point is the robot_state.
        """
        angle = robot_state[2]
        rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
        ])

        pos_lidar_2d = np.dot(rot_matrix, np.array([trans_lidar[0], trans_lidar[1]])) + robot_state[0:2]
        pos_lidar_3d = np.array([pos_lidar_2d[0], pos_lidar_2d[1], trans_lidar[2]])

        print(pos_lidar_2d)
        #print(trans_lidar)
        #print(robot_state)

        self._fsd.set_position(pos_lidar_3d)
        self._fsd.compute_constraints(point_cloud)
        return list(self._fsd.asdict().values()), self._fsd.constraints(), self._fsd.points()    
    
    def preprocess(self, x_ref, point_cloud, trans_lidar):
        self._linear_constraints = []
        halfplanes = []
        for j in range(self._N):
            self._linear_constraints_j, halfplanes_j, points_j = self.compute_constraints(x_ref, point_cloud, trans_lidar)
            self._linear_constraints.append(self._linear_constraints_j)
            halfplanes.append(halfplanes_j)
            self._closest_points = points_j
           


