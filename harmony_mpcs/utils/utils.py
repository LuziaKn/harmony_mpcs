import numpy as np

def output2array(output):
    i = 0
    N = len(output)
    num_str = str(N)

    dec= len(num_str)

    #dec = int(N//10+1)
    output_array = np.zeros((N,len(output["x{:0{}d}".format(1, dec)])))
    for key in output.keys():
        output_array[i,:] = output[key]
        i +=1

    return output_array

def compute_point_cloud(robot_state: np.ndarray, lidar_obs: np.ndarray, relative_pos_lidar: np.ndarray) -> np.ndarray:
    """
    Computes point cloud in world frame based on raw relative lidar measurements.
    """
    angle = robot_state[2]
    rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)],
    ])

    position_lidar = np.dot(rot_matrix, relative_pos_lidar[:2]) + robot_state[0:2]

    number_rays = lidar_obs.shape[0]
    lidar_position = np.array([position_lidar[0], position_lidar[1], relative_pos_lidar[2]])
    relative_positions = lidar_obs
    #relative_positions = [[1, 0, 0.3] ]
    absolute_positions = np.concatenate(
           (np.dot(relative_positions[:,:2], rot_matrix.T),
            np.expand_dims(relative_positions[:,2],axis=1)),
            axis=1) + lidar_position

    z_threshold = 0.2
    absolute_positions_filtered = absolute_positions[absolute_positions[:,2]>=z_threshold]
    return absolute_positions_filtered