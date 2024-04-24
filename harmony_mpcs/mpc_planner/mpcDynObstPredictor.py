import numpy as np

class MPCDynObstPredictor(object):

    def __init__(self, config) -> None:
        self._dt = config['time_step']
        

    
    def predict(self, poses, vels, N) -> None:
        self.predictions = np.zeros((poses.shape[0], N, 3))
        for i in range(poses.shape[0]):
            self.predictions[i,:,:] = [poses[i,:] + self._dt * vels[i,:]*j for j in range(N)]
        