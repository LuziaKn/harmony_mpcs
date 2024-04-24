import numpy as np

class MPCDynObstPredictor(object):

    def __init__(self, config) -> None:
        pass
    
    def predict(self, poses, vels, N) -> None:
        self.predictions = np.zeros((poses.shape[0], N, 3))
        for i in range(poses.shape[0]):
            self.predictions[i] = [poses[i,:] + vels[i,:]*j for j in range(N)]