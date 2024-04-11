import numpy as np
import os
import pickle

class MPCPlanner(object):

    def __init__(self, solverDir, solverName):

        self._solverFile = (
            solverDir
            + solverName
        )
        print(self._solverFile)
                
        if not os.path.isdir(self._solverFile):
            raise(SolverDoesNotExistError(self._solverFile))
        
        with open(self._solverFile + "/params.pkl", 'rb') as file:
            try:
                data = pickle.load(file)
                self._properties = data['properties']
                self._map_runtime_par = data['map_runtime_par']
                self._modules = data['modules']
            except FileNotFoundError:
                print(f"File {params_file_path} not found.")
            except pickle.UnpicklingError:
                print(f"Error occurred while unpickling {params_file_path}.")
            except Exception as e:
                print(f"An error occurred: {e}")
        
        self._nx = self._properties['nx']
        self._nu = self._properties['nu']
        self._npar = self._properties['npar']
        print(self._npar)
        self._N = self._properties['N']

    def reset(self):
        self._x0 = np.zeros(shape=(self._N, self._nx + self._nu))
        self._xinit = np.zeros(self._nx)

    def solve(self):
        pass
    
    def computeAction(self):
     

        #self._action, output, info, exitflag = self.solve(obs)
        self._action = [0.1, 0]
        print('action: ', self._action)
        return self._action #, output, exitflag, self.vel_limit