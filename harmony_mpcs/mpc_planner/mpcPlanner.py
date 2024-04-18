import numpy as np
import os
import pickle
import forcespro

from harmony_mpcs.utils.utils import output2array
from harmony_mpcs.mpc_planner.mpcPreprocessor import MPCPreprocessor


class MPCPlanner(object):

    def __init__(self, solverDir, solverName, config):

        self._config = config
        

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
        
        try:
            print("Loading solver %s" % self._solverFile)
            self._solver = forcespro.nlp.Solver.from_directory(self._solverFile)
        except Exception as e:
            print("FAILED TO LOAD SOLVER")
            raise e
        
        self._nx = self._properties['nx']
        self._nu = self._properties['nu']
        self._npar = self._properties['npar']
        self._N = self._properties['N']
        self._output = np.zeros((self._N, self._nx + self._nu))

        self._preprocessor = MPCPreprocessor(config, self._N)

        

    def reset(self):
        self._initial_step = True

        self._x0 = np.zeros(shape=(self._N, self._nx + self._nu))
        self._xinit = np.zeros(self._nx)
        print(self._nx)
        self._params = np.zeros(shape=(self._npar * self._N), dtype=float)

        self.setWeights()

    def setX0(self, initialize_type="current_state", initial_step= True):
        if initialize_type == "current_state" or initialize_type == "previous_plan" and initial_step:
            for i in range(self._N):
                self._x0[i][0 : self._nx] = self._xinit
                self._initial_step = False
        elif initialize_type == "previous_plan":
            self.shiftHorizon(self.output)
        else:
            np.zeros(shape=(self._N, self._nx + self._nu + self._ns))

    def setWeights(self):
        for N_iter in range(self._N):
            k = N_iter * self._npar
            selected_weights = {key: value for key, value in self._map_runtime_par.items() if key.startswith('W')}
            

            for key, _ in selected_weights.items():
                try:
                    self._params[k+self._map_runtime_par[key][0]] = self._config['weights'][key]
                except Exception as e:
                    print(f"An error occurred: {e}")

    def setParams(self, obs):
        for N_iter in range(self._N):
            k = N_iter * self._npar
        
            self.dim = 3
            for i in range(self.dim):
                self._params[k+self._map_runtime_par['goal_position'][i]] = obs['goal']['position'][i]
                self._params[k+self._map_runtime_par['initial_pose'][i]] = self._xinit[i]
            self._params[k+self._map_runtime_par['goal_orientation'][0]] = obs['goal']['orientation']
            #self._params[k+self._map_runtime_par['disc_r'][0]] = 0.4
            others_state = np.array([-10, -10, 0, 0, 0, 0.25])
            for i in range(len(others_state)):
                self._params[k+self._map_runtime_par['agents_pos_r_1'][i]] = others_state[i]
            
    def setLinearConstraints(self, lin_constr, r_body):
        pass
        # for j in range(self._config.time_horizon):
        #     self._params[self._npar * j + self._paramMap["r_body"][0]] = r_body
        #     for i in range(self._config.number_obstacles):
        #         for m in range(4):
        #             idx = self._npar * j + self._paramMap["lin_constrs_" + str(i)][m]
        #             self._params[idx] = lin_constr[j][i][m]
    
    def solve(self, obs):

        self._xinit = np.concatenate([obs['x'], obs['xdot']])
        

        self.setX0(initialize_type="current_state", initial_step=self._initial_step)
        self.setParams(obs)
        self._preprocessor.preprocess(obs['x'], obs['lidar_point_cloud'], obs['trans_lidar'])
        self.setLinearConstraints(self._preprocessor._linear_constraints, r_body=0.5) #todo

        problem = {}
        problem["xinit"] = self._xinit
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params


        self._output, exitflag, info = self._solver.solve(problem)
        self._output = output2array(self._output)

        if exitflag <= 0:
            print(exitflag)

        if exitflag == 1 or exitflag == 0:
            action = self._output[1,self._nu + self._nx-3:]
        else: action = np.array([0,0,0])

        return action, self._output

    
    def computeAction(self, obs):
        self._action, output = self.solve(obs)

        # print('action: ', self._action)
        return self._action, output, self._preprocessor._linear_constraints , self._preprocessor._closest_points#, exitflag, self.vel_limit