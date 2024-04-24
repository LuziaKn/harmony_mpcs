import numpy as np
import os
import pickle
import forcespro

from harmony_mpcs.utils.utils import output2array
from harmony_mpcs.mpc_planner.mpcPreprocessor import MPCPreprocessor


class MPCPlanner(object):

    def __init__(self, solverDir, solverName, config, robot_config, ped_config):

        self._config = config
        self._robot_config = robot_config
        self._ped_config = ped_config

        self._dt = self._config['time_step']
        

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

        self._n_static_obst = self._config['n_static_obst']
        self._n_dynamic_obst =  self._config['n_dynamic_obst']

        self._robot_radius = self._robot_config['radius']

        self._output = np.zeros((self._N, self._nx + self._nu))
        print(self._map_runtime_par)

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

            self._params[k+ self._map_runtime_par['disc_0_r'][0]] = self._robot_radius
            self._params[k+ self._map_runtime_par['disc_0_offset'][0]] = 0
            #print(self._params)
            

    def setLinearConstraints(self, lin_constr, r_body):
        for N_iter in range(self._N):
            k = N_iter * self._npar
            for obst_id in range(self._n_static_obst): #todo
                name = "linear_constraint_" + str(obst_id)
                self._params[k + self._map_runtime_par[name + "_a1"][0]] = lin_constr[N_iter][obst_id][0]
                self._params[k + self._map_runtime_par[name + "_a2"][0]] = lin_constr[N_iter][obst_id][1]
                self._params[k + self._map_runtime_par[name + "_b"][0]] = lin_constr[N_iter][obst_id][-1]
           
    def setEllipsoidConstraints(self, pos_dynamic_obst, vel_dynamic_obst):
        for N_iter in range(self._N):
           
            k = N_iter * self._npar
            for obst_id in range(self._n_dynamic_obst):
                pos =  pos_dynamic_obst[obst_id,:] + N_iter * self._dt * vel_dynamic_obst[obst_id,:]
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_pos"][0]]] = pos[0]
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_pos"][1]]] = pos[1]
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_r" ][0]]] = self._ped_config['radius']
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_psi" ][0]]] = 0
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_major" ][0]]] = 0
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_minor" ][0]]] = 0
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_chi" ][0]]] = 0

    def solve(self, obs):

        self._xinit = np.concatenate([obs['x'], obs['xdot']])
        self._dyn_obst = obs['dyn_obst']
        

        self.setX0(initialize_type="current_state", initial_step=self._initial_step)
        self.setParams(obs)
        self._preprocessor.preprocess(obs['x'], obs['lidar_point_cloud'], obs['trans_lidar'])
        name = "disc_"+ str(0)+"_linear_constraint"
        self.setLinearConstraints(self._preprocessor._linear_constraints, r_body=self._robot_radius) #todo
        pos_dynamic_obst = self._dyn_obst[:,:3]
        vel_dynamic_obst = self._dyn_obst[:,3:]
        self.setEllipsoidConstraints(pos_dynamic_obst=pos_dynamic_obst, vel_dynamic_obst=vel_dynamic_obst) 

        problem = {}
        problem["xinit"] = self._xinit
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params


        self._output, self._exitflag, info = self._solver.solve(problem)
        self._output = output2array(self._output)

        if self._exitflag < 0:
            print(self._exitflag)

        if self._exitflag == 1 or self._exitflag == 0:
            action = self._output[1,self._nu + self._nx-3:]
        else: 
            action = np.array([0,0,0])
            self._output = np.zeros((self._N, self._nx + self._nu))
            self._output[self._nu,self._nu:self._nx] = self._xinit


        return action, self._output

    
    def computeAction(self, obs):
        self._action, output = self.solve(obs)

        print(output[0,:2])

        # print('action: ', self._action)
        return self._action, output, self._preprocessor._linear_constraints , self._preprocessor._closest_points#, exitflag, self.vel_limit