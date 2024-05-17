import numpy as np
import os
import pickle

from harmony_mpcs.utils.utils import output2array
from harmony_mpcs.mpc_planner.mpcPreprocessor import MPCPreprocessor
from harmony_mpcs.mpc_planner.mpcDynObstPredictor import MPCDynObstPredictor


class MPCPlanner(object):

    def __init__(self, solverDir=None, solverName=None, solver_function=None, config=None, robot_config=None, ped_config=None, mode='gazebo_ros1'):

        if config is not None and robot_config is not None and ped_config is not None:
            self._config = config
            self._robot_config = robot_config
            self._ped_config = ped_config
        else:
            raise Exception("Configurations not provided")

        
        self._mode = mode
        self._solver_function = solver_function

        self._dt = self._config['time_step']
        

        self._solverFile = (
            solverDir
            + solverName
        )
        print(self._solverFile)
                
        if not os.path.isdir(self._solverFile):
            raise Exception("Solver file  does not exist", self._solverFile)
        
        with open(self._solverFile + "/params.pkl", 'rb') as file:
            try:
                data = pickle.load(file)
                self._properties = data['properties']
                self._map_runtime_par = data['map_runtime_par']
                self._modules = data['modules']
            except Exception as e:
                print(f"An error occurred: {e}")

        if not 'ros2' in self._mode:
            import forcespro
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

        self._preprocessor = MPCPreprocessor(config, self._N, self._nu, self._nx)
        self._predictor = MPCDynObstPredictor(config)
        
        self._debug = self._config['debug']


    def reset(self):
        self._initial_step = True

        self._x0 = np.zeros(shape=(self._N, self._nx + self._nu))
        self._xinit = np.zeros(self._nx)
        self._params = np.zeros(shape=(self._npar * self._N), dtype=float)

        self.setWeights()

    def setX0(self, initialize_type="current_state", initial_step= True):
        if initialize_type == "current_state" or initialize_type == "previous_plan" and initial_step:
            for i in range(self._N):
                self._x0[i][self._nu:] = self._xinit
                self._initial_step = False
        elif initialize_type == "previous_plan":
            self.shiftHorizon(self.output)
        else:
            np.zeros(shape=(self._N, self._nx + self._nu + self._ns))

    def setWeights(self,update_goal_region=False):
        for N_iter in range(self._N):
            k = N_iter * self._npar
            selected_weights = {key: value for key, value in self._map_runtime_par.items() if key.startswith('W')}
            
            
            for key, _ in selected_weights.items():
                if self._debug:
                    print('!!!!!!!!!!!!!!!!!!!!', key, flush=True)
                try:
                    self._params[k+self._map_runtime_par[key][0]] = self._config['weights'][key]
                except Exception as e:
                    print(f"Set the weight {e} in the config file")
            self._params[k+self._map_runtime_par['Wgoal_orientation'][0]] = 0
            
            if update_goal_region:
                self._params[k+self._map_runtime_par['Wvel_orientation'][0]] = 0
                self._params[k+self._map_runtime_par['Wgoal_orientation'][0]] = self._config['weights']['Wgoal_orientation']
                

    def setParams(self, params_dict):
        for N_iter in range(self._N):
            k = N_iter * self._npar
        
            self.dim = 3
            for i in range(self.dim):
                self._params[k+self._map_runtime_par['goal_position'][i]] = params_dict['goal_position'][i]
                self._params[k+self._map_runtime_par['initial_pose'][i]] = params_dict['initial_pose'][i]
            
            self._params[k+self._map_runtime_par['goal_orientation'][0]] =  params_dict['goal_orientation']
            self._params[k+ self._map_runtime_par['disc_0_r'][0]] = params_dict['robot_radius']
            self._params[k+ self._map_runtime_par['disc_0_offset'][0]] = 0

        self.setLinearConstraints(self._preprocessor._linear_constraints, r_body=self._robot_radius) 

        pos_dynamic_obst = self._dyn_obst[:,:3]
        vel_dynamic_obst = self._dyn_obst[:,3:]
        self.setEllipsoidConstraints(pos_dynamic_obst=pos_dynamic_obst, vel_dynamic_obst=vel_dynamic_obst) 

    def setLinearConstraints(self, lin_constr, r_body):
        for N_iter in range(self._N):
            k = N_iter * self._npar
            for obst_id in range(self._n_static_obst): 
                name = "linear_constraint_" + str(obst_id)
                self._params[k + self._map_runtime_par[name + "_a1"][0]] = lin_constr[N_iter][obst_id][0]
                self._params[k + self._map_runtime_par[name + "_a2"][0]] = lin_constr[N_iter][obst_id][1]
                self._params[k + self._map_runtime_par[name + "_b"][0]] = lin_constr[N_iter][obst_id][-1]
 
                #print('constraint a1 obst' + str(obst_id), self._params[k + self._map_runtime_par[name + "_a1"][0]], flush=True)

           
    def setEllipsoidConstraints(self, pos_dynamic_obst, vel_dynamic_obst):
        major = 0
        minor = 0
        for N_iter in range(self._N):
           
            k = N_iter * self._npar
            self._predictor.predict(self._dyn_obst[:,:3], self._dyn_obst[:,3:], self._N)

            manual_noise = 0.3
            major = np.sqrt(major**2 + (manual_noise * self._dt)**2)
            minor = np.sqrt(minor**2 + (manual_noise * self._dt)**2)
            
            for obst_id in range(self._n_dynamic_obst):
                pos =  self._predictor.predictions[obst_id][N_iter]

                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_pos"][0]]] = pos[0]
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_pos"][1]]] = pos[1]
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_r" ][0]]] = self._ped_config['radius']
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_psi" ][0]]] = 0
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_major" ][0]]] = major
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_minor" ][0]]] = major
                self._params[[k+self._map_runtime_par["ellipsoid_constraint_agent_" + str(obst_id) + "_chi" ][0]]] = 1

    def solve(self, obs, info):

        self._xinit = np.concatenate([obs['x'], obs['xdot']])
        self._dyn_obst = obs['dyn_obst']
        
        self.setX0(initialize_type=self._config['initialization'], initial_step=self._initial_step)
        
        self._preprocessor.preprocess(obs, info, self._output)
        params_dict = self._preprocessor.get_params_dict()
        
        goal_dist = np.linalg.norm(self._xinit[:2] - self._preprocessor._goal_position[:2])
        print('goal_dist:',  goal_dist, flush=True)
        if goal_dist < self._config['goal_threshold']:
            print('in goal region', flush=True)
            self.setWeights(update_goal_region=True)
        else: 
            self.setWeights()
            
        if obs['xdot'][0] <0.1 and obs['xdot'][1] <=0.1:
            for N_iter in range(self._N):
                k = N_iter * self._npar
                self._params[k+self._map_runtime_par['Wvel_orientation'][0]] = 0
        else:
            for N_iter in range(self._N):
                k = N_iter * self._npar
                self._params[self._map_runtime_par['Wvel_orientation'][0]] = self._config['weights']['Wvel_orientation']
            
   
 
            
            
        
       
        self.setParams(params_dict)

        problem = {}
        problem["xinit"] = self._xinit
        problem["x0"] = self._x0.flatten()[:]
        problem["all_parameters"] = self._params
        if self._debug:
            print('x0:', self._x0, flush=True)
            #print('xinit', self._xinit, flush=True)
            #print(self._nu)
            #self.print_parameters()
        
        self._debug = False
        if self._debug:
            from harmony_mpcs.mpc_solver_generation.inequality import LinearConstraints
            
        
        
        if 'ros1' in self._mode:
            self._output, self._exitflag, info = self._solver.solve(problem)
        elif 'ros2' in self._mode:
            self._output, self._exitflag = self._solver_function(problem)
            if self._debug:
                print('mpcs planner solve function reached (ros2)', flush=True)
            self._output = np.reshape(self._output, (self._N, -1))
            #print('output:', self._output.shape, flush=True)
            #self._output = np.reshape(self._output, (self._N, -1))
            
            #print('output:', self._output.shape, flush=True)
            

        if isinstance(self._output, dict):
            self._output = output2array(self._output)
        

        if self._exitflag < 0:
            print('exit flag:', self._exitflag)

        if self._exitflag == 1 or self._exitflag == 0:
            action = self._output[1,self._nu + self._nx-3:]
        else: 
            action = np.array([0.,0.,0.])
            self._output = np.zeros((self._N, self._nx + self._nu))
            self._output[:,self._nu:self._nu+self._nx] = self._xinit


        return action, self._output

    
    def computeAction(self, obs):
        info = {'robot_radius': self._robot_radius}
        self._action, output = self.solve(obs, info)
        #print('output:', output, flush=True)

        return self._action, output, self._exitflag, self._preprocessor._linear_constraints , self._preprocessor._closest_points, self._preprocessor._lidar_pc#, exitflag, self.vel_limit
    
    def print_parameters(self):
        for N_iter in range(self._N):
            k = N_iter * self._npar
       
            for key, value in self._map_runtime_par.items():
                for i in range(len(value)):
                    print(key, self._params[k+value[i]])