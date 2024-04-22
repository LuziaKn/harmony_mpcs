
import yaml
import harmony_mpcs.mpc_solver_generation.helpers as helpers
import harmony_mpcs.mpc_solver_generation.control_modules as control_modules

class SolverSettings(object):

    def __init__(self, config_dir: str, config_file_name: str):
        self._config_path = config_dir + '/' + config_file_name
        # load yaml file
        with open(self._config_path, 'r') as stream:
            try:
                print(self._config_path)
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                print("yaml file could not be loaded")
        print(config['mpc'])

        self._model_name = config['mpc']['model_name']
        self._robot_config = config['robot']

        self._N = config['mpc']['time_horizon']     
        self._N_bar = self._N + 2          # MPC Horizon
        self._dt = config['mpc']['time_step']                 # Timestep of the integrator

        # collision avoidance
        self._n_discs = 1
        self._n_other_agents = config['mpc']['n_other_agents_mpc']
        self._n_static_obst = 1

        self._use_sqp_solver = False      # Note: SQP for scenario has max_it = 1
        
        # robot
        self._params = helpers.ParameterStructure()
        self._modules = control_modules.ModuleManager(self._params)

        for disc in range(self._n_discs):
            self._params.add_parameter("disc_" + str(disc) + "_r")
            self._params.add_parameter("disc_" + str(disc) + "_offset")

        weight_list = list()

        self._weights = helpers.WeightStructure(self._params, weight_list)

        self.set_ineq_constraints(self._n_discs, self._n_static_obst)
        self.set_ineq_constr_dynamic(self._n_discs, self._n_other_agents)
        self.set_obj()

        print(self._params)
        self._npar = self._params.n_par()
        self._nh = self._modules.number_of_constraints()

    def set_ineq_constr_dynamic(self, n_discs, n_obst):
        self._modules.add_module(control_modules.EllipsoidalConstraintModule(self._params, n_discs=n_discs, max_obstacles=n_obst))

    def set_ineq_constraints(self, n_discs, n_obst):
        self._modules.add_module(control_modules.LinearConstraintModule(self._params, n_discs=n_discs, num_constraints=n_obst, horizon_length = self._N+2))

    def set_obj(self):
        self._modules.add_module(control_modules.FixedMPCModule(self._params, self._N)) # Track a reference path


