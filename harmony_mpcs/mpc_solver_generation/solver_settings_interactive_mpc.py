from harmony_mpcs.mpc_solver_generation.solver_settings import SolverSettings
import harmony_mpcs.mpc_solver_generation.dynamics as dynamics
import harmony_mpcs.mpc_solver_generation.control_modules as control_modules
import copy
class SolverSettingsinteractive_mpc(SolverSettings):

    def __init__(self, config_dir: str, config_file_name: str):
        super(SolverSettingsinteractive_mpc, self).__init__(config_dir, config_file_name)

        # add parameters of response dynnamics for pedestrians
        for i in range(self._n_dynamic_obst):
            self._params.add_parameter("pedestrians_sfm_param" + str(i), 6)

        model0 = dynamics.PointMass_2order_Model(robot_config=self._robot_config)
        model1 = dynamics.PointMass_2order_Model(robot_config=self._pedestrian_config)
        models = [model0]
        for i in range(0, self._n_dynamic_obst):
            models.append(copy.deepcopy(model1))
        self._model = dynamics.CombinedDynamics(models, self._params, self._robot_config)
        self._nx_per_agent = model0.nx

    def set_ineq_constr_dynamic(self, n_discs, n_obst):
        self._modules.add_module(control_modules.EllipsoidalConstraintModule(self._params, n_discs=n_discs, n_obst=n_obst, interactive=True))

    def set_ineq_constraints(self, n_discs, n_obst):
        self._modules.add_module(control_modules.LinearConstraintModule(self._params, n_discs=n_discs, n_obst=n_obst, horizon_length = self._N+2))

    def set_obj(self):
        self._modules.add_module(control_modules.FixedMPCModule(self._params, self._N)) # Track a reference path

