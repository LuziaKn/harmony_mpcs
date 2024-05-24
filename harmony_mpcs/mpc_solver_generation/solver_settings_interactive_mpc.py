from harmony_mpcs.mpc_solver_generation.solver_settings import SolverSettings

class SolverSettingsinteractive_mpc(SolverSettings):

    def __init__(self, config_dir: str, config_file_name: str):
        super(SolverSettingsinteractive_mpc, self).__init__(config_dir, config_file_name)

        # add parameters of response dynnamics for pedestrians
        # for i in range(self._n_dynamic_obst):
        #     self._params.add_parameter("pedestrians_sfm_param" + str(i), 6)

