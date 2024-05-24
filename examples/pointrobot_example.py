import numpy as np
import time

from mpc_example import MPCExample

from nav_simulator.nav2d_env import Nav2DEnv


class PointRobotExample(MPCExample):
    def __init__(self, config_file_str) :
        super(PointRobotExample, self).__init__(config_file_str)


    def run(self):
        env = Nav2DEnv(self._env_config, self._config)

        # Sample usage
        obs = env.reset()
        self._planner._output[:, self._planner._nu:self._planner._nu+2] = obs['pos_global_frame']


        done = False
        while not done:
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         done = True

            x = np.concatenate([obs['pos_global_frame'], np.array([obs['heading_global_frame']])])
            goal = {}
            goal['position'] = np.concatenate([obs['goal_global_frame'], np.array([0.0])])
            goal['orientation'] = 0.0
            dyn_obst = obs['other_agents_states'][:,:self._planner._nx_per_agent]

            x_joint = x
            xdot_joint = self.output[1, self._planner._nu + 3: self._planner._nu + self._planner._nx]

            if self._config['interactive']:
                # append state
                x_joint = np.zeros((self._config['n_dynamic_obst']+1, int(self._planner._nx_per_agent/2)))
                x_joint[0,:] = x
                xdot_joint = np.zeros((self._config['n_dynamic_obst']+1, int(self._planner._nx_per_agent/2)))
                xdot_joint[0,:] = self.output[1, self._planner._nu + 3: self._planner._nu + self._planner._nx_per_agent]
                for i in range(self._config['n_dynamic_obst']):
                    id = i+1
                    x_joint[id,:] = dyn_obst[i,:int(self._planner._nx_per_agent/2)]
                    xdot_joint[id,:] = dyn_obst[i,int(self._planner._nx_per_agent/2):int(self._planner._nx_per_agent/2)+3]

            obs_dict = {"x": x_joint,
                   "xdot": xdot_joint,
                   "goal": goal,
                   "lidar_point_cloud": self._lidar_pc,
                   "trans_lidar": self._trans,
                   "dyn_obst": dyn_obst, }

            action, self.output, self.exitflag, _, _ = self._planner.computeAction(obs_dict)
            if self.exitflag <1:
                print('exit flag:', self.exitflag)
                action = np.array([0,0,0])
            obs, reward, done, _ = env.step(action)




            env._plot_infos_dict['output'] = self.output[:, self._planner._nu:]
            env._plot_infos_dict['predictions'] = self._planner._predictor.predictions
            start_time = time.time()
            env.render()
            end_time = time.time()
            print(end_time - start_time)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

        env.close()
def main():
    point_robot_example = PointRobotExample('pointrobot_config.yaml')
    point_robot_example.run()

if __name__ == "__main__":
    main()