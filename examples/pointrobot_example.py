import numpy as np

from mpc_example import MPCExample

from nav_simulator.nav2d_env import Nav2DEnv


class PointRobotExample(MPCExample):
    def __init__(self, config_file_str) :
        super(PointRobotExample, self).__init__(config_file_str)


    def run(self):
        env = Nav2DEnv()

        # Sample usage
        obs = env.reset()
        self._planner._output[:, self._planner._nu:self._planner._nu+2] = obs['pos_global_frame']
        env.render()

        done = False
        while not done:
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         done = True

            x = np.concatenate([obs['pos_global_frame'], np.array([obs['heading_global_frame']])])
            goal = {}
            goal['position'] = np.concatenate([obs['goal_global_frame'], np.array([0.0])])
            goal['orientation'] = 0.0
            dyn_obst = obs['other_agents_states'][:,:self._planner._nx]

            obs_dict = {"x": x,
                   "xdot": self.output[1, self._planner._nu + 3: self._planner._nu + self._planner._nx],
                   "goal": goal,
                   "lidar_point_cloud": self._lidar_pc,
                   "trans_lidar": self._trans,
                   "dyn_obst": dyn_obst, }

            action, self.output, _, _ = self._planner.computeAction(obs_dict)
            obs, reward, done, _ = env.step(action)

            env._plot_infos_dict['output'] = self.output
            env.render()
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

        env.close()
def main():
    point_robot_example = PointRobotExample('pointrobot_config.yaml')
    point_robot_example.run()

if __name__ == "__main__":
    main()