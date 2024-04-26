import numpy as np
import gymnasium as gym

from harmony_mpcs.examples.mpc_example import MPCExample
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

class DingoExample(MPCExample):

    def __init__(self, config_file_name: str):
        super().__init__(config_file_name)

    def initialize_environment(self):

        robots = [
            GenericUrdfReacher(urdf="pointRobot.urdf", mode='vel'),
        ]

        self._env = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=True
        ).unwrapped

        # Set the initial position and velocity of the point mass.
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
        )
        self._r_body = 0.3
        self._limits = np.array([
                [-10, 10],
                [-10, 10],
                [-10, 10],
        ])
        self._limits_u = np.array([
                [-1, 1],
                [-1, 1],
                [-15, 15],
        ])
        # Definition of the obstacle.
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [4.0, -0.5, 0.0], "radius": self._ped_config['radius']},
        }
        obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
        self._obstacles = [obst1]
        # Definition of the goal.
        self._n = 3
        self._goal = {'position': [8.2, -0.2, 0], 'orientation': 0.0}
        goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": self._n,
                "desired_position": self._goal['position'][:2],
                "epsilon": 0.1,
                "type": "staticSubGoal"
            }
        }
        self._goal_comp = GoalComposition(name="goal", content_dict=goal_dict)
        pos0 = np.median(self._limits, axis = 1)
        vel0 = np.array([0.1, 0.0, 0.0])
        print(type(self._env))
        self._env.reset(pos=pos0, vel=vel0)
        self._env.add_sensor(full_sensor, [0])
        self._env.add_goal(self._goal_comp.sub_goals()[0])
        for obstacle in self._obstacles:
            self._env.add_obstacle(obstacle)
        self._env.set_spaces()

        for i in range(self._config['time_horizon']):
            self._env.add_visualization(size=[self._r_body, 0.1])

        self._lidar_pc = 1000* np.ones((3,1))
        self._trans = np.zeros((3,))

        return {}
        

    def run(self):
        
        action = np.array([0.0, 0.0, 0.0])
        ob, *_ = self._env.step(action)
        n_steps = 1000
        for i in range(n_steps):
            self._x = ob["robot_0"]['joint_state']['position']
            self._xdot = ob["robot_0"]['joint_state']['velocity']
            self._dyn_obst = np.zeros((1,6))
            self._dyn_obst[0, :3] = [4.0, -0.5, 0.0]
            obs = {"x": self._x,
                    "xdot": self.output[1,self._planner._nu+ 3 : self._planner._nu + self._planner._nx],
                    "goal": self._goal,
                    "lidar_point_cloud": self._lidar_pc,
                    "trans_lidar": self._trans,
                    "dyn_obst": self._dyn_obst,}
            action, self.output, _, _ = self._planner.computeAction(obs)
      

            ob, *_ = self._env.step(np.concatenate([action[:2],np.zeros(1)]))
            print(action)
            self._env.update_visualizations(self.output[:,:3])
            print('no error')

def main():
    dingo_example = DingoExample("dingo_config.yaml")
    dingo_example.initialize_environment()
    dingo_example.run()


if __name__ == "__main__":
    main()