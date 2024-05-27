"""
Objective.py defines the objective for the solver. Currently it is system specific, it should be general.
@Todo: Generalize the objectives used.
"""

import numpy as np
import casadi as ca
from harmony_mpcs.mpc_solver_generation.helpers import approx_max, approx_min, get_min_angle_between_vec




class CostTerm:

    def __init__(self, weight, variable):
        self.weight = weight
        self.variable = variable

    def cost(self):
        raise IOError('Costterm with undefined cost!')

class QuadraticCost(CostTerm):

    def __init__(self, weight, variable):
        super().__init__(weight, variable)

    def cost(self):
        return self.weight * self.variable ** 2


class FixedMPCObjective:

    def __init__(self, params, N):
        self.define_parameters(params)
        self._N = N

    def define_parameters(self, params):
        params.add_parameter("goal_position", 3)
        params.add_parameter("initial_pose", 3)
        params.add_parameter("goal_orientation")
        params.add_parameter("Wgoal_position")
        params.add_parameter("Wgoal_orientation")
        params.add_parameter("Wvel_orientation")
        params.add_parameter("Wv")
        params.add_parameter("Wa")
  
    def get_value(self, x, u, settings, stage_idx):

        pos = x[:2]
        psi = x[2]
        v_x = x[3]
        v_y = x[4]
        a_x = u[0]
        a_y = u[1]

        speed = ca.norm_2(x[3:5])

        epsilon = 0.01
        # get ego speed
        # theta_rad = psi
        # R = np.array([[np.cos(theta_rad), np.sin(theta_rad)],
        #     [-np.sin(theta_rad), np.cos(theta_rad)]])
    
        # vel_ego = np.dot(R, x[3:5])

        # Parameters
        initial_pose = getattr(settings._params, "initial_pose")
 
        goal_position = getattr(settings._params, "goal_position")
        goal_orientation = getattr(settings._params, "goal_orientation")

        Wgoal_position = getattr(settings._params, "Wgoal_position")
        Wgoal_orientation = getattr(settings._params, "Wgoal_orientation")

        Wvel_orientation = getattr(settings._params, "Wvel_orientation")
        Wv = getattr(settings._params, "Wv")
        Wa = getattr(settings._params, "Wa")
    
        # Derive position error
        goal_dist_error = (pos[0] - goal_position[0]) ** 2 + (pos[1] - goal_position[1]) ** 2
        initial_goal_dist_error = (initial_pose[0] - goal_position[0]) ** 2 + (initial_pose[1] - goal_position[1]) ** 2 + epsilon
        goal_dist_error_normalized = goal_dist_error/initial_goal_dist_error    



        goal_orientation_error = get_min_angle_between_vec(psi,goal_orientation) / (get_min_angle_between_vec(initial_pose[2],goal_orientation) + epsilon)

        dist_threshold = 0.5

           
        if u.shape[0] >= 2:  # Todo check meaning
            if stage_idx == self._N + 1:
                cost = Wgoal_position * goal_dist_error_normalized
            else:
                cost = Wa * a_x * a_x + Wa * a_y * a_y + Wv * v_x * v_x + Wv* v_y * v_y
        else:
            print("not implemented yet")

        return cost


def objective(z, param, model, settings, stage_idx):
    # print("stage idx in jackal_objective: {}".format(stage_idx))
    cost = 0.
    settings._params.load_params(param)

    # Retrieve variables
    x = z[model.nu:model.nu + model.nx]
    u = z[0:model.nu]



    # Weights
    settings._weights.set_weights(param)

    for module in settings._modules.modules:
        if module.type == "objective":
            for module_objective in module.objectives:
                cost += module_objective.get_value(x, u, settings, stage_idx)

    #cost += settings.weights.velocity * ((v - settings.weights.velocity_reference) ** 2)

    # cost += settings.weights.acceleration * (a ** 2)  # /(model.upper_bound()[0] - model.lower_bound()[0])
    #cost += settings.weights.angular_velocity * (w ** 2)  # /(model.upper_bound()[1] - model.lower_bound()[1])
    return cost



