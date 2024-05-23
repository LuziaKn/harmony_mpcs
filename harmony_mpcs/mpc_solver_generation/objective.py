"""
Objective.py defines the objective for the solver. Currently it is system specific, it should be general.
@Todo: Generalize the objectives used.
"""

import numpy as np
import casadi as ca
from harmony_mpcs.mpc_solver_generation.helpers import approx_max, approx_min, get_min_angle_between_vec_squared
import harmony_mpcs.mpc_solver_generation.helpers as helpers
from harmony_mpcs.utils.utils import min_angle_diff



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

    def __init__(self, params, N, n_static_obst, n_dyn_obst, n_discs):
        self.define_parameters(params)
        self._N = N
        self._n_static_obst = n_static_obst
        self._n_dyn_obst = n_dyn_obst
        self._n_discs = n_discs

    def define_parameters(self, params):
        params.add_parameter("goal_position", 3)
        params.add_parameter("initial_pose", 3)
        params.add_parameter("goal_orientation")
        params.add_parameter("Wgoal_position")
        params.add_parameter("Wgoal_orientation")
        params.add_parameter("Wvel_orientation")
        params.add_parameter("Wv")
        params.add_parameter("Wa")
        params.add_parameter("Wstatic")
        params.add_parameter("Wdynamic")
        params.add_parameter("Wslack")
  
    def get_value(self, x, u, settings, stage_idx):

        pos = x[:2]
        psi = x[2]
        v_x = x[3]
        v_y = x[4]
        w = x[5]
        a_x = u[0]
        a_y = u[1]
        alpha = u[2]
        slack = u[3]

        speed = ca.norm_2(x[3:5])

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
        Wstatic = getattr(settings._params, "Wstatic")
        Wdynamic = getattr(settings._params, "Wdynamic")
        Wslack = getattr(settings._params, "Wslack")
        
        ## GOAL POSITION REACHING COST
        # Derive position error
        goal_dist_error = (pos[0] - goal_position[0]) ** 2 + (pos[1] - goal_position[1]) ** 2
        initial_goal_dist_error = (initial_pose[0] - goal_position[0]) ** 2 + (initial_pose[1] - goal_position[1]) ** 2 + 0.01
        goal_dist_error_normalized = goal_dist_error/initial_goal_dist_error    

        ## GOAL ORIENTATION COST 
        goal_orientation_error = min_angle_diff(psi,goal_orientation)**2 / (min_angle_diff(initial_pose[2],goal_orientation)**2 + 0.01)
        
        ## VELOCITY ORIENTATION COST
        # derive velocity vector angle
   
        vel_orientation = ca.if_else(v_x > 0.01, ca.atan2(v_y,v_x), ca.sign(v_y)*ca.pi/2)

        vel_orientation_error = min_angle_diff(psi,vel_orientation)**2/(min_angle_diff(initial_pose[2],vel_orientation)**2 + 0.01)
        
        rotation_car = helpers.rotation_matrix(psi)
        dist2constraint = 0
        for obst_id in range(self._n_static_obst):
            # A'x <= b
            a1_all = getattr(settings._params, "linear_constraint_" + str(obst_id) + "_a1")
            a2_all = getattr(settings._params, "linear_constraint_" + str(obst_id) + "_a2")
            b_all= getattr(settings._params, "linear_constraint_" + str(obst_id) + "_b")
            
            for disc in range(self._n_discs):
                disc_x = getattr(settings._params, "disc_" + str(disc) + "_offset")[0]
                disc_relative_pos = ca.vertcat(disc_x, 0)
                disc_pos = pos + rotation_car @ disc_relative_pos
                disc_pos_initial = initial_pose[:2] + rotation_car @ disc_relative_pos

                a1 = a1_all[disc]
                a2 = a2_all[disc]
                b = b_all[disc]
                
                disc_r = getattr(settings._params, "disc_" + str(disc) + "_r")
                
                norm_a = ca.norm_2(ca.vertcat(a1, a2))
                a1 = a1 / norm_a
                a2 = a2 / norm_a
                b = b / norm_a
                
                dist2constraint += 1/((a1 * disc_pos[0] + a2 * disc_pos[1] - b + disc_r)**2 + 0.01)
                disc2constraint_initial = 1/((a1 * disc_pos_initial[0] + a2 * disc_pos_initial[1] - b + disc_r)**2 + 0.01)
        
        dist2constraint = ca.fmin(dist2constraint/self._n_discs/self._n_static_obst,1)
        
        
        ## DYNAMIC OBSTACLE COST
        dyn_obst_cost = 0
        for obst_id in range(self._n_dyn_obst):
            obst_pos = getattr(settings._params, "ellipsoid_constraint_agent_" + str(obst_id) + "_pos")
            for disc in range(self._n_discs):
                disc_x = getattr(settings._params, "disc_" + str(disc) + "_offset")[0]
                disc_relative_pos = ca.vertcat(disc_x, 0)
                disc_pos = pos + rotation_car @ disc_relative_pos
                
                dist2obst = (disc_pos[0] - obst_pos[0]) ** 2 + (disc_pos[1] - obst_pos[1]) ** 2 + 0.01
                dyn_obst_cost += 1/dist2obst
            
        dyn_obst_cost = dyn_obst_cost/self._n_discs/self._n_dyn_obst
        
        if u.shape[0] >= 2:  # Todo check meaning
            if stage_idx == self._N + 1:
                cost = Wgoal_position * goal_dist_error_normalized + \
                    Wgoal_orientation * goal_orientation_error 
            elif stage_idx == 1:
                cost =  Wstatic * dist2constraint + \
                        Wdynamic * dyn_obst_cost + \
                        Wvel_orientation * vel_orientation_error + \
                        Wa * a_x * a_x + Wa * a_y * a_y + Wv * v_x * v_x + Wv* v_y * v_y +\
                        Wslack * slack * slack
            else:
                cost =  Wvel_orientation * vel_orientation_error + \
                    Wa * a_x * a_x + Wa * a_y * a_y  + Wv * v_x * v_x + Wv* v_y * v_y + \
                    Wslack * slack * slack
            
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



