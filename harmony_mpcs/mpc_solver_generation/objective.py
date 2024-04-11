"""
Objective.py defines the objective for the solver. Currently it is system specific, it should be general.
@Todo: Generalize the objectives used.
"""

import numpy as np
import casadi as ca


import config


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


class GOMPCObjective:

    def __init__(self, params, system, config):
        self.define_parameters(params)
        self.system = system
        self.config = config


    def define_parameters(self, params):
        params.add_parameter("goal", 3)
        params.add_parameter("Wrepulsive")
        params.add_parameter("Wx")
        params.add_parameter("Wy")
        params.add_parameter("Walpha")
        params.add_parameter("Wtheta")
        params.add_parameter("Wa")
        params.add_parameter("Ws")
        params.add_parameter("Wv")
        params.add_parameter("Ww")
        params.add_parameter("Wothers")



    def get_value(self, x, u, settings, stage_idx):

        pos_x = x[0]
        pos_y = x[1]
        pos = x[:2]
        phi = x[2]
        v = x[3]
        w = x[4]

        a = u[0]
        alpha = u[1]

        vel = ca.vertcat(v*ca.cos(phi), v*ca.sin(phi))

        # Parameters
        goal = getattr(settings.params, "goal")


        Wrepulsive = getattr(settings.params, "Wrepulsive")
        Wx = getattr(settings.params, "Wx")
        Wy = getattr(settings.params, "Wy")
        Walpha = getattr(settings.params, "Walpha")
        Wa = getattr(settings.params, "Wa")
        Ws = getattr(settings.params, "Ws")
        Wv = getattr(settings.params, "Wv")
        Ww = getattr(settings.params, "Ww")
        Wothers = getattr(settings.params, "Wothers")

        r_other = getattr(settings.params, "agents_pos_r_1")[-1]
        r_robot = getattr(settings.params, "disc_r")

        # Derive position error
        start_pos = getattr(settings.params, "start_state_0")[:2]
        initial_pos = getattr(settings.params, "initial_state_0")[:2]

        goal_dist_error = (pos[0]-goal[0])**2 + (pos[1]-goal[1])**2

        # get future state based on constant velocity
        goal_heading = ca.atan2(goal[1] - start_pos[1], goal[0] - start_pos[0])

        # find closest postition to intitial position on line from start_pos to goal
        dx = goal[0] - start_pos[0]
        dy = goal[1] - start_pos[1]
        x3, y3 = initial_pos[0], initial_pos[1]
        t = ((x3 - start_pos[0]) * dx + (y3 - start_pos[1]) * dy) / (dx ** 2 + dy ** 2)
        closest_point = start_pos + t * (goal[:2]- start_pos)
        goal_dist = approx_min([approx_norm(goal[:2] - closest_point),4])
        goal_dist_increment = goal_dist/(self.config.MPCConfig.FORCES_N+1)
        pos_constant_vel = closest_point + (stage_idx+1) * goal_dist_increment * ca.vertcat(ca.cos(goal_heading), ca.sin(goal_heading))
        error_constant_vel = (pos[0] - pos_constant_vel[0]) ** 2 + (pos[1] - pos_constant_vel[1]) ** 2

        pref_speed = self.config.RobotConfig.upper_bound['v']
        speed_error = (v - pref_speed) ** 2

        radii = [r_robot]
        for i in range(1, int(x.shape[0] / 5)):
            radii.append(getattr(settings.params, "agents_pos_r_" + str(i))[-1])

        total_acc = 0

        goal_error =  goal_dist_error
        cost_closeness = 0.0
        goal_dist_error_others = 0.0
        total_error_constant_velocity = 0.0
        encouraging_decision_cost = ca.SX.zeros(self.config.MPCConfig.N_AGENTS_MPC-1,1)
        influencing_right = 0.0
        for id in range(1, self.config.MPCConfig.N_AGENTS_MPC):
            pos_obst = x[5*(id):5*id + 2]
            vel_obst = x[5*(id)+3:5*id + 5]
            speed_obst = approx_norm(vel_obst)
            r_obst = getattr(settings.params, "agents_pos_r_" + str(id))[-1]
            combined_radius = r_robot + r_obst
            diff = pos - pos_obst
            dist = approx_norm(diff)
            cost_closeness += 1 / dist


            #compute goal error others
            goal_obst = getattr(settings.params, "other_agents_hidden_params_" + str(id))[1:3]
            goal_dist_error_others += (pos_obst[0] - goal_obst[0]) ** 2 + (pos_obst[1] - goal_obst[1]) ** 2

            # cost other agents deviation from constant velocity
            other_agents_hidden_params = getattr(settings.params, "other_agents_hidden_params_" + str(id))
            goal_other = other_agents_hidden_params[1:3]
            initial_pos_other = getattr(settings.params, "initial_state_" + str(id))[:2]
            goal_heading_other = ca.atan2(goal_other[1] - initial_pos_other[1], goal_other[0] - initial_pos_other[0])
            goal_dist = approx_min([approx_norm(goal_other[:2] - initial_pos_other),4])
            goal_dist_increment = goal_dist / (self.config.MPCConfig.FORCES_N + 1)
            pos_constant_vel_other = initial_pos_other + stage_idx * goal_dist_increment * ca.vertcat(ca.cos(goal_heading_other), ca.sin(goal_heading_other)) * settings.integrator_stepsize * (stage_idx + 1)
            error_constant_vel_other = approx_norm(pos_obst - pos_constant_vel_other)
            total_error_constant_velocity += error_constant_vel_other

            # velocity change for other agents
            other_agents_hidden_params = getattr(settings.params, "other_agents_hidden_params_" + str(id))
            pref_speed = other_agents_hidden_params[0]
            sfm_pref_add_dist = other_agents_hidden_params[3]
            sfm_param_social = other_agents_hidden_params[4]



            side_preference = getattr(settings.params, "side_preference_" + str(id))

            goal_dist = approx_norm(goal_obst - pos_obst)
            m = -6

            def decreasing_sigmoid(x, m):
                return 1 / (1 + np.exp(m * approx_abs(x - 1)))

            acc_limit = getattr(settings, "sfm_acc_limit")
            pref_speed_reduced = pref_speed  # approx_min(ca.vertcat(1, goal_dist/(1.5**2)))
            a_sfm_variable, theta = sfm_step(joint_state=x, index=id, radii=radii, desired_speed=pref_speed_reduced,
                                      goal=goal_obst, pref_add_dist=sfm_pref_add_dist, pref_side=side_preference,
                                      social_weight=sfm_param_social, acc_limit=acc_limit)

            encouraging_decision_cost[id-1,0] = ca.fabs(theta)

            total_acc += approx_norm(a_sfm_variable)

            influencing_right += approx_sign(pos_obst[1] - pos[1])

        goal_error_others = goal_dist_error_others
        min_encouraging_decision_cost = ca.mmin(encouraging_decision_cost)
        if u.shape[0] >=2: #Todo check meaning
            if stage_idx == self.config.MPCConfig.FORCES_N+1:
                # cost = (Wx*(x_ratio+y_ratio)
                #         + Wothers*total_error_ratio_others/(self.config.MPCConfig.N_AGENTS_MPC-1)**2)
                cost = Wx* goal_error +Wothers * (0.5-speed_obst)**2+ Wrepulsive*cost_closeness
            else:
                # cost = (Wx*(x_ratio+y_ratio)
                #         + Wothers*total_error_ratio_others/(self.config.MPCConfig.N_AGENTS_MPC-1)**2)
                cost = Wx* goal_error
                + Walpha * alpha * alpha + Wa * a * a + Ws* u[2]**2
                + Wothers * (0.5-speed_obst) + Wrepulsive*cost_closeness

        else: print("not implemented yet")



        return cost



class FixedMPCObjective:

    def __init__(self, params, system, config):
        self.define_parameters(params)
        self.system = system
        self.config = config


    def define_parameters(self, params):
        params.add_parameter("goal", 3)
        params.add_parameter("Wrepulsive")
        params.add_parameter("Wx")
        params.add_parameter("Wy")
        params.add_parameter("Walpha")
        params.add_parameter("Wtheta")
        params.add_parameter("Wa")
        params.add_parameter("Ws")
        params.add_parameter("Wv")
        params.add_parameter("Ww")
        params.add_parameter("Wothers")



    def get_value(self, x, u, settings, stage_idx):

        # print(stage_idx)
        cost = 0
        # if stage_idx == settings.N_bar - 1:  # settings.N - 1:
        # if stage_idx == settings.N - 1:
        if True:
            pos_x = x[0]
            pos_y = x[1]
            pos = x[:2]
            v = x[3]
            a = u[0]
            alpha = u[1]


            # Parameters
            goal = getattr(settings.params, "goal")

            Wrepulsive = getattr(settings.params, "Wrepulsive")
            Wx = getattr(settings.params, "Wx")
            Wy = getattr(settings.params, "Wy")
            Walpha = getattr(settings.params, "Walpha")
            Wa = getattr(settings.params, "Wa")
            Ws = getattr(settings.params, "Ws")
            Wv = getattr(settings.params, "Wv")
            Ww = getattr(settings.params, "Ww")
            Wothers = getattr(settings.params, "Wothers")

            r_robot = getattr(settings.params, 'disc_r')

            initial_pos = getattr(settings.params, "initial_state_0")[:2]
            start_pos = getattr(settings.params, "start_state_0")[:2]



            # Derive position error
            goal_dist_error = (pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2




           
            if u.shape[0] >= 2:  # Todo check meaning
                if stage_idx == self.config.MPCConfig.FORCES_N + 1:
                    cost = Wx * goal_dist_error 
                else:
                    cost =  Wx * goal_dist_error
                    + Walpha * alpha * alpha + Wa * a * a
            else:
                print("not implemented yet")

        return cost








def objective(z, param, model, settings, stage_idx):
    # print("stage idx in jackal_objective: {}".format(stage_idx))
    cost = 0.
    settings.params.load_params(param)

    # Retrieve variables
    x = z[model.nu:model.nu + model.nx]
    u = z[0:model.nu]



    # Weights
    settings.weights.set_weights(param)

    for module in settings.modules.modules:
        if module.type == "objective":
            for module_objective in module.objectives:
                cost += module_objective.get_value(x, u, settings, stage_idx)

    #cost += settings.weights.velocity * ((v - settings.weights.velocity_reference) ** 2)

    # cost += settings.weights.acceleration * (a ** 2)  # /(model.upper_bound()[0] - model.lower_bound()[0])
    #cost += settings.weights.angular_velocity * (w ** 2)  # /(model.upper_bound()[1] - model.lower_bound()[1])
    return cost



