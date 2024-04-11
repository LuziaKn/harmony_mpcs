import casadi
import numpy as np
import mpc_solver_generation.helpers as helpers


""" 
Defines inequality constraints of different types. 
See control_modules.py for their integration into the controller 
"""

# Class to aggregate the number of constraints and nh, nparam
class Constraints:

    def __init__(self, params):
        self.upper_bound = []
        self.lower_bound = []
        self.constraints = []

        self.constraint_modules = []

        self.nh = 0
        # self.npar = 0
        self.params = params
        # self.param_idx = param_idx

    def add_constraint(self, constraint):
        self.constraints.append(constraint) # param_idx keeps track of the indices
        constraint.append_upper_bound(self.upper_bound)
        constraint.append_lower_bound(self.lower_bound)

        self.nh += constraint.nh

    def inequality(self, z, param, settings, model):
        result = []

        for constraint in self.constraints:
            constraint.append_constraints(result, z, param, settings, model)

        return result

class LinearConstraints:

    def __init__(self, params, n_discs, num_constraints):
        self.num_constraints = num_constraints
        self.n_discs = n_discs
        self.params = params
        self.name = "linear_obst"

        self.nh = (num_constraints) * n_discs

        # @Todo: Be able to add multiple sets of constraints
        for disc in range(n_discs):
            params.add_parameter(self.constraint_name(disc) + "_a1" , num_constraints) # a1 a2 b
            params.add_parameter(self.constraint_name(disc) + "_a2" , num_constraints)  # a1 a2 b
            params.add_parameter(self.constraint_name(disc) + "_b" , num_constraints)  # a1 a2 b


    def constraint_name(self, disc_idx):
        return "disc_"+str(disc_idx)+"_linear_constraint"

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.num_constraints):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.num_constraints):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self, constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = np.array([x[0], x[1]])
        #slack = u[2]
        psi = x[2]


        #pos2 = np.array([x[10], x[11]])
        #pos3 = np.array([x[15], x[16]])

        rotation_car = helpers.rotation_matrix(psi)
        for disc_it in range(0, self.n_discs):
            disc_x = getattr(settings.params, "disc_offset")[disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos #+ rotation_car.dot(disc_relative_pos)

            # A'x <= b
            a1_all = getattr(settings.params, self.constraint_name(disc_it) + "_a1")
            a2_all = getattr(settings.params, self.constraint_name(disc_it) + "_a2")
            b_all= getattr(settings.params, self.constraint_name(disc_it) + "_b")
            for constraint_it in range(0, int(self.num_constraints)):
                a1 = a1_all[constraint_it]
                a2 = a2_all[constraint_it]
                b = b_all[constraint_it]

                radius = getattr(settings.params, "disc_r")
                #alpha = casadi.atan2(-a1,a2)
                #h = radius/casadi.sin(alpha)
                constraints.append(a1 * pos[0] + a2 * pos[1] - b)
                #constraints.append(a1 * pos1[0] + a2 * pos1[1] - (b-h*a2))
                #constraints.append(a1 * pos2[0] + a2 * pos2[1] - (b-h*a2))
                #constraints.append(a1 * pos3[0] + a2 * pos3[1] - (b-h*a2))

class InteractiveLinearConstraints:

    def __init__(self, params, n_discs, max_obstacles, num_constraints):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "interactive_linear_obst"

        self.nh = self.max_obstacles * n_discs



        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles+1):
                if num_constraints>0:
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_x", num_constraints)
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_y", num_constraints)


    def constraint_name(self, disc_idx, obst_id):
        return "disc_"+str(disc_idx)+"_interactive_linear_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[:2]

        r_robot = getattr(settings.params, 'disc_r')


        for i in range(0, self.max_obstacles):
            id = i+1
            r_obst = getattr(settings.params, "other_agents_pos_r_" + str(i))[-1]
            pos_obst = casadi.vertcat(x[id*5], x[id*5+1])
            diff = pos - pos_obst
            dist = approx_norm(diff)
            diff_normalized = diff/dist
            point = pos_obst + (r_obst + r_robot) * diff_normalized

            diff_point_pos = point - pos

            a = diff_point_pos
            a_norm = approx_norm(a)
            a_normalized = a /a_norm

            b = casadi.mtimes(a_normalized.T,point)

            # ax-b<=0
            constraints.append(a_normalized[0] * pos[0] + a_normalized[1] * pos[1] - b)
            print('constraint added agent' + str(id))


class InteractiveEllipsoidConstraints:

    def __init__(self, params, n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "interactive_ellipsoid_obst"

        self.nh = self.max_obstacles * n_discs

        self.num_states_ego_agent = num_states_ego_agent
        self.num_states_other_agent = num_states_other_agent

        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles+1):
                if num_constraints>0:
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_x", num_constraints)
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_y", num_constraints)


    def constraint_name(self, disc_idx, obst_id):
        return "disc_"+str(disc_idx)+"_interactive_ellipsoid_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[:2]

        r_robot = getattr(settings.params, 'disc_r')


        for i in range(0, self.max_obstacles):
            id = i+1
            r_obst = getattr(settings.params, "agents_pos_r_" + str(id))[-1]
            pos_obst = casadi.vertcat(x[self.num_states_ego_agent + self.num_states_other_agent*(id-1)], x[self.num_states_ego_agent + self.num_states_other_agent*(id-1)+1])
            diff = pos - pos_obst
            dist_approx = approx_norm(diff)
            dist_lowerbound = dist_approx-casadi.sqrt(0.01)



            constraints.append(-(dist_approx-r_obst-r_robot))
            print('constraint added agent' + str(id))

class InteractiveGaussianEllipsoidConstraints:

    def __init__(self, params, n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "interactive_ellipsoid_obst"

        self.nh = self.max_obstacles * n_discs

        self.num_states_ego_agent = num_states_ego_agent
        self.num_states_other_agent = num_states_other_agent

        params.add_parameter("sigma_x")
        params.add_parameter("sigma_y")
        params.add_parameter("epsilon")

        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles+1):
                if num_constraints>0:
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_x", num_constraints)
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_y", num_constraints)


    def constraint_name(self, disc_idx, obst_id):
        return "disc_"+str(disc_idx)+"_interactive_ellipsoid_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(0.)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(np.inf)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[:2]

        r_robot = getattr(settings.params, 'disc_r')

        sigma_x = getattr(settings.params, "sigma_x")
        sigma_y = getattr(settings.params, "sigma_y")
        Sigma = casadi.diag(casadi.vertcat(sigma_x**2, sigma_y**2))

        epsilon = getattr(settings.params, "epsilon")

        approx_epsilon = 0.001


        for i in range(0, self.max_obstacles):
            id = i+1
            r_obst = getattr(settings.params, "agents_pos_r_" + str(id))[-1]
            pos_obst = casadi.vertcat(x[self.num_states_ego_agent + self.num_states_other_agent*(id-1)], x[self.num_states_ego_agent + self.num_states_other_agent*(id-1)+1])
            diff = pos - pos_obst

            combined_radius = r_obst + r_robot

            a_ij = diff / approx_norm(diff, 0.001)

            b_ij = combined_radius

            x_erfinv = 1. - 2. * epsilon

            z = casadi.sqrt(-casadi.log((1.0 - x_erfinv) / 2.0))
            # Manual inverse erf, because somehow lacking from casadi...
            # From here: http: // casadi.sourceforge.net / v1.9.0 / api / internal / d4 / d99 / casadi__calculus_8hpp_source.html  # l00307
            y_erfinv = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454) / \
                       ((1.637067800 * z + 3.543889200) * z + 1.0)
            y_erfinv = y_erfinv - (casadi.erf(y_erfinv) - x_erfinv) / (
                    2.0 / casadi.sqrt(casadi.pi) * casadi.exp(-(y_erfinv * y_erfinv)))
            y_erfinv = y_erfinv - (casadi.erf(y_erfinv) - x_erfinv) / (
                    2.0 / casadi.sqrt(casadi.pi) * casadi.exp(-(y_erfinv * y_erfinv)))

            constraints.append(a_ij.T @ casadi.SX(diff) - b_ij - y_erfinv * casadi.sqrt(2. * a_ij.T @ Sigma @ a_ij+approx_epsilon) )


class FixedEllipsoidConstraints:

    def __init__(self, params, n_discs, max_obstacles, num_constraints):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "fixed_ellipsoid_obst"

        self.nh = self.max_obstacles * n_discs

        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles+1):
                if num_constraints>0:
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_x", num_constraints)
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_y", num_constraints)


    def constraint_name(self, disc_idx, obst_id):
        return "disc_"+str(disc_idx)+"_fixed_ellipsoid_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[:2]

        r_robot = getattr(settings.params, 'disc_r')


        for i in range(0, self.max_obstacles):
            id = i+1
            r_obst = getattr(settings.params, "agents_pos_r_" + str(id))[-1]
            pos_obst = getattr(settings.params, "agents_pos_r_" + str(id))[:2]
            diff = pos - pos_obst
            dist = casadi.sqrt(diff[0]**2 + diff[1]**2)

            constraints.append(-(dist-r_obst-r_robot)-u[2])
            print('constraint added agent' + str(id))


class FixedLinearConstraints:

    def __init__(self, params, n_discs, max_obstacles):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs

        self.nh = self.max_obstacles * n_discs
        self.name = "fixed_linear_obst"

        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles):
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_predictions", 2)


    def constraint_name(self, disc_idx, constraint_idx):
        return "disc_"+str(disc_idx)+"_fixed_linear_constraint_"+str(constraint_idx)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos =casadi.vertcat(x[0], x[1])
        heading = x[2]
        speed = x[3]


        r_robot = getattr(settings.params, 'disc_r')


        for disc_id in range(self.n_discs):
            for i in range(self.max_obstacles):
                r_obst = getattr(settings.params, "agents_pos_r_" + str(i+1))[-1]
                pos_obst = getattr(settings.params, "agents_pos_r_" + str(i+1))[:2]#getattr(settings.params,self.constraint_name(disc_id, i) + "_predictions")
                diff = pos - pos_obst
                diff_norm = approx_norm(diff)
                diff_normalized = diff/ diff_norm
                point = pos_obst + (r_obst + r_robot) * diff_normalized  # todo consider different radii

                diff_point_pos = point - pos
                diff_point_pos_norm = approx_norm(diff_point_pos)

                a = -diff_normalized * diff_point_pos_norm
                a_norm = approx_norm(a)

                a_normalized = a / a_norm

                b = a_normalized.T @ point

                constraints.append(a_normalized[0] * pos[0] + a_normalized[1] * pos[1] - b)

class StaticLinearConstraints:

    def __init__(self, params, n_discs, max_obstacles):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs

        self.nh = 2 * n_discs
        self.name = "fixed_linear_obst"


    def constraint_name(self, disc_idx, constraint_idx):
        return "disc_"+str(disc_idx)+"_fixed_linear_constraint_"+str(constraint_idx)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, 2):
            for disc in range(0, self.n_discs):
                lower_bound.append(0)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, 2):
            for disc in range(0, self.n_discs):
                upper_bound.append(casadi.inf)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos =casadi.vertcat(x[0], x[1])
        heading = x[2]
        speed = x[3]


        r_robot = getattr(settings.params, 'disc_r')

        linear_constraints = [casadi.vertcat(0, 1, -1, 0), casadi.vertcat(0, -1, -1, 0)]
        for disc_id in range(self.n_discs):
            for i in range(len(linear_constraints)):
                linear_constraint = linear_constraints[i]

                pos = x[:2]
                linear_constraint_point = linear_constraint[0:2]
                linear_constraint_vector = linear_constraint[2:4]
                closest_pt = linear_constraint_point + casadi.dot(pos - linear_constraint_point,
                                                                linear_constraint_vector) * linear_constraint_vector
                diff_point_pos = closest_pt - pos
                diff_point_pos_norm = casadi.norm_2(diff_point_pos)

                #a_normalized = diff_point_pos / diff_point_pos_norm

                #b = a_normalized.T @ (closest_pt)

                #constraints.append(a_normalized[0] * pos[0] + a_normalized[1] * pos[1] - b)
                constraints.append(diff_point_pos_norm-r_robot)


class FixedGaussianEllipsoidConstraints:

    def __init__(self, params, n_discs, max_obstacles, num_constraints):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "fixed_ellipsoid_obst"

        self.nh = self.max_obstacles * n_discs

        params.add_parameter("sigma_x")
        params.add_parameter("sigma_y")
        params.add_parameter("epsilon")

        for disc in range(n_discs):
            for obst_id in range(0, max_obstacles+1):
                if num_constraints>0:
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_x", num_constraints)
                    params.add_parameter(self.constraint_name(disc, obst_id) + "_closest_points_y", num_constraints)


    def constraint_name(self, disc_idx, obst_id):
        return "disc_"+str(disc_idx)+"_fixed_ellipsoid_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(0.)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(np.inf)

    def append_constraints(self,constraints, z, param, settings, model):
        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[:2]

        r_robot = getattr(settings.params, 'disc_r')

        # Retrieve covariance
        sigma_x = getattr(settings.params, "sigma_x")
        sigma_y = getattr(settings.params, "sigma_y")
        Sigma = casadi.diag(casadi.vertcat(sigma_x**2, sigma_y**2))

        epsilon = getattr(settings.params, "epsilon")
        approx_epsilon =0.01

        for disc_id in range(self.n_discs):
            for i in range(self.max_obstacles):
                r_obst = getattr(settings.params, "agents_pos_r_" + str(i+1))[-1]
                pos_obst = getattr(settings.params, "agents_pos_r_" + str(i+1))[:2]#getattr(settings.params,self.constraint_name(disc_id, i) + "_predictions")
                diff = pos_obst-pos
                diff = pos - pos_obst

                combined_radius = r_obst + r_robot

                a_ij = diff / approx_norm(diff)

                b_ij = combined_radius

                x_erfinv = 1. - 2. * epsilon

                z = casadi.sqrt(-casadi.log((1.0 - x_erfinv) / 2.0))
                # Manual inverse erf, because somehow lacking from casadi...
                # From here: http: // casadi.sourceforge.net / v1.9.0 / api / internal / d4 / d99 / casadi__calculus_8hpp_source.html  # l00307
                y_erfinv = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454) / \
                           ((1.637067800 * z + 3.543889200) * z + 1.0)
                y_erfinv = y_erfinv - (casadi.erf(y_erfinv) - x_erfinv) / (
                        2.0 / casadi.sqrt(casadi.pi) * casadi.exp(-(y_erfinv * y_erfinv)))
                y_erfinv = y_erfinv - (casadi.erf(y_erfinv) - x_erfinv) / (
                        2.0 / casadi.sqrt(casadi.pi) * casadi.exp(-(y_erfinv * y_erfinv)))

                constraints.append(a_ij.T @ casadi.SX(diff) - b_ij - y_erfinv * casadi.sqrt(
                    2. * a_ij.T @ Sigma @ a_ij+approx_epsilon) )
# class EllipsoidConstraints:
#
#     def __init__(self, n_discs, max_obstacles, params, rotation_clockwise=True):
#         self.max_obstacles = max_obstacles
#         self.n_discs = n_discs
#
#         self.nh = max_obstacles * n_discs
#
#         # Add parameters
#         for obs_id in range(max_obstacles):
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_x")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_y")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_psi")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_major")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_minor")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_chi")
#             params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_r")
#
#
#
#         self.rotation_clockwise = rotation_clockwise
#
#     def append_lower_bound(self, lower_bound):
#         for obs in range(0, self.max_obstacles):
#             for disc in range(0, self.n_discs):
#                 lower_bound.append(1.0)
#
#     def append_upper_bound(self, upper_bound):
#         for obs in range(0, self.max_obstacles):
#             for disc in range(0, self.n_discs):
#                 upper_bound.append(np.Inf)
#
#     def append_constraints(self, constraints, z, param, settings, model):
#
#         settings.params.load_params(param)
#
#         # Retrieve variables
#         x = z[model.nu:model.nu + model.nx]
#         u = z[0:model.nu]
#
#         # States
#         pos = np.array([x[0], x[1]])
#         psi = x[2]
#         slack = u[-1]
#
#         rotation_car = helpers.rotation_matrix(psi)
#
#         r_disc = getattr(settings.params, 'disc_r') #param[self.start_param]
#
#         # Constraint for dynamic obstacles
#         for obstacle_it in range(0, self.max_obstacles):
#             # Retrieve parameters
#             obst_x = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_x")
#             obst_y = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_y")
#             obst_psi = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_psi")
#             obst_major = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_major")
#             obst_minor = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_minor")
#             obst_r = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_r")
#
#             # multiplier for the risk when obst_major, obst_major only denote the covariance
#             # (i.e., the smaller the risk, the larger the ellipsoid).
#             # This value should already be a multiplier (using exponential cdf).
#             chi = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_it) + "_chi")
#
#             # obstacle computations
#             obstacle_cog = np.array([obst_x, obst_y])
#
#             # Compute ellipse matrix
#             obst_major *= casadi.sqrt(chi)
#             obst_minor *= casadi.sqrt(chi)
#             ab = np.array([[1. / ((obst_major + (r_disc + obst_r)) ** 2), 0],
#                            [0, 1. / ((obst_minor + (r_disc + obst_r)) ** 2)]])
#
#             # In the original LMPCC paper the angle of the obstacles is defined clockwise
#             # While it could also make sense to define this anti-clockwise, just like the orientation of the Roboat
#             if self.rotation_clockwise:
#                 obstacle_rotation = helpers.rotation_matrix(obst_psi)
#             else:
#                 obstacle_rotation = helpers.rotation_matrix(-obst_psi)
#
#             obstacle_ellipse_matrix = obstacle_rotation.transpose().dot(ab).dot(obstacle_rotation)
#
#             for disc_it in range(0, self.n_discs):
#                 # Get and compute the disc position
#                 disc_x = getattr(settings.params, 'disc_offset_' + str(disc_it))
#                 disc_relative_pos = np.array([disc_x, 0])
#                 disc_pos = pos + rotation_car.dot(disc_relative_pos)
#
#                 # construct the constraint and append it
#                 disc_to_obstacle = disc_pos - obstacle_cog
#                 c_disc_obstacle = disc_to_obstacle.transpose().dot(obstacle_ellipse_matrix).dot(disc_to_obstacle)
#                 constraints.append(c_disc_obstacle + slack)


class MultiEllipsoidConstraints:

    def __init__(self, n_discs, max_obstacles, params, rotation_clockwise=True):
        self.max_obstacles = max_obstacles
        self.n_discs = n_discs

        self.nh = max_obstacles * n_discs

        # Add parameters
        for obs_id in range(max_obstacles):
            params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_major")
            params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_minor")
            params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_chi")
            params.add_parameter("ellipsoid_obst_" + str(obs_id) + "_r")



        self.rotation_clockwise = rotation_clockwise

    def append_lower_bound(self, lower_bound):
        for obs in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(1.0)

    def append_upper_bound(self, upper_bound):
        for obs in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(np.Inf)

    def append_constraints(self, constraints, z, param, settings, model):

        settings.params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = np.array([x[0], x[1]])
        psi = x[2]
        slack = u[-1]

        rotation_car = helpers.rotation_matrix(psi)

        r_disc = getattr(settings.params, 'disc_r') #param[self.start_param]

        # Constraint for dynamic obstacles
        for obstacle_id in range(self.max_obstacles): #todo needs to be adapted for vaiable number of agents
            # Retrieve parameters
            obst_x = x[(obstacle_id+1)*5]
            obst_y = x[(obstacle_id+1)*5+1]
            obst_psi = x[(obstacle_id+1)*5+2]
            obst_major = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_id) + "_major")
            obst_minor = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_id) + "_minor")
            obst_r = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_id) + "_r")

            # multiplier for the risk when obst_major, obst_major only denote the covariance
            # (i.e., the smaller the risk, the larger the ellipsoid).
            # This value should already be a multiplier (using exponential cdf).
            chi = getattr(settings.params, "ellipsoid_obst_" + str(obstacle_id) + "_chi")

            # obstacle computations
            obstacle_cog = np.array([obst_x, obst_y])

            # Compute ellipse matrix
            obst_major *= casadi.sqrt(chi)
            obst_minor *= casadi.sqrt(chi)
            ab = np.array([[1. / ((obst_major + (r_disc)) ** 2), 0],
                           [0, 1. / ((obst_minor + (r_disc)) ** 2)]])

            # In the original LMPCC paper the angle of the obstacles is defined clockwise
            # While it could also make sense to define this anti-clockwise, just like the orientation of the Roboat
            if self.rotation_clockwise:
                obstacle_rotation = helpers.rotation_matrix(obst_psi)
            else:
                obstacle_rotation = helpers.rotation_matrix(-obst_psi)

            obstacle_ellipse_matrix = obstacle_rotation.transpose().dot(ab).dot(obstacle_rotation)

            for disc_it in range(0, self.n_discs):
                # Get and compute the disc position
                disc_x = getattr(settings.params, 'disc_offset_' + str(disc_it))
                disc_relative_pos = np.array([disc_x, 0])
                disc_pos = pos + rotation_car.dot(disc_relative_pos)

                # construct the constraint and append it
                disc_to_obstacle = disc_pos - obstacle_cog
                c_disc_obstacle = disc_to_obstacle.transpose().dot(obstacle_ellipse_matrix).dot(disc_to_obstacle)
                constraints.append(c_disc_obstacle + slack)

