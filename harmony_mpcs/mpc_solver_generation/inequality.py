import casadi as ca
import numpy as np
import harmony_mpcs.mpc_solver_generation.helpers as helpers


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

    def __init__(self, params, n_discs, n_obst):
        self._n_obst = n_obst
        self._n_discs = n_discs
        self.params = params
        self.name = "linear_constr"
        print('n_discs', self._n_discs)
        print('n_static_obst', self._n_obst)

        self.nh = (self._n_obst) * n_discs

        # @Todo: Be able to add multiple sets of constraints
        for obst_id in range(self._n_obst):
            params.add_parameter(self.constraint_name(obst_id) + "_a1" , 1) # a1 a2 b
            params.add_parameter(self.constraint_name(obst_id) + "_a2" , 1)  # a1 a2 b
            params.add_parameter(self.constraint_name(obst_id) + "_b" , 1)  # a1 a2 b

    def constraint_name(self, obst_id):
        return "linear_constraint_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for _ in range(self._n_obst):
            for disc in range(self._n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for _ in range(self._n_obst):
            for disc in range(self._n_discs):
                upper_bound.append(0.0)

    def append_constraints(self, constraints, z, param, settings, model):
        settings._params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = x[0:2]
        psi = x[2]

        rotation_car = helpers.rotation_matrix(psi)
        for obst_id in range(self._n_obst):
            # A'x <= b
            a1_all = getattr(settings._params, self.constraint_name(obst_id) + "_a1")
            a2_all = getattr(settings._params, self.constraint_name(obst_id) + "_a2")
            b_all= getattr(settings._params, self.constraint_name(obst_id) + "_b")
            for disc in range(self._n_discs):
                disc_x = getattr(settings._params, "disc_" + str(disc) + "_offset")[0]
                disc_relative_pos = ca.vertcat(disc_x, 0)
                disc_pos = pos + rotation_car @ disc_relative_pos

                a1 = a1_all[disc]
                a2 = a2_all[disc]
                b = b_all[disc]

                disc_r = getattr(settings._params, "disc_" + str(disc) + "_r")
                constraints.append(a1 * disc_pos[0] + a2 * disc_pos[1] - b + disc_r)



class FixedEllipsoidConstraints:
    

    def __init__(self, params, n_discs, max_obstacles):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.name = "fixed_ellipsoid_obst"

        print(self.name)

        self.nh = self.max_obstacles * n_discs

        
        for obst_id in range(max_obstacles):
            params.add_parameter(self.constraint_name(obst_id) + "_pos", 2)
            params.add_parameter(self.constraint_name(obst_id) + "_r", 1)
            params.add_parameter(self.constraint_name(obst_id) + "_psi", 1)
            params.add_parameter(self.constraint_name(obst_id) + "_major", 1)
            params.add_parameter(self.constraint_name(obst_id) + "_minor", 1)
            params.add_parameter(self.constraint_name(obst_id) + "_chi", 1)


    def constraint_name(self, obst_id):
        return "ellipsoid_constraint_agent_" + str(obst_id)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(1.0)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(np.Inf)

    def append_constraints(self,constraints, z, param, settings, model):
        settings._params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]
        slack = u[-1]

        # States
        pos = x[0:2]
        psi = x[2]

        rotation_car = helpers.rotation_matrix(psi)

        for i in range(self.max_obstacles):
            obst_r = getattr(settings._params, self.constraint_name(i) + "_r")[0]
            obst_pos = getattr(settings._params, self.constraint_name(i) + "_pos")[:2]
            obst_psi = getattr(settings._params, self.constraint_name(i) + "_psi")[0]
            obst_major = getattr(settings._params, self.constraint_name(i) + "_major")[0]
            obst_minor = getattr(settings._params, self.constraint_name(i) + "_minor")[0]
            chi = getattr(settings._params, self.constraint_name(i) + "_chi")[0]

            for disc in range(self.n_discs):              
                disc_r = getattr(settings._params, "disc_" + str(disc) + "_r")
                disc_offset = getattr(settings._params, "disc_" + str(disc) + "_offset")

                # Compute ellipse matrix
                obst_major *= ca.sqrt(chi)
                obst_minor *= ca.sqrt(chi)
                ab = ca.vertcat(ca.horzcat(1. / ((obst_major + (disc_r + obst_r)) ** 2), 0),
                           ca.horzcat(0, 1. / ((obst_minor + (disc_r + obst_r)) ** 2)))

                obstacle_rotation = helpers.rotation_matrix(obst_psi)
                obstacle_ellipse_matrix = ca.mtimes(ca.mtimes(obstacle_rotation.T,ab),obstacle_rotation)

                disc_relative_pos = ca.vertcat(disc_offset, 0)
                disc_pos = pos + ca.mtimes(rotation_car,disc_relative_pos)

                disc_to_obstacle = disc_pos - obst_pos
                c_disc_obstacle = ca.mtimes(disc_to_obstacle.T,ca.mtimes(obstacle_ellipse_matrix,disc_to_obstacle))

                A = (obst_pos - disc_pos) / ca.norm_2(disc_pos - obst_pos)
                b = A.T @ (obst_pos - A*(obst_r + disc_r))
                constraints.append(c_disc_obstacle + slack)
                print('constraint added agent' + str(i+1))


class FixedLinearConstraints:

    def __init__(self, params, n_discs, max_obstacles):

        self.max_obstacles = max_obstacles
        self.n_discs = n_discs

        self.nh = self.max_obstacles * n_discs
        self.name = "fixed_linear_obst"

        for obst_id in range(0, max_obstacles):
            params.add_parameter(self.constraint_name(obst_id) + "_predictions", 2)


    def constraint_name(self, disc_idx, constraint_idx):
        return "linear_constraint_"+str(constraint_idx)

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self,constraints, z, param, settings, model):
        settings._params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos =casadi.vertcat(x[0], x[1])
        heading = x[2]
        speed = x[3]


        r_robot = getattr(settings._params, 'disc_r')


        for disc_id in range(self.n_discs):
            for i in range(self.max_obstacles):
                r_obst = getattr(settings._params, "agents_pos_r_" + str(i+1))[-1]
                pos_obst = getattr(settings._params, "agents_pos_r_" + str(i+1))[:2]#getattr(settings.params,self.constraint_name(disc_id, i) + "_predictions")
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


    def constraint_name(self, constraint_idx):
        return "linear_constraint_"+str(constraint_idx)

    def append_lower_bound(self, lower_bound):
            for disc in range(0, self.n_discs):
                lower_bound.append(0)

    def append_upper_bound(self, upper_bound):
            for disc in range(0, self.n_discs):
                upper_bound.append(casadi.inf)

    def append_constraints(self,constraints, z, param, settings, model):
        settings._params.load_params(param)

        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos =casadi.vertcat(x[0], x[1])
        heading = x[2]
        speed = x[3]


        r_robot = getattr(settings._params, 'disc_r')

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


