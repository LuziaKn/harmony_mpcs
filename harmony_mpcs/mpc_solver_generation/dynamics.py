import copy

import numpy as np
import casadi
from typing import List
import forcespro.nlp
import casadi as ca

from harmony_mpcs.mpc_solver_generation.sfm import SocialForcesPolicy
from harmony_mpcs.utils.utils import get_agent_states




# Returns discretized dynamics of a given model (see below)
def rk4_discretization(continuous_system, x, u, dt, *args):
    k1 = continuous_system(x, u, *args)
    k2 = continuous_system(x + 0.5 * dt * k1, u, *args)
    k3 = continuous_system(x + 0.5 * dt * k2, u, *args)
    k4 = continuous_system(x + dt * k3, u, *args)

    next_state = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state
def discrete_dynamics(z, model, integrator_stepsize, p= None, settings= None):
    """

    @param z: state vector (u, x)
    @param model: Model of the system
    @param integrator_stepsize: Integrator stepsize in seconds
    @return:
    """
    # We use an explicit RK4 integrator here to discretize continuous dynamics

    result = forcespro.nlp.integrate(
        model.continuous_model,
        z[model.nu:],
        z[0:model.nu],
        p, settings,
        integrator=forcespro.nlp.integrators.RK4,
        stepsize=integrator_stepsize,
        )

    #result = rk4_discretization(model.continuous_model, z[model.nu:model.nu+model.nx], z[0:model.nu],integrator_stepsize, p, settings )

    return result

def add_dual(model, nh):
    model.nu += nh
    model.continuous_model = lambda x, u: casadi.vstack(model.continuous_model(x, u), np.zeros((nh, 1))) # lambdas have no dynamics

# Dynamics, i.e. equality constraints #
# This class contains models to choose from
# They can be coupled with physical limits using Systems defined in systems.py
# See Bicycle model for an example of the required parameters
class DynamicModel:

    def __init__(self,robot_config):
        self.nvar = self.nu + self.nx
        self.robot_config = robot_config

    def __str__(self):
        print(self.robot_config)
        result = 'Dynamical Model: ' + str(type(self)) + '\n' +\
               'System: ' + str(self.robot_config['robot_name']) + '\n'

        if hasattr(self, 'interfaces'):
            result += 'Interfaces: '

            for interface in self.interfaces:
                result += interface + " "

            result += "\n"

        result += 'States: ' + str(self.states) + '\n'
        result += 'Inputs: ' + str(self.inputs) + '\n'
        return result

    # Appends upper bounds from system
    def upper_bound(self):
        result = np.array([])

        for input in self.inputs:
            result = np.append(result, self.robot_config['input_constraints']['upper_bounds'][input])

        for state in self.states:
            result = np.append(result, self.robot_config['state_constraints']['upper_bounds'][state])

        return result

    # Appends lower bounds from system
    def lower_bound(self):
        result = np.array([])

        for input in self.inputs:
            result = np.append(result, self.robot_config['input_constraints']['lower_bounds'][input])

        for state in self.states:
            result = np.append(result, self.robot_config['state_constraints']['lower_bounds'][state])

        return result


# Second-order Bicycle model
class BicycleModel(DynamicModel):

    def __init__(self, robot_config):
        self.nu = 3 # number of control variables
        self.nx = 5 # number of states

        super(BicycleModel, self).__init__(robot_config)

        self.states = ['x', 'y', 'psi', 'v', 'w']  # , 'ax', 'ay'
        self.states_from_sensor = [True, True, True, True, True]  # , True, True
        self.states_from_sensor_at_infeasible = [True, True, True, True, True]  # False variables are guessed 0 at infeasible

        self.inputs = ['a', 'alpha', 'slack']
        self.inputs_to_vehicle = [True, True, False]
        self.possible_inputs_to_vehicle = ['v', 'a' ,'alpha','w', 'psi']

    def continuous_model(self, x, u, *args):
        a = u[0]
        alpha = u[1]
        psi = x[2]
        v = x[3]
        w = x[4]

        return ca.vertcat(v * casadi.cos(psi),
                         v * casadi.sin(psi),
                         w,
                         a,
                         alpha)

class PointMass_2order_Model(DynamicModel):

    def __init__(self, robot_config):
        self.nu = 4 # number of control variables
        self.nx = 6 # number of states

        super(PointMass_2order_Model, self).__init__(robot_config)

        self.states = ['x', 'y', 'psi', 'v_x', 'v_y', 'w']  # , 'ax', 'ay'
        self.states_from_sensor = [True, True, True, True, True, True]  # , True, True
        self.states_from_sensor_at_infeasible = [True, True, True, True, True, True]  # False variables are guessed 0 at infeasible

        self.inputs = ['a_x', 'a_y', 'alpha', 'slack']
        self.inputs_to_vehicle = [True, True, False]
        self.possible_inputs_to_vehicle = ['a_x', 'a_y']

    def continuous_model(self, x, u, *args):


        a_x = u[0]
        a_y = u[1]
        alpha = u[2]
        psi = x[2]
        v_x = x[3]
        v_y = x[4]
        w = x[5]

        return ca.vertcat(v_x,
                         v_y,
                         w,
                         a_x,
                         a_y,
                         alpha)

class CombinedDynamics(DynamicModel):

    def __init__(self, models: List, params, robot_config):

        self.robot_config = robot_config

        self.nu = models[0].nu# number of control variables
        self.nx = sum([model.nx for model in models]) # number of states

        self.states = []
        self.inputs = []
        self.possible_inputs_to_vehicle = []


        self.states_from_sensor = [model.states_from_sensor for model in models]
        self.states_from_sensor = [model.states_from_sensor_at_infeasible for model in models]
        self.inputs_to_vehicle = [model.inputs_to_vehicle for model in [models[0]]]

        self.nvar = self.nu + self.nx
        self.models = models

        radii = []
        for i in range(len(models)):
            radii.append(1) #radii.append(getattr(params, "disc_" + str(i) + "_r"))

        self.sfm_policies = []
        for i in range(1, len(models)):
            sfm_params = [ 1, 1, 1, 1, 1, 1]#sfm_params = getattr(params, "pedestrians_sfm_param" + str(i))
            self.sfm_policies.append(SocialForcesPolicy(sfm_params=sfm_params, radii=radii, id=1))


    def continuous_model(self, x, u, param, settings):

        settings._params.load_params(param)


        pos = x[0:2]
        psi = x[2]
        v_x = x[3]
        v_y = x[4]
        w = x[5]

        a_x = u[0]
        a_y = u[1]
        alpha = u[2]
        slack = u[3]

        xdot_joint = ca.vertcat(
                v_x,
                      v_y,
                      w,
                      a_x,
                      a_y,
                      alpha)

        for id in range(1, len(self.models)):
            i = id -1
            vel = get_agent_states(x,id)[3:5]
            w = get_agent_states(x,id)[5]


            sfm_policy = self.sfm_policies[i]

            sfm_policy.update_params(getattr(settings._params, "pedestrians_sfm_param" + str(i)))
            sfm_action = sfm_policy.step(joint_state=x)

            xdot = ca.vertcat(vel[0], vel[1], w, sfm_action[0], sfm_action[1], 0)

            xdot_joint = ca.vertcat(xdot_joint, xdot)
        return xdot_joint
    def upper_bound(self):
        result = np.array([])

        for model in [self.models[0]]:
            for input in model.inputs:
                result = np.append(result, model.robot_config['input_constraints']['upper_bounds'][input])

        for model in self.models:
            for state in model.states:
                result = np.append(result, model.robot_config['state_constraints']['upper_bounds'][state])

        return result

    # Appends lower bounds from system
    def lower_bound(self):
        result = np.array([])

        for model in [self.models[0]]:
            for input in model.inputs:
                result = np.append(result, model.robot_config['input_constraints']['lower_bounds'][input])

        for model in self.models:
            for state in model.states:
                result = np.append(result, model.robot_config['state_constraints']['lower_bounds'][state])

        return result










