import harmony_mpcs.mpc_solver_generation.inequality as inequality
import harmony_mpcs.mpc_solver_generation.objective as objective

class ModuleManager:
    """
    The idea of modules is that they can include multiple constraint sets if necessary
    In addition, they are directly linked to the c++ code module
    """

    def __init__(self, params):
        self.constraint_manager = inequality.Constraints(params)
        self.modules = []

        self.params = params

    def add_module(self, module):
        self.modules.append(module)

        if module.type == "constraint":
            for constraint in module.constraints:
                self.constraint_manager.add_constraint(constraint)

    def inequality_constraints(self, z, param, settings, model):
        ineq = self.constraint_manager.inequality(z, param, settings, model)
        return ineq

    def number_of_constraints(self):
        return self.constraint_manager.nh

    def __str__(self):
        result = "--- MPC Modules ---\n"
        for module in self.modules:
            result += str(module) + "\n"

        return result


class Module:

    def __init__(self):
        self.module_name = "UNDEFINED"

    def __str__(self):
        result = self.type.capitalize() + ": " + self.module_name
        return result


""" OBJECTIVE MODULES """
class GOMPCModule(Module):

    """
    Navigate to a subgoal/goal
    """

    def __init__(self, params, system, config):
        self.module_name = "NOTDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "NOTDEFINED"
        self.type = "objective"
        self.system = system

        self.objectives = []
        self.objectives.append(objective.GOMPCObjective(params, system, config))

class AdaptiveSimulationModule(Module):

    """
    Navigate to a subgoal/goal
    """

    def __init__(self, params, system, config):
        self.module_name = "NOTDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "NOTDEFINED"
        self.type = "objective"
        self.system = system

        self.objectives = []
        self.objectives.append(objective.AdaptiveSimulationObjective(params, system, config))

class FixedMPCModule(Module):

    """
    Navigate to a subgoal/goal
    """

    def __init__(self, params, N):
        self.module_name = "NOTDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "NOTDEFINED"
        self.type = "objective"
        #self.system = system

        self.objectives = []
        self.objectives.append(objective.FixedMPCObjective(params, N))

class SF_MPCModule(Module):

    """
    Navigate to a subgoal/goal
    """

    def __init__(self, params):
        self.module_name = "NOTDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "NOTDEFINED"
        self.type = "objective"

        self.objectives = []
        self.objectives.append(objective.SF_Objective(params))

""" CONSTRAINT MODULES """


class EllipsoidalConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles, num_constraints):
        self.module_name = "EllipsoidalConstraints"  # Needs to correspond to the c++ name of the module
        self.import_name = "modules_constraints/ellipsoidal_constraints.h"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.FixedEllipsoidConstraints(params,n_discs, max_obstacles, num_constraints))

class GaussianEllipsoidalConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles, num_constraints):
        self.module_name = "EllipsoidalConstraints"  # Needs to correspond to the c++ name of the module
        self.import_name = "modules_constraints/ellipsoidal_constraints.h"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.FixedGaussianEllipsoidConstraints(params,n_discs, max_obstacles, num_constraints))

class MultiEllipsoidalConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles):
        self.module_name = "MultiEllipsoidalConstraints"  # Needs to correspond to the c++ name of the module
        self.import_name = "modules_constraints/ellipsoidal_constraints.h"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.MultiEllipsoidConstraints(n_discs, max_obstacles, params))

class LinearConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, num_constraints, horizon_length):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.LinearConstraints(params,n_discs, num_constraints))


class InteractiveLinearConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles, num_constraints):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.InteractiveLinearConstraints(params,n_discs, max_obstacles, num_constraints))



class InteractiveEllipsoidConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.InteractiveEllipsoidConstraints(params,n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent))

class InteractiveGaussianEllipsoidConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.InteractiveGaussianEllipsoidConstraints(params,n_discs, max_obstacles, num_constraints, num_states_ego_agent, num_states_other_agent))

class StaticLinearConstraintModule:
    def __init__(self, params, n_discs, max_obstacles):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.StaticLinearConstraints(params,n_discs, max_obstacles))


class FixedLinearConstraintModule(Module):

    """
    Linear constraints for scenario-based motion planning
    """

    def __init__(self, params, n_discs, max_obstacles):
        self.module_name = "TOBEDEFINED"  # Needs to correspond to the c++ name of the module
        self.import_name = "TOBEDEFINED"
        self.type = "constraint"

        self.constraints = []
        self.constraints.append(inequality.FixedLinearConstraints(params,n_discs, max_obstacles))

