import numpy as np
import casadi
import os, sys

def load_forces_path():

    print_paths = ["PYTHONPATH"]
    # Is forces in the python path?
    try:
        import forcespro.nlp
        print('Forces found in PYTHONPATH')

        return
    except:
        pass

    paths = [os.path.join(os.path.expanduser("~"), "forces_pro_client"),
             os.path.join(os.getcwd(), "forces"),
             os.path.join(os.getcwd(), "../forces"),
             os.path.join(os.getcwd(), "forces/forces_pro_client"),
             os.path.join(os.getcwd(), "../forces/forces_pro_client")]
    for path in paths:
        if check_forces_path(path):
            return
        print_paths.append(path)

    print('Forces could not be imported, paths tried:\n')
    for path in print_paths:
        print('{}'.format(path))
    print("\n")


def check_forces_path(forces_path):
    # Otherwise is it in a folder forces_path?
    try:
        if os.path.exists(forces_path) and os.path.isdir(forces_path):
            sys.path.append(forces_path)
        else:
            raise IOError("Forces path not found")

        import forcespro.nlp
        print('Forces found in: {}'.format(forces_path))

        return True
    except:
        return False


class ParameterStructure:

    def __init__(self):
        self._parameters = dict()
        self._organization = dict() # Lists parameter grouping and indices
        self._npar = 0

    def add_parameter(self, name, n_par=1):
        self._organization[name] = n_par
        self._parameters[name] = list(range(self._npar, self._npar + n_par))
        #setattr(self, name+ "_index", self.param_idx)
        self._npar += n_par

    def add_multiple_parameters(self, name, amount):
        self.organization[self.param_idx] = amount
        for i in range(amount):
            self.parameters[self.param_idx] = name + "_" + str(i)
            setattr(self, name + "_" + str(i) + "_index", self.param_idx)
            self.param_idx += 1

    def has_parameter(self, name):
        return hasattr(self, name)

    def n_par(self): # Does not need + 1, because it is always increased
        return self._npar

    def save(self):
        return self.__dict__['_parameters']

    def __str__(self):
        result = "--- Parameter Structure ---\n"
        for name, amount in self._organization.items():
                result += "{}\t:\t{}\n".format(name, self._parameters[name])




        result += "--------------------\n"
        return result

    # When operating, retrieve the weights from param
    def load_params(self, params):
        for key, idxs in self._parameters.items(): # This is a parameter name
            setattr(self, key, params[self._parameters[key]]) # this is an index

class WeightStructure:

    # When defining the structure we define a structure with the weights as variables
    def __init__(self, parameters, weight_list):
        self.weight_list = weight_list

        for idx, weight in enumerate(weight_list):
            setattr(self, weight + "_index", parameters.param_idx)
            parameters.add_parameter(weight + "_weight")

        self.npar = len(weight_list)
        # parameters

    # When operating, retrieve the weights from param
    def set_weights(self, param):
        for weight in self.weight_list:
            # print(weight + ": " + str(getattr(self, weight+"_index")))
            setattr(self, weight , param[getattr(self, weight+"_index")])


def rotation_matrix(angle):
    return np.array([[casadi.cos(angle), -casadi.sin(angle)],
                      [casadi.sin(angle), casadi.cos(angle)]])

