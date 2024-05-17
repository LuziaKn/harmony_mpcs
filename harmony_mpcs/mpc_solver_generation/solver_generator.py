import sys, os, shutil

sys.path.append("../")
sys.path.append("")

sys.path.append('/home/luzia/code/forces_pro_client/') #add path to forces_pro files here ToDo make generic
import harmony_mpcs.mpc_solver_generation.helpers as helpers

# If your forces is in this directory add it
helpers.load_forces_path()

import numpy as np
import pickle
import forcespro.nlp
import harmony_mpcs.mpc_solver_generation.dynamics as dynamics
import harmony_mpcs.mpc_solver_generation.objective as objective


class SolverGenerator(object):

    def __init__(self, solver_settings, current_dir, save_dir):
        self.solver_settings = solver_settings
        self._current_dir = current_dir
        self._solver_dir = current_dir + save_dir

    def define_problem(self):

        print("--- Starting Model creation ---")

        '''
            Important: Problem formulation
            - Forces considers the first input/state (k = 1 in docs, k = 0 in python) to be the INITIAL state, which
            should not be constrained. I.e., all inequalities are shifted one step (k = 1, ..., k = N in python)
            
            - Because we want to optimize the final stage, while forces optimizes until N-1 in docs (N-2 in Python), 
            we need to add an additional stage at the end for x_N in python (i.e., N+1 in docs). Note that this stage does 
            not do anything, it just exist so we can optimize the rest 
            
            -> For these two reasons, define N_bar (forces horizon) versus N our horizon:
            N_bar = N + 2 (i.e., one stage added for the initial stage and one for the final optimized stage)
        '''
        self._N_bar = self.solver_settings._N + 2

        # Systems to control

        model = dynamics.PointMass_2order_Model(robot_config=self.solver_settings._robot_config)
        #model.interfaces = settings.interfaces

        print(model)
        print(self.solver_settings._modules)

        # Load model parameters from the settings
        solver = forcespro.nlp.SymbolicModel(self.solver_settings._N_bar)
        solver.N = self.solver_settings._N_bar      # prediction/planning horizon
        solver.nvar = model.nvar   # number of online variables
        solver.neq = model.nx     # number of equality constraints
        solver.npar = self.solver_settings._npar  # number of parameters

        solver.lb = model.lower_bound()
        solver.ub = model.upper_bound()

        print(model.lower_bound())
        # Functions used in the optimization
        # Note that we use solver.N = N_bar here!
        for i in range(0, solver.N):
            # Although optimizing the first stage does not contribute to anything, it is fine.
            # We also do not really have to optimize the final input, but it only effects the runtimes
            if i < self.solver_settings._N_bar:  # should be ignored already, but to be sure, we can ignore the objective in the final stage

                # Python cannot handle lambdas without an additional function that manually creates the lambda with the correct value
                def objective_with_stage_index(stage_idx):
                    return lambda z, p: objective.objective(z, p, model, self.solver_settings, stage_idx)

                solver.objective[i] = objective_with_stage_index(i)

            # For all stages after the initial stage (k = 0) and ignoring the final stage (k = N_bar-1), we specify inequalities
            if (i > 0) and (i < solver.N):


                solver.ineq[i] = lambda z, p: self.solver_settings._modules.inequality_constraints(z=z,
                                                                        param=p,
                                                                        settings=self.solver_settings,
                                                                        model=model)

                solver.nh[i] = self.solver_settings._nh

                solver.hu[i] = self.solver_settings._modules.constraint_manager.upper_bound
                solver.hl[i] = self.solver_settings._modules.constraint_manager.lower_bound
            else:
                solver.nh[i] = 0  # No constraints here
  
        # Equalities are specified on all stages
        solver.continuous_dynamics = lambda x, u, p: model.continuous_model(x, u, p, self.solver_settings)
        solver.E = np.concatenate([np.zeros((model.nx, model.nu)), np.eye(model.nx)], axis=1)

        # Initial stage (k = 0) specifies the states
        solver.xinitidx = range(model.nu, model.nvar)
        self._solver = solver
        self._model = model

        # Set the solver options
        self.set_solver_options()



    def generate_solver(self):
        print("--- Generating solver -!!!!!!!!!!!!G--")

        self.define_problem()
        print(self.solver_settings._params)

        # Remove the previous solver
        name = self.solver_settings._model_name + 'FORCESNLPsolver_fixed/'
        solver_path = self._current_dir + "/" + self.solver_settings._model_name + 'FORCESNLPsolver_fixed/'
        new_solver_path = self._solver_dir + "/" 
        print(new_solver_path)

        #print("Path of the new solver: {}".format(new_solver_path))
        if os.path.exists(new_solver_path) and os.path.isdir(new_solver_path):
            shutil.rmtree(new_solver_path)

        if not os.path.exists(new_solver_path):
            os.makedirs(new_solver_path)

        # Creates code for symbolic model formulation given above, then contacts server to generate new solver
        #output1 = ("sol", [], [])
        generated_solver = self._solver.generate_solver(self._options) #, outputs


        # Move the solver up a directory for convenience
        if os.path.isdir(solver_path):
            shutil.move(solver_path, new_solver_path)

        # save settings
        solver_property_dict = {}
        properties = {"nx": self._solver.neq, "nu": self._model.nu, "npar": self._solver.npar, "N":self._solver.N,
                      "n_static_obstacles": self.solver_settings._n_dyn_obst, "n_discs": self.solver_settings._n_discs,
                      "n_static_constraints": self.solver_settings._n_static_obst}

        solver_property_dict['map_runtime_par'] = self.solver_settings._params.save()
        solver_property_dict['properties'] = properties
        solver_property_dict['modules'] = [constraint.name for constraint in
                                           self.solver_settings._modules.constraint_manager.constraints]


        file_name = new_solver_path + name + "/params.pkl"
        with open(file_name, 'wb') as outp:
            pickle.dump(solver_property_dict, outp, pickle.HIGHEST_PROTOCOL)

    def set_solver_options(self):
        # Set solver options
        options = forcespro.CodeOptions(self.solver_settings._model_name + 'FORCESNLPsolver_fixed')
        options.printlevel = 0  # 1 = timings, 2 = print progress
        options.optlevel = 0  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        options.timing = 1
        options.overwrite = 1
        options.cleanup = 0
        options.nlp.integrator.Ts = self.solver_settings._dt
        options.nlp.integrator.nodes = 5
        options.solver_timeout = 1
        options.noVariableElimination = 1
        # FOR FLOATING LICENSES
        if self.solver_settings._floating:
            options.license.use_floating_license = 1
            options.embedded_timing = 1
        #options.init = 1 # Warm start?

        # Todo: Make solver options a lambda defined in the settings
        if not self.solver_settings._use_sqp_solver:
            # -- PRIMAL DUAL INTERIOR POINT (Default Solver!) -- #
            options.nlp.integrator.type = 'ERK4'
            options.maxit = 500  # Maximum number of iterations
            options.mu0 = 10  # IMPORTANT: CANNOT BE 20 FOR SQP!
            #options.init = 1
            #options.solvemethod = "PDIP_NLP"
            #options.nlp.linear_solver = 'symm_indefinite'
        else:
            # -- SQP (Useful under strict computation time limits) -- #
            # https://forces.embotech.com/Documentation/high_level_interface/index.html#sequential-quadratic-programming-algorithm
            # https://forces.embotech.com/Documentation/examples/robot_arm_sqp/index.html#sec-robot-arm-sqp
            options.solvemethod = "SQP_NLP"

            # Number of QPs to solve, default is 1. More iterations is higher optimality but longer computation times
            options.sqp_nlp.maxqps = 1
            options.maxit = 500  # This should limit the QP iterations, but if it does I get -8: QP cannot proceed


            options.sqp_nlp.qpinit = 0 #1 # 1 = centered start, 0 = cold start

            # Should be a faster robust linear system solver (default in the future they say)
            options.nlp.linear_solver = 'symm_indefinite_fast'

            # Increasing helps when exit code is -8
            options.sqp_nlp.reg_hessian = 5e-9  # = default
            options.exportBFGS = 1 # Todo: Make this "2" later for lower triangular instead of diagonal
            options.nlp.parametricBFGSinit = 1  # Allows us to initialize the estimate at run time with the exported one

            # Speeding up the solver
            # Makes a huge difference (obtained from the exported BFGS)
            #options.nlp.bfgs_init = np.diag(np.array([0.000408799, 0.398504, 1.22784, 0.697318, 1.27293, 0.0718911, 0.0718882, 0.980375, 0.19122]))
            #settings.bfgs_init = options.nlp.bfgs_init

            # Disables checks for NaN and Inf (only use this if your optimization is working)
            options.nlp.checkFunctions = 0
        self._options = options