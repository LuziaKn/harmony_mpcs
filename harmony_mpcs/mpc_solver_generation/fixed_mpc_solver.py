"""
TEMPLATE
Main system specific file to create an MPC optimization solver using Forces Pro
"""

# Add forces to the path here.
# The forces client files should be under "python_forces_code/forces"!
# Or needs to be in the path already
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
import copy

import harmony_mpcs.mpc_solver_generation.dynamics as dynamics
import harmony_mpcs.mpc_solver_generation.objective as objective

#import generate_cpp_files

import harmony_mpcs.mpc_solver_generation.fixed_mpc_settings as settings

# Press the green button in the gutter to run the script.
def generate_solver(current_dir, save_dir):

    # Set to False to only generate C++ code
    generate_solver = True

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
    settings.N_bar = settings.N + 2


    # Systems to control

    model = dynamics.PointMass_2order_Model(robot_config=settings.robot_config)
    model.interfaces = settings.interfaces

    print(model)
    print(settings.modules)

    # Load model parameters from the settings
    solver = forcespro.nlp.SymbolicModel(settings.N_bar)
    solver.N = settings.N_bar       # prediction/planning horizon
    solver.nvar = model.nvar   # number of online variables
    solver.neq = model.nx     # number of equality constraints
    solver.npar = settings.npar

    # Bounds
    solver.lb = model.lower_bound()
    solver.ub = model.upper_bound()



    # Functions used in the optimization
    # Note that we use solver.N = N_bar here!
    for i in range(0, solver.N):
        # Although optimizing the first stage does not contribute to anything, it is fine.
        # We also do not really have to optimize the final input, but it only effects the runtimes
        if i < settings.N_bar:  # should be ignored already, but to be sure, we can ignore the objective in the final stage

            # Python cannot handle lambdas without an additional function that manually creates the lambda with the correct value
            def objective_with_stage_index(stage_idx):
                return lambda z, p: objective.objective(z, p, model, settings, stage_idx)

            solver.objective[i] = objective_with_stage_index(i)

        # For all stages after the initial stage (k = 0) and ignoring the final stage (k = N_bar-1), we specify inequalities
        if (i > 0) and (i < solver.N):


            solver.ineq[i] = lambda z, p: settings.modules.inequality_constraints(z=z,
                                                                     param=p,
                                                                     model=model,
                                                                     settings=settings)

            solver.nh[i] = settings.nh

            solver.hu[i] = settings.modules.constraint_manager.upper_bound
            solver.hl[i] = settings.modules.constraint_manager.lower_bound
        else:
            solver.nh[i] = 0  # No constraints here

    # Equalities are specified on all stages
    solver.continuous_dynamics = lambda x, u, p: model.continuous_model(x, u, p, settings)
    solver.E = np.concatenate([np.zeros((model.nx, model.nu)), np.eye(model.nx)], axis=1)

    # Initial stage (k = 0) specifies the states
    solver.xinitidx = range(model.nu, model.nvar)

    # Set solver options
    options = forcespro.CodeOptions(settings.robot_config.name + 'FORCESNLPsolver_fixed')
    options.printlevel = 0  # 1 = timings, 2 = print progress
    options.optlevel = 0  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
    options.timing = 1
    options.overwrite = 1
    options.cleanup = 0
    options.nlp.integrator.Ts = settings.integrator_stepsize
    options.nlp.integrator.nodes = 5
    options.solver_timeout = 1
    options.noVariableElimination = 1
    #options.init = 1 # Warm start?

    # Todo: Make solver options a lambda defined in the settings
    if not settings.use_sqp_solver:
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



    if generate_solver:
        print("--- Generating solver ---")

        # Remove the previous solver
        solver_path = current_dir + "/" + settings.robot_config.name + 'FORCESNLPsolver_fixed/'
        new_solver_path = save_dir + "/" + settings.robot_config.name + 'FORCESNLPsolver_fixed'
        print(new_solver_path)

        #print("Path of the new solver: {}".format(new_solver_path))
        if os.path.exists(new_solver_path) and os.path.isdir(new_solver_path):
            shutil.rmtree(new_solver_path)

        if not os.path.exists(new_solver_path):
            os.makedirs(new_solver_path)

        # Creates code for symbolic model formulation given above, then contacts server to generate new solver
        #output1 = ("sol", [], [])
        generated_solver = solver.generate_solver(options) #, outputs


        # Move the solver up a directory for convenience
        if os.path.isdir(solver_path):
            shutil.move(solver_path, new_solver_path)

        # save settings
        solver_property_dict = {}
        properties = {"nx": solver.neq, "nu": model.nu, "npar": solver.npar, "N":solver.N,
                      "n_static_obstacles": settings.n_other_agents, "n_discs": settings.n_discs,
                      "n_static_constraints": settings.n_static_constraints}

        solver_property_dict['map_runtime_par'] = settings.params.save()
        solver_property_dict['properties'] = properties
        solver_property_dict['modules'] = [constraint.name for constraint in
                                           settings.modules.constraint_manager.constraints]


        file_name = new_solver_path + "/params.pkl"
        with open(file_name, 'wb') as outp:
            pickle.dump(solver_property_dict, outp, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.realpath(__file__)) + "/../solvers"
    generate_solver(save_dir=file_dir)
