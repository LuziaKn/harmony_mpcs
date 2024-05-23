import numpy as np

def perpendicular(a):
    # returns the vector perpendicular to a (left)
    b = np.zeros_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def forward_sim_sfm(xinit, goal, dyn_obsts, N, dt):
    
    n_agents = int(dyn_obsts.shape[0] + 1)

    # get position matrix
    positions_others = np.array([dyn_obst[:2] for dyn_obst in dyn_obsts])
    velocities_others = np.array([dyn_obst[3:5] for dyn_obst in dyn_obsts])
    goal_others = positions_others.copy()
    
    # constant vel prediction for others
    for k in range(int(2*N)):
        goal_others += velocities_others * dt
    
    # print('shape positions others', positions_others.shape)
    # print('shape xitin', np.expand_dims(xinit[:2],0).shape)
    
    positions = np.concatenate([np.expand_dims(xinit[:2],0), positions_others.reshape(-1, 2)], axis=0)
    velocities = np.concatenate([np.expand_dims(xinit[3:5],0), velocities_others.reshape(-1, 2)], axis=0)

    goals = np.concatenate([np.expand_dims(np.array(goal),0), goal_others], axis=0)


    # initial
    sfm_predictions = np.zeros((N, n_agents, 4))
    for k in range(N):
        # get forces
        forces = social_forces(positions, velocities, goals)
        # update velocities
        velocities += forces * dt
        # update positions
        velocities = np.clip(velocities, -0.5, 0.5) # todo get from config
        positions += velocities * dt
        # store predictions
        sfm_predictions[k] = np.concatenate([positions, velocities], axis=1)
    
    return sfm_predictions

def social_forces(positions, velocities, goals):
    # Parameters
    A = 4.5 # relative importance of position vs velocity vector
    lambdaImportance = 2 # speed interaction
    gamma = 0.35 # speed interaction
    n = 2 # angular interaction
    n_prime = 3

    # Initialize forces
    n_agents = positions.shape[0]
    forces = np.zeros((n_agents , 2))

    epsilon = 0.01
    
    social_factor = 1.5

    # Repulsive forces between agents
    for i in range(n_agents):
        ego_pos = positions[i]
        ego_vel = velocities[i]
        for j in range(n_agents):
            other_pos = positions[j]
            other_vel = velocities[j]
            if i != j:
                rij = other_pos - ego_pos
                d = np.linalg.norm(rij)
                rij_direction = rij /( d + epsilon)

                vij = ego_vel - other_pos
                vd = np.linalg.norm(vij)
                vij_direction = vij /( vd + epsilon)

                interaction = lambdaImportance * vij_direction + rij_direction
                interaction_length = np.linalg.norm(interaction)
                interaction_direction = interaction / (interaction_length + epsilon)

                v = interaction
                w =  rij

                cross_product = v[0] * w[1] - v[1] * w[0]

                dot_product = v[0] * w[0] + v[1] * w[1]

                theta = np.arctan2(cross_product, dot_product)

                B = gamma * interaction_length
                eps = 0.2
                theta_ = theta + B * eps

                v_input = -d / B - (n_prime * B * theta_) * (n_prime * B * theta_)
                a_input = -d / B - (n * B * theta_) * (n * B * theta_)
                forceVelocityAmount = - A * np.exp(v_input)
                forceAngleAmount = - A * np.sign(theta_) * np.exp(a_input)

                forceVelocity = forceVelocityAmount * (interaction_direction)
                forceAngle = forceAngleAmount * perpendicular(interaction_direction)

                force = forceVelocity + forceAngle

                forces[i] += social_factor * force


    # Attractive forces towards goals
    rel_time = 0.54
    for i in range(n_agents):
        vel = velocities[i]
        goal = goals[i]
        rgi = goal - positions[i]
        d = np.linalg.norm(rgi)
        rgi_direction = rgi / (d + epsilon)

        force = (rgi_direction * 1 - vel) / rel_time
        forces[i] += force



    return forces