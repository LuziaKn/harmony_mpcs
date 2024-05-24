import numpy as np

def perpendicular(a):
    # returns the vector perpendicular to a (left)
    b = np.zeros_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def forward_sim_sfm(agents, N, dt):

    # get position matrix
    positions = np.array([agent.pos_global_frame for agent in agents])
    velocities = np.array([agent.vel_global_frame for agent in agents])
    goals = np.array([agent.goal_global_frame for agent in agents])

    # initial
    sfm_predictions = np.zeros((N, len(agents), 4))
    for k in range(N):
        # get forces
        forces = social_forces(positions, velocities, goals)
        # update velocities
        velocities += forces * dt
        # update positions
        positions += velocities * dt
        # store predictions
        sfm_predictions[k] = np.concatenate([positions, velocities], axis=1)

def social_forces(positions, velocities, goals):
    # Parameters
    A = 4.5 # relative importance of position vs velocity vector
    lambdaImportance = 2 # speed interaction
    gamma = 0.35 # speed interaction
    n = 2 # angular interaction
    n_prime = 3

    # Initialize forces
    n_agents = positions.shape[0]
    forces = np.zeros_like((n_agents , 2))



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
                rij_direction = rij / d

                vij = ego_vel - other_pos
                vd = np.linalg.norm(vij)
                vij_direction = vij / vd

                interaction = lambdaImportance * vij_direction + rij_direction
                interaction_length = np.linalg.norm(interaction)
                interaction_direction = interaction / interaction_length

                v = interaction
                w =  rij

                cross_product = v[0] * w[1] - v[1] * w[0]

                dot_product = v[0] * w[0] + v[1] * w[1]

                theta = np.arctan2(cross_product, dot_product)

                B = gamma * interaction_length
                eps = 0.05
                theta_ = theta + B * eps

                v_input = -d / B - (n_prime * B * theta_) * (n_prime * B * theta_)
                a_input = -d / B - (n * B * theta_) * (n * B * theta_)
                forceVelocityAmount = - A * np.exp(v_input)
                forceAngleAmount = - A * np.sign(theta_) * np.exp(a_input)

                forceVelocity = forceVelocityAmount * (interaction_direction)
                forceAngle = forceAngleAmount * perpendicular(interaction_direction)

                force = forceVelocity + forceAngle


                forces[i] += force


    # Attractive forces towards goals
    rel_time = 0.54
    for i in range(n_agents):
        vel = velocities[i]
        goal = goals[i]
        rgi = goal - positions[i]
        d = np.linalg.norm(rgi)
        rgi_direction = rgi / d

        force = (rgi_direction * 1 - vel) / rel_time

        forces[i] += force



    return forces