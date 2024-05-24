import casadi as ca
from harmony_mpcs.utils.utils import get_agent_states, perpendicular
class SocialForcesPolicy:
    def __init__(self, sfm_params, radii, id):


        self._ego_id = id

        # Parameters
        self._rel_time = 0.54
        self._A = 4.5  # relative importance of position vs velocity vector
        self._lambdaImportance = 2  # speed interaction
        self._gamma = 0.35  # speed interaction
        self._n = 2  # angular interaction
        self._n_prime = 3

        self._epsilon = 0.01
        self._eps = 0.05

        self._goal = sfm_params[:2]
        self._desired_speed = sfm_params[2]
        self._w_goal = sfm_params[3]
        self._w_social = sfm_params[4]
        self._w_obstacle = sfm_params[5]
        self._radii = radii

    def step(self, joint_state):


        goal_force = self.compute_goal_force(joint_state)
        social_force = self.compute_ped_repulsive_force(joint_state)

        summed_force = self._w_goal * goal_force + self._w_social * social_force
        action = ca.vertcat([summed_force, [0]])
        return action

    def compute_goal_force(self, joint_state):
        # Attractive forces towards goals

        pos = get_agent_states(joint_state, self._ego_id)[:2]
        vel = get_agent_states(joint_state, self._ego_id)[3:5]
        goal = self._goal

        rgi = goal - pos
        d = ca.norm_2(rgi)
        rgi_direction = rgi / (d + self._epsilon)

        if d < 0.5:
            self._desired_speed = 0.1

        force = (rgi_direction * self._desired_speed - vel) / self._rel_time
        return force

    def compute_ped_repulsive_force(self, joint_state):
        # computes the repulsive force from pedestrians
        force = 0
        ego_pos = get_agent_states(joint_state, self._ego_id)[:2]
        ego_vel = get_agent_states(joint_state, self._ego_id)[3:5]
        ego_radius = self._radii[self._ego_id]
        for i in range(len(self._radii)):
            if i != self._ego_id:
                other_pos = get_agent_states(joint_state, i)[:2]
                other_vel = get_agent_states(joint_state, i)[:2]
                other_radius = self._radii[i]

                rij = other_pos - ego_pos
                d = ca.norm_2(rij)
                rij_direction = rij /(d + self._epsilon)
                d_without_radius = d - ego_radius - other_radius

                vij = ego_vel - other_vel
                vd = ca.norm_2(vij)
                vij_direction = vij /(vd + self._epsilon)

                interaction = self._lambdaImportance * vij_direction + rij_direction
                interaction_length = ca.norm_2(interaction)
                interaction_direction = interaction / (interaction_length + self._epsilon)

                v = interaction
                w = rij

                cross_product = v[0] * w[1] - v[1] * w[0]

                dot_product = v[0] * w[0] + v[1] * w[1]

                theta = ca.atan2(cross_product, dot_product)

                B = self._gamma * interaction_length

                theta_ = theta + B * self._eps

                v_input = -d_without_radius / B - (self._n_prime * B * theta_) * (self._n_prime * B * theta_)
                a_input = -d_without_radius / B - (self._n * B * theta_) * (self._n * B * theta_)
                forceVelocityAmount = - self._A * ca.exp(v_input)
                forceAngleAmount = - self._A * ca.sign(theta_) * ca.exp(a_input)

                forceVelocity = forceVelocityAmount * (interaction_direction)
                forceAngle = forceAngleAmount * perpendicular(interaction_direction)

                force += forceVelocity + forceAngle

        return force

