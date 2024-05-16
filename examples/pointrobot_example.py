from nav_simulator.nav2d_env import Nav2DEnv

env = Nav2DEnv()

# Sample usage
obs = env.reset()
env.render()

done = False
while not done:
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         done = True

    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    env.render()
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

env.close()