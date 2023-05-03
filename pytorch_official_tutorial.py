import gymnasium as gym

env = gym.make('CartPole-v1')

# Access the time limit of the environment
time_limit = env._max_episode_steps

print(f"Time limit of the CartPole-v1 environment: {time_limit}")
