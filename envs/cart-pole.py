from gymnasium.envs.classic_control import CartPoleEnv
import math


class CartPoleRewardEnv1(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = self.get_reward(observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def get_reward(self, observation, reward, terminated, truncated, info):
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        if terminated or truncated:
            reward = -1
        else:
            reward = reward - abs(cart_position) * 0.5 - abs(pole_angle) * 1.0

        return reward


class CartPoleObsEnv1(CartPoleEnv):
    def __init__(self, render_mode=None, theta_threshold=12):
        super().__init__()
        self.render_mode = render_mode
        self.theta_threshold_radians = theta_threshold * 2 * math.pi / 360

