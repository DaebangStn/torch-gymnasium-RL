from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium import logger
import math
import numpy as np


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


class CartPoleDistEnv1(CartPoleEnv):
    def __init__(self, render_mode=None, theta_threshold=12):
        super().__init__()
        self.render_mode = render_mode
        self.theta_threshold_radians = theta_threshold * 2 * math.pi / 360

    def step(self, action):
        sample = np.random.random_sample()
        if sample < 0.99:
            observation, reward, terminated, truncated, info = super().step(action)
        else:
            observation, reward, terminated, truncated, info = self.dist_step(action)

        return observation, reward, terminated, truncated, info

    def dist_step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        dist_theta = self.theta_threshold_radians * np.random.randn() * 0.1
        prev_theta = theta
        theta += dist_theta
        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, \
            {f"disturbance occured. theta {prev_theta}->{theta}"}
