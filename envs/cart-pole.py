from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium import logger
from gymnasium.error import DependencyNotInstalled
import gymnasium as gym
import math
import numpy as np
import pygame


class CartPoleRewardEnv1(CartPoleEnv):
    def __init__(self, render_mode=None, theta_threshold=12):
        super().__init__()
        self.render_mode = render_mode
        self.theta_threshold_radians = theta_threshold * 2 * math.pi / 360

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = centered_reward(observation, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info


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
        self.text_title = 'CartPoleDist-v1'
        self.text_body = None

    def render(self):
        return render_with_info(self, info={'head': self.text_title, 'body': self.text_body})

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
        self.text_body = f"disturbance! theta {prev_theta:.3f}->{theta:.3f}"

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


def render_with_info(self, info=None):
    if self.render_mode is None:
        assert self.spec is not None
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            "You can specify the render_mode at initialization, "
            f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
        )
        return

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as e:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gymnasium[classic-control]`"
        ) from e

    if self.screen is None:
        pygame.init()

        self.font32 = pygame.font.SysFont("arial", 32)
        self.font16 = pygame.font.SysFont("arial", 16)

        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        else:  # mode == "rgb_array"
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
    if self.clock is None:
        self.clock = pygame.time.Clock()

    world_width = self.x_threshold * 2
    scale = self.screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * self.length)
    cartwidth = 50.0
    cartheight = 30.0

    if self.state is None:
        return None

    x = self.state

    self.surf = pygame.Surface((self.screen_width, self.screen_height))
    self.surf.fill((255, 255, 255))

    l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    axleoffset = cartheight / 4.0
    cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
    carty = 100  # TOP OF CART
    cart_coords = [(l, b), (l, t), (r, t), (r, b)]
    cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
    gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

    l, r, t, b = (
        -polewidth / 2,
        polewidth / 2,
        polelen - polewidth / 2,
        -polewidth / 2,
    )

    pole_coords = []
    for coord in [(l, b), (l, t), (r, t), (r, b)]:
        coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
        coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        pole_coords.append(coord)
    gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
    gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

    gfxdraw.aacircle(
        self.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )
    gfxdraw.filled_circle(
        self.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )

    gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

    self.surf = pygame.transform.flip(self.surf, False, True)

    if info is not None:
        if info['head'] is not None:
            text_head = self.font32.render(info['head'], True, (0, 0, 0))
            self.surf.blit(text_head, (0, 0))

        if info['body'] is not None:
            text_body = self.font16.render(info['body'], True, (0, 0, 0))
            self.surf.blit(text_body, (0, 40))

    self.screen.blit(self.surf, (0, 0))
    if self.render_mode == "human":
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    elif self.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )


def centered_reward(observation, reward, terminated, truncated, info):
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    if terminated or truncated:
        reward = -1
    else:
        reward = reward - abs(cart_position) * 0.5 - abs(pole_angle) * 1.0

    return reward
