import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import agents


class DQN(nn.Module):
    def __init__(self, env, device, logger):
        super(DQN, self).__init__()

        self.env = env
        self.logger = logger

        logger.info(f'using environments: {env.unwrapped.spec.id}')

        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        logger.info(f'env obs dim : {self.num_observations},'
                    f' act dim : {self.num_actions}')

        # saved parameters when you call save()
        self.policy = agents.utils.NetworkLayer3(self.num_observations, self.num_actions)
        self.memory = agents.utils.ReplayMemory(10000)
        self.epsilon = None

        self.target = agents.utils.NetworkLayer3(self.num_observations, self.num_actions)
        self.target.load_state_dict(self.policy.state_dict())

        self.device = device
        self.counter = 0

        self.hyper_params = None
        self.optimizer = None

    def train_with_episodes(self, hyper_param, early_stop=True):
        self.hyper_params = hyper_param
        self.epsilon = hyper_param.epsilon_begin

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.hyper_params.lr, amsgrad=True)
        rewards_list_for_episodes = []

        for episode in range(self.hyper_params.num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            reward_for_episode = 0
            reward_trend_list_for_episode = []

            for t in range(self.hyper_params.max_steps):
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                reward_for_episode += reward

                reward = torch.tensor([reward], device=self.device, dtype=torch.float32).unsqueeze(0)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                reward_trend_list_for_episode.append(reward_for_episode)

                if terminated:
                    self.logger.warning(f'episode: {episode}, terminated in step: {t}, reward: {reward_for_episode}')
                    break

                if truncated:
                    self.logger.warning(f'episode: {episode}, truncated in step: {t}, reward: {reward_for_episode}')
                    break

                if np.mean(reward_trend_list_for_episode[-10:]) > 200 and early_stop:
                    self.logger.warning(f'episode: {episode}, early stopped in step: {t}, reward: {reward_for_episode}')
                    break

                state = next_state
                self.update_counter()
                self.learn_and_update_target_by_replay()

                # TODO: why lagged update does not work?
                target_net_state_dict = self.target.state_dict()
                policy_net_state_dict = self.policy.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = self.hyper_params.tau * policy_net_state_dict[key] + \
                                                 (1 - self.hyper_params.tau) * target_net_state_dict[key]

                self.target.load_state_dict(target_net_state_dict)
                self.logger.debug(f'episode: {episode}, step: {t} policy network updated')

            rewards_list_for_episodes.append(reward_for_episode)
            self.epsilon = max(self.epsilon * self.hyper_params.epsilon_decay, self.hyper_params.epsilon_end)

            if np.mean(rewards_list_for_episodes[-50:]) > 200 and early_stop:
                self.logger.warning(f'DQN model training early completed in episode: {episode}')
                break

        self.logger.warning(
            f'DQN model training completed and mean last 50 reward: {np.mean(rewards_list_for_episodes[-50:])}')

    def get_action(self, state):
        if self.epsilon is None:
            self.logger.warning('Epsilon is empty. returning random action')
            return np.random.choice(self.env.action_space, 1)

        if np.random.rand() < self.epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

        with torch.no_grad():
            return self.policy(state).max(1)[1].view(1, 1)

    def update_counter(self):
        if self.hyper_params is None:
            self.logger.error('Hyper Parameters are empty. call optim_model() first')
            return

        self.counter = (self.counter + 1) % self.hyper_params.target_update

    def learn_and_update_target_by_replay(self):
        if self.hyper_params is None:
            self.logger.error('Hyper Parameters are empty. call optim_model() first')
            return

        if len(self.memory) < self.hyper_params.batch_size:
            self.logger.debug('not enough memory, accumulating steps')
            return

        self.logger.debug('updating target by replay')

        transitions = self.memory.sample(self.hyper_params.batch_size)
        batch = agents.utils.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.hyper_params.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.hyper_params.gamma) + reward_batch.squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

    # Saving the models
    # Path: /base_dir/{YY-mm-dd-HH-mm-ss-env_id}
    def save(self, base_dir):
        if not os.path.exists(base_dir):
            self.logger.warning(f'{base_dir} does not exist. creating directory')
            os.makedirs(base_dir)

        filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-' + self.env.unwrapped.spec.id + '.pth'

        self.logger.info(f'saving model to {base_dir} with filename: {filename}')

        torch.save({
            'policy_net_dict': self.policy.state_dict(),
            'epsilon': self.epsilon,
            'replay_memory': self.memory,
        }, base_dir + '/' + filename)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_net_dict'])
        self.target.load_state_dict(checkpoint['policy_net_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint['replay_memory']
