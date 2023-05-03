from collections import deque, namedtuple
from typing import NamedTuple
import random
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class HyperParams(NamedTuple):
    lr: float = 0.001
    epsilon_begin: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    num_episodes: int = 100
    batch_size: int = 64
    tau: float = 0.005
    target_update: int = 5
    render: bool = False


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NetworkLayer3(nn.Module):
    def __init__(self, num_input, num_output):
        super(NetworkLayer3, self).__init__()
        self.layer1 = nn.Linear(num_input, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_output)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
