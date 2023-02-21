from collections import namedtuple, deque
import random
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.transition = namedtuple("Transition", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state"])
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, *args):
        """Store a transition in the experience replay memory

        Args:
            state : Current state
            action : Action taken on the state
            reward : Reward received for the action in the current state
            next_state : New state obtained by performing the action in the current state
        """
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        """Store a transition in the experience replay memory

        Args:
            batch_size : Number of experiences to be sampled

        Returns:
            tuple : A tuple of 2-D torch.Tensor() containing state, action, reward and next_state
        """
        if self.__len__() < batch_size:
            return ()

        experiences = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)