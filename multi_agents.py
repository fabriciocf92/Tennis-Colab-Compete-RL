from ddpg_agent import Agent
import random
import torch
import numpy as np
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor


class MultiAgents:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agents = [Agent(state_size=state_size, action_size=action_size, agents=num_agents, index=agent_index,
                             random_seed=random_seed) for agent_index in range(num_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.steps = 0

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.steps += 1
        states = states.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for agent in self.agents:
                experiences = self.memory.sample()
                states, actions, rewards, next_states, dones = experiences
                actions_pred = self.actions_pred(states=states)
                actions_next_pred = self.actions_pred(states=next_states)
                agent.learn(experiences, GAMMA, actions_pred=actions_pred,
                            actions_next_pred=actions_next_pred, steps=self.steps)

    def actions_pred(self, states):
        actions = []
        for i, agent in enumerate(self.agents):
            state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent.index).squeeze(1)
            actions.append(agent.actor_local(state))
        return torch.cat(actions, dim=1)

    def act(self, state, i_episode, add_noise=True):
        actions = []
        for agent in self.agents:
            actions.append(agent.act(state[agent.index], add_noise=add_noise, i_episode=i_episode))
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
