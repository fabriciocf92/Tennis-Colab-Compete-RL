import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as f
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

EPSILON_INITIAL = 1.0
EPSILON_DECAY = 0.998
EPSILON_FINAL = 0.001

POLICY_DELAY = 2


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agents, index, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.total_state_size = state_size * agents
        self.action_size = action_size
        self.total_action_size = action_size * agents
        self.seed = random.seed(random_seed)
        self.index = torch.tensor([index]).to(device)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.actor_target = Actor(state_size=state_size, action_size=action_size, seed=random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size=self.total_state_size, action_size=self.total_action_size,
                                   seed=random_seed).to(device)
        self.critic_target = Critic(state_size=self.total_state_size, action_size=self.total_action_size,
                                    seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        if np.random.random() < max(EPSILON_INITIAL * EPSILON_DECAY**i_episode, EPSILON_FINAL):
            return 2 * (np.random.rand(self.action_size) - 0.5)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).cpu().data.numpy().squeeze()  ##########
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            # action += self.noise.sample(decay_factor=i_episode)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, actions_pred, actions_next_pred, steps):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next_pred)  ########

        # Compute Q targets for current states (y_i)
        Q_targets = rewards.index_select(1, self.index) + \
                    (gamma * Q_targets_next * (1 - dones.index_select(1, self.index)))
        # Compute critic loss
        Q_expected = self.critic_local(states, torch.reshape(actions, (-1, self.total_action_size)))
        critic_loss = f.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if steps % POLICY_DELAY == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(actions_pred)]
            actions_pred = torch.stack(actions_pred)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            soft_update(self.critic_local, self.critic_target, TAU)
            soft_update(self.actor_local, self.actor_target, TAU)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, theta=0.2, sigma=0.15):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = theta * (self.mu - x) + sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
