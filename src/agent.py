"""Classes for the REINFORCE agent training.

This module contains classes and functions for implementing a reinforcement learning
agent. It includes a policy network for discrete action spaces, a padding function for
handling terminal and truncated states, and a memory class for storing and managing
experiences.

Classes
-------
    PolicyNetworkDiscrete(nn.Module)
        A simple fully-connected policy network for discrete action spaces.
    Padding
        A class used to apply padding to terminal and truncated states.
    Memory
        A class to store and manage the memory for reinforcement learning agents.
"""

from collections import deque

import gymnasium as gym
import numpy as np
import torch
import tqdm
from torch import nn


class PolicyNetworkDiscrete(nn.Module):
    """A simple fully-connected policy network for discrete action spaces.

    This network takes an observation as input and outputs a probability distribution
    over discrete actions. During training, it samples actions from this distribution
    and returns the log probabilities of the actions. During evaluation, it returns
    the action with the highest probability.
        Dimension of the flattened observation space.

    Methods
    -------
    forward(x)
        Perform a forward pass through the network. If in training mode, sample actions
        and return them along with their log probabilities. If in evaluation mode,
        return the action with the highest probability.

    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize a simple fully-connected network.

        Parameters
        ----------
        input_dim : int
            Dimension of the flatten observation space.
        hidden_dim : int
            Number of neurons in the hidden layer.
        output_dim : int
            Number of discrete actions.

        """
        super().__init__()
        self.perceptron = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """Perform a forward pass through the policy network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.

        Returns
        -------
        tuple[torch.Tensor] | torch.Tensor
            If in training mode, returns a tuple containing:
                - actions: Sampled actions from the categorical distribution.
                - log_probs: Log probabilities of the sampled actions.
            If not in training mode, returns:
                - actions: The actions with the highest probability.

        """
        probs = self.perceptron(x)
        if self.training:
            m = torch.distributions.Categorical(probs)
            actions = m.sample()
            log_probs = m.log_prob(actions)
            return actions, log_probs
        else:
            return torch.argmax(probs, dim=1)


class PolicyNetworkContinuous(nn.Module):
    """A simple fully-connected policy network for continious action spaces.

    Methods
    -------
    forward(x)
        Perform a forward pass through the network.

    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, eps: float = 1e-6
    ):
        """Initialize a simple fully-connected network.

        Parameters
        ----------
        input_dim : int
            Dimension of the flatten observation space.
        hidden_dim : int
            Number of neurons in the hidden layer.
        output_dim : int
            Number of discrete actions.
        eps : float, optional
            A small number for policy stability (1e-6 by default).

        """
        super().__init__()
        self.perceptron = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim),
        )
        # self.log_std = nn.Parameter(torch.zeros(output_dim))
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.std_layer = nn.Linear(hidden_dim, output_dim)

        self._eps = eps

    def forward(self, x):
        """Perform a forward pass through the policy network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.

        Returns
        -------
        tuple[torch.Tensor] | torch.Tensor
            If in training mode, returns a tuple containing:
                - actions: Sampled actions from the categorical distribution.
                - log_probs: Log probabilities of the sampled actions.

        """
        features = self.perceptron(x)
        action_means = self.mean_layer(features)
        if self.training:
            action_stds = torch.log(1 + torch.exp(self.std_layer(features)))
            m = torch.distributions.Normal(
                action_means + self._eps, action_stds + self._eps
            )
            actions = m.sample()
            log_probs = m.log_prob(actions)
            return actions, log_probs
        else:
            return action_means


class Masking:
    """A class used to apply mask to terminal and truncated states.

    Parameters
    ----------
    size : int
        The size of the masking array.

    Attributes
    ----------
    _size : int
        The number of environments to apply the mask.
    _is_terminated : np.ndarray
        An array indicating whether each state has terminated.

    Methods
    -------
    reset():
        Resets the mask function.
    __call__(terminated, truncated):
        Yields the mask values and the total termination flag.

    """

    def __init__(self):
        """Initialize padding function."""
        self.reset()

    def reset(self):
        """Reset the padding function."""
        self._is_terminated = None

    def __call__(
        self, terminated: np.ndarray, truncated: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Yield the padding values.

        Parameters
        ----------
        terminated : np.ndarray
            The array of the terminal states
        truncated : np.ndarray
            The array of the truncated states

        Returns
        -------
        np.ndarray
            The padding values (1 if value should be used, 0 otherwise)
        bool
            The total termination flag. True If all the states have reached the
            terminal states and the training loop should be stopped.

        """
        # Init internal state at first step
        if self._is_terminated is None:
            self._is_terminated = np.full_like(terminated, False)
        # As the function has a lag, we need firstly yield the previous values
        output = np.logical_not(self._is_terminated).astype(int)
        # Then we update the values
        done = np.logical_or(terminated, truncated)
        self._is_terminated = np.logical_or(self._is_terminated, done)
        # As termination flag checked after the step, we can yield it, using the updated
        # values
        total_termination = np.all(self._is_terminated)
        return output, total_termination


class Memory:
    """A class to store and manage the memory for reinforcement learning agents.

    Parameters
    ----------
    discount_factor : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch for padding.
    device : torch.device
        The device (CPU or GPU) to perform computations on.

    Methods
    -------
    clear():
        Clears the memory.
    __len__():
        Returns the length of the rewards list.
    append(log_prob, reward, terminated, truncated) -> bool:
        Appends a new experience to the memory.
    loss() -> tuple:
        Computes the loss, mean reward, and mean length of episodes.
    _rescale(returns: np.ndarray, padding: np.ndarray) -> np.ndarray:
        Rescales the returns using mean and standard deviation of valid rewards.
    _get_mean_reward(reward: np.ndarray, padding: np.ndarray) -> float:
        Computes the mean reward across the batch.
    _get_mean_length(padding: np.ndarray) -> float:
        Computes the mean length of episodes across the batch.

    """

    def __init__(self, discount_factor: float):
        """Initialize the agent with the given parameters.

        Parameters
        ----------
        discount_factor : float
            The discount factor for future rewards.
        batch_size : int
            The size of the batch for training.
        device : torch.device
            The device (CPU or GPU) to be used for computations.

        """
        self._discount_factor = discount_factor
        self._padding_function = Masking()

        self.clear()

    def clear(self):
        """Clean the buffer."""
        self._padding_function.reset()
        self._log_probs = []
        self._rewards = []
        self._mask = []
        self._device = None

    def __len__(self):
        """Get length of buffer."""
        return len(self._rewards)

    def append(
        self,
        log_prob: torch.Tensor,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> bool:
        """Append the state results to the agent's memory.

        Parameters
        ----------
        log_prob : torch.Tensor
            The log probability of the action taken.
        reward : np.ndarray
            The reward received after taking the action.
        terminated : np.ndarray
            Whether the episode has terminated.
        truncated : np.ndarray
            Whether the episode has been truncated.

        Returns
        -------
        bool
            The termination status after padding.

        """
        if self._device is None:
            self._device = log_prob.device
        elif self._device != log_prob.device:
            msg = (
                f"Input tensor's device '{log_prob.device}' does not match the "
                f"memorized one '{self._device}'"
            )
            raise ValueError(msg)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=1)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        padding, termination = self._padding_function(terminated, truncated)
        self._mask.append(padding)

        return termination

    def loss(self) -> tuple:
        """Compute the final loss.

        Returns
        -------
        loss : torch.Tensor
            A loss for back propogation.

        """
        if self._device is None:
            msg = (
                ".loss() method can not be called before putting at least one value"
                " to the memory with .append() method!"
            )
            raise Exception(msg)
        returns = deque()
        # Calculate the cumulative rewards for all runs in the batch
        step_return = np.zeros_like(self._rewards[0])
        for reward, mask in zip(self._rewards, self._mask):
            step_return = reward * mask + self._discount_factor * step_return
            returns.appendleft(step_return)
        # Pack rewards to the matrix and rescale them, using the mean and std values of
        # the valid values.
        returns = np.stack(returns, axis=1)
        returns = self._rescale(returns, self._mask)
        returns = torch.tensor(returns, dtype=torch.float).to(self._device)
        # Calculate mean reward and length of all episodes in the batch
        self._rewards = np.stack(self._rewards, axis=1)
        self._mask = np.stack(self._mask, axis=1)
        mean_reward = self._get_mean_reward(self._rewards, self._mask)
        mean_length = self._get_mean_length(self._mask)
        # Compute the reward across timeline (sum over first dimension)
        self._log_probs = torch.stack(self._log_probs, dim=1)
        # Get approximation by averaging the batch
        loss = torch.sum(-self._log_probs * returns, dim=1).mean()
        return loss, mean_reward, mean_length

    @staticmethod
    def _rescale(returns: np.ndarray, mask: np.ndarray) -> np.ndarray:
        returns_copy = np.array(returns, copy=True)
        returns_copy[mask == 0] = np.nan
        mean = np.stack([np.nanmean(returns_copy, axis=1)] * returns.shape[1], axis=1)
        std = np.stack([np.nanstd(returns_copy, axis=1)] * returns.shape[1], axis=1)
        eps = np.finfo(np.float32).eps.item()
        return (returns - mean) / (std + eps)

    @staticmethod
    def _get_mean_reward(reward: np.ndarray, mask: np.ndarray) -> float:
        copy_of_reward = np.array(reward, copy=True)
        copy_of_reward[mask == 0] = np.nan
        return np.nansum(copy_of_reward, axis=1).mean()

    @staticmethod
    def _get_mean_length(mask: np.ndarray) -> float:
        return mask.sum(axis=1).mean()


def _seeding(batch_size, n_epochs, seed: int | None = None):
    """Create a sequence of seeds for training/validation functions."""
    seq = np.random.SeedSequence(entropy=seed).generate_state(batch_size * n_epochs)
    seq = seq.reshape(n_epochs, batch_size)
    # Gymnasium environments accept Python integers only, so we need to unstack matrix.
    seeds = [[int(seed) for seed in np.unstack(seq[i, :])] for i in range(seq.shape[0])]
    return seeds


def train(  # noqa: PLR0914, PLR0917
    policy: nn.Module,
    envs: gym.Env,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    n_epochs: int,
    device: str = "cpu",
    seed: int | None = None,
    pbar: bool = True,
) -> tuple:
    """Train the REINFORCE agent.

    Parameters
    ----------
    policy : nn.Module
        The neural network model representing the agent.
    envs : gym.Env
        The environment(s) in which the agent will be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the agent.
    gamma : float
        The discount factor for future rewards.
    n_epochs : int
        The number of training epochs.
    device : str, optional
        The device to run the training on, by default "cpu".
    seed : int or None, optional
        The seed for random number generation, by default None.
    pbar : bool, optional
        Enable the progressbar.

    Returns
    -------
    tuple
        A tuple containing the trained agent, the dictionary of training statistics.

    """
    batch_size = envs.observation_space.shape[0]

    env_config = envs.unwrapped.spec.kwargs["config"]
    max_steps = env_config["duration"] * env_config["policy_frequency"]
    del env_config
    # Define seeds
    if seed:
        torch.manual_seed(seed=seed)
    seeds = _seeding(seed=seed, batch_size=batch_size, n_epochs=n_epochs)
    # Prepare iteration memory
    memory = Memory(discount_factor=gamma)
    policy.train()
    policy = policy.to(device)
    # Reserve lists for stats
    rewards = []
    lengths = []
    postfix = None
    # Train the agent
    for iteration in range(n_epochs):
        # Reset the environments with seeds and clean the memory
        obs, _ = envs.reset(seed=seeds[iteration])
        memory.clear()
        for _ in tqdm.trange(
            1,
            max_steps,
            postfix=postfix,
            desc=f"Iteration {iteration + 1: >2}/{n_epochs}",
            disable=not pbar,
        ):
            # Get actions and log probabilities
            actions, log_probs = policy(obs.to(device))
            # Get new observations and push them to the memory
            obs, reward, terminated, truncated, _ = envs.step(actions)
            done = memory.append(log_probs, reward, terminated, truncated)
            if done:
                break

        optimizer.zero_grad()
        # Compute reward and apply padding mask
        gradient, mean_reward, mean_length = memory.loss()
        # Record metrics
        rewards.append(mean_reward)
        lengths.append(mean_length)
        postfix = {"reward": mean_reward, "length": mean_length}
        # Make the back propogation
        gradient.backward()
        optimizer.step()
    norm_len = [length / max_steps for length in lengths]
    stats = {"reward": rewards, "length": lengths, "norm_length": norm_len}
    return policy, stats


def validate(  # noqa: PLR0917
    policy: nn.Module,
    env: gym.Env,
    n_episodes: int,
    device: str = "cpu",
    seed: int | None = None,
    pbar: bool = True,
) -> dict:
    """Validate the performance of a reinforcement learning agent over multiple epochs.

    Parameters
    ----------
    policy : nn.Module
        The neural network model representing the agent.
    env : gym.Env
        The environment in which the agent operates.
    n_episodes : int
        The number of epochs to run the validation.
    device : str, optional
        The device to run the computations on, by default "cpu".
    seed : int | None, optional
        The seed for random number generation, by default None.
    pbar : bool, optional
        Enable the progressbar.

    Returns
    -------
    dict
        A dictionary containing the mean rewards, mean lengths, and gradients for each
        epoch.

    """
    env_config = env.unwrapped.spec.kwargs["config"]
    max_steps = env_config["duration"] * env_config["policy_frequency"]
    del env_config
    # Define seeds
    if seed:
        torch.manual_seed(seed=seed)
    seeds = _seeding(seed=seed, batch_size=1, n_epochs=n_episodes)

    policy.eval()
    policy = policy.to(device)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    with torch.no_grad():
        for episode in tqdm.trange(n_episodes, desc="Validation", disable=not pbar):
            obs, _ = env.reset(seed=seeds[episode][0])
            done = False
            while not done:
                actions = policy(obs.to(device))
                if actions.dim() > 1:
                    actions = actions[0]
                obs, _, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

    norm_lengths = [length / max_steps for length in env.length_queue]
    stats = {
        "reward": env.return_queue,
        "length": env.length_queue,
        "norm_length": norm_lengths,
    }
    return stats
