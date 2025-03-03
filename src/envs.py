"""Utilities for creating vectorized environments for reinforcement learning."""

from os import cpu_count

import gymnasium as gym
import highway_env  # noqa: F401
import torch


def make_envs(
    environment: str, num_envs: int = 0, config: dict | None = None
) -> gym.Env:
    """Create a vectorized environment for reinforcement learning.

    Parameters
    ----------
    environment : str
        The name of the environment to create.
    num_envs : int, optional
        The number of environments to create. Defaults to 0, which sets it to the number
        of CPU cores.
    config : dict, optional
        Environment configuration. Defaults to an empty dictionary.

    Returns
    -------
    gym.Env
        A vectorized and transformed Gym environment.

    """
    num_envs = cpu_count() if num_envs <= 0 else num_envs
    envs = gym.make_vec(
        environment, num_envs=num_envs, config=config, vectorization_mode="async"
    )
    envs = gym.wrappers.vector.FlattenObservation(envs)
    envs = gym.wrappers.vector.TransformObservation(
        envs, lambda x: torch.from_numpy(x).float()
    )
    envs = gym.wrappers.vector.TransformAction(envs, lambda x: x.numpy(force=True))
    return envs


def make_env(environment: str, config: dict | None = None) -> gym.Env:
    """Create an environment for validation with video recording.

    Parameters
    ----------
    environment : str
        The name of the environment to create.
    config : dict, optional
        Environment configuration. Defaults to an empty dictionary.

    Returns
    -------
    gym.Env
        A Gym environment with observation and action transformations, and video
        recording enabled.

    """
    env = gym.make(environment, render_mode="rgb_array", config=config)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda x: torch.from_numpy(x).unsqueeze(0).float(), env.observation_space
    )
    env = gym.wrappers.TransformAction(
        env, lambda x: x.numpy(force=True), env.action_space
    )
    return env
