"""Visualization utilities for Q-learning agent."""

import os
from itertools import product

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rl_agents.agents.common.factory import agent_factory


def benchmark(  # noqa: PLR0914
    env_configs: dict[dict],
    agent_configs: dict[dict],
    episodes: int = 1000,
    folder: str = "./results/baselines",
) -> pd.DataFrame:
    """Run a benchmark for the given environment and agent configurations.

    Parameters
    ----------
    env_configs : dict of dict
        A dictionary where keys are environment names and values are dictionaries
        containing environment configurations.
    agent_configs : dict of dict
        A dictionary where keys are agent names and values are dictionaries
        containing agent configurations.
    episodes : int, optional
        The number of episodes to run for each agent-environment combination,
        by default 100.
    folder : str, optional
        The folder where results and videos will be saved, by default
        "./results/baselines".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the benchmark. The DataFrame has a
        MultiIndex with agent names, environment names, and metrics
        ("reward" and "length").

    """
    # The docstring has been generated with the Copilot, using the following prompt:
    # "Finish the doctring for the function in Numpy style."

    os.makedirs(folder, exist_ok=True)
    trigger = lambda x: x == 0 or x == (episodes - 1)

    columns = pd.MultiIndex.from_tuples(
        product(agent_configs.keys(), env_configs.keys(), ["reward", "length"])
    )
    results = pd.DataFrame(index=np.arange(episodes), columns=columns)
    # Iterate over all possible combinations of agent and environment configurations
    total_runs = len(agent_configs) * len(env_configs)
    runs = product(agent_configs.items(), env_configs.items())
    for idx, (agent_config, env_config) in enumerate(runs, start=1):
        env_name, agent_name = env_config[0], agent_config[0]
        length = env_config[1]["duration"] * env_config[1]["policy_frequency"]

        # Define the environment
        env = gym.make("highway-v0", render_mode="rgb_array", config=env_config[1])
        env = gym.wrappers.RecordVideo(
            env=env,
            video_folder=folder,
            name_prefix=agent_name + "-" + env_name,
            video_length=length,
            episode_trigger=trigger,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env=env, buffer_length=episodes)

        # Define the agent
        agent = agent_factory(env, agent_config[1])

        # Run the iteration for the given number of episodes
        for _ in tqdm.trange(episodes, desc=f"Run {idx}/{total_runs}", unit="episode"):
            (obs, _) = env.reset()
            done, truncated = False, False
            while not (done or truncated):
                action = agent.act(obs)
                obs, _, done, truncated, _ = env.step(action)

        env.close()
        # Write the results to the DataFrame
        rewards = np.array(env.return_queue)
        results[agent_name, env_name, "reward"] = rewards
        norm_lengths = np.array(env.length_queue) / length
        results[agent_name, env_name, "length"] = norm_lengths

    results.to_csv(os.path.join(folder, "results.csv"))
    return results


def moving_average(input: np.ndarray, n: int = 500, mode="valid") -> np.ndarray:
    """Get the moving average."""
    output = np.convolve(np.array(input).flatten(), np.ones(n), mode=mode) / n
    if mode == "valid":
        steps = np.arange(output.size) + n // 2
    elif mode == "same":
        steps = np.arange(output.size)
    return steps, output


def cumulative(input: np.ndarray) -> np.ndarray:
    """Get the cumulative value."""
    input = np.array(input).flatten()
    temp = 0
    for i in range(input.size):
        temp += input[i]
        input[i] = temp
    steps = np.arange(input.size)
    return steps, input


def show_metrics(results: pd.DataFrame, roll_length: int = 1) -> Figure:
    """Display statistics of the benchmark results.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing the benchmark results with multi-index columns (agent,
        environment, metric).
    roll_length : int, optional
        Length of the rolling window for smoothing the data, by default 1.

    Returns
    -------
    Figure
        Matplotlib Figure object containing the plots of the statistics.

    """
    # The docstring has been generated with the Copilot, using the following prompt:
    # "Write the doctring for the function, using Numpy style."
    agents, envs, metrics = [], [], []
    for option in results.columns:
        if option[0] not in agents:
            agents.append(option[0])
        if option[1] not in envs:
            envs.append(option[1])
        if option[2] not in metrics:
            metrics.append(option[2])
    # Plot the results
    fig, axs = plt.subplots(ncols=len(metrics), nrows=len(envs), figsize=(16, 9))
    for metric_idx, metric in enumerate(metrics):
        for env_idx, env in enumerate(envs):
            for agent_idx, agent in enumerate(agents):
                # Plot metrics for each agent
                data = results[agent, env, metric]
                axs[env_idx, metric_idx].plot(
                    *moving_average(data, roll_length),
                    color=f"C{agent_idx}",
                    label=f"{agent} (smoothed)",
                )
                axs[env_idx, metric_idx].axhline(
                    data.mean(),
                    linestyle="--",
                    color=f"C{agent_idx}",
                    label=f"{agent} (mean)",
                )
            # Finish the plot
            label = f"normalized {metric}" if metric == "length" else metric
            axs[env_idx, metric_idx].set_title(f"Episode {label} in {env} environment")
            axs[env_idx, metric_idx].set_xlabel("Episode")
            axs[env_idx, metric_idx].set_ylabel(f"Episode {label}")
            axs[env_idx, metric_idx].legend()
            axs[env_idx, metric_idx].grid()
            if metric == "length":
                axs[env_idx, metric_idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    return fig
