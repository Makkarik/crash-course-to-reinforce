"""Misc utilities."""

import os

import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def mp4_to_gif(folder: str) -> None:
    """Convert MP4 video to GIF.

    Parameters
    ----------
    folder : str
        The folder containing MP4 files to be converted.

    """
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    gif_paths = [p[: p.rfind(".")] + ".gif" for p in paths]

    for video_path, gif_path in zip(paths, gif_paths):
        with imageio.get_reader(video_path) as reader:
            fps = reader.get_meta_data()["fps"]

            writer = imageio.get_writer(gif_path, fps=fps, loop=0)
            for frame in reader:
                writer.append_data(frame)
            writer.close()

        os.remove(video_path)


def show_training_results(results: dict) -> Figure:
    """Show training metrics."""
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))
    iterations = np.arange(len(results["reward"]))
    axs[0].plot(iterations, results["reward"])
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Cumulative reward")
    axs[0].grid()
    axs[0].set_title("Change of the cumulative reward over training iterations")

    axs[1].plot(iterations, results["norm_length"])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Episode normalized length")
    axs[1].grid()
    axs[1].set_title("Change of episode length over iterations")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return fig


def append_results(
    df: pd.DataFrame, results: dict, agent_label: str, env_label: str
) -> pd.DataFrame:
    """Append results from run."""
    for key, value in results.items():
        df[agent_label, env_label, key] = np.array(value)
    return df


def moving_average(input: np.ndarray, n: int = 500, mode="valid") -> np.ndarray:
    """Get the moving average."""
    output = np.convolve(np.array(input).flatten(), np.ones(n), mode=mode) / n
    if mode == "valid":
        steps = np.arange(output.size) + n // 2
    elif mode == "same":
        steps = np.arange(output.size)
    return steps, output


def compare_results(
    model_results: pd.DataFrame, baselines: pd.DataFrame, roll_lebgth: int = 1
):
    """Compare inference results."""
    baselines = baselines.xs("Highway 5Hz", axis=1, level=1).mean()
    fig, axs = plt.subplots(figsize=(16, 6), ncols=2)

    agents, baseline_agents = [], []
    for option in model_results.columns:
        if option[0] not in agents:
            agents.append(option[0])

    for option in baselines.index:
        if option[0] not in baseline_agents:
            baseline_agents.append(option[0])

    for metric_idx, metric in enumerate(["reward", "norm_length"]):
        for agent_idx, agent in enumerate(agents):
            data = model_results[agent, "Highway 5Hz", metric]
            axs[metric_idx].plot(
                *moving_average(data, roll_lebgth),
                label=f"{agent}",
                color=f"C{agent_idx}",
            )
            mean_value = model_results[agent, "Highway 5Hz", metric].mean()
            axs[metric_idx].axhline(
                mean_value,
                color=f"C{agent_idx}",
                label=f"{agent} (mean value)",
                linestyle="--",
            )
    for metric_idx, metric in enumerate(["reward", "length"]):
        for agent_idx, agent in enumerate(baseline_agents):
            axs[metric_idx].axhline(
                baselines[agent, metric],
                color=f"C{agent_idx + 2}",
                label=f"{agent} (mean value)",
                linestyle="--",
            )
        axs[metric_idx].legend()
        axs[metric_idx].grid()
        axs[metric_idx].set_xlabel("Episode")

    axs[0].set_ylabel("Cumulative Reward")
    axs[1].set_ylabel("Episode normalized length")
    axs[0].set_title("Cumulative reward at validation")
    axs[1].set_title("Episode length at validation")
    axs[1].set_ylim(0, 1.03)

    plt.tight_layout()
    plt.show()
    return fig
