"""Hyperparameters optimization for continuous action space environment."""

import numpy as np
import optuna
import torch
from optuna import trial  # noqa: F401

from src.agent import PolicyNetworkContinuous, train, validate
from src.envs import make_envs

SEED = 0x42  # Just a seed
ENV_NAME = "highway-v0"  # Environment name
NUM_ENVS = 15  # A number of independent environments
EVAL_ITERATIONS = 5  # A number of iterations for validation stage
PROGRESS = False  # Enable progressbar

TRIALS = 64  # Total number of trials
THREADS = 1  # NB: Each environment use the separate process, so the total
# number of environments is ENVS * THREADS. Change only if CPU utilization is below 50%
DEFAULT = {"lr": 1e-2, "iterations": 10, "gamma": 0.99, "hidden_dim": 128}


CONFIG = {
    "observation": {
        # Use an occupancy grid. The grid size and features can be adjusted.
        "type": "OccupancyGrid",  # or "Kinematics" / "TimeToCollision"
        "grid_size": [
            [-5, 5],
            [-5, 5],
        ],  # Two dimensions: x from -5 to 5 and y from -5 to 5
        "grid_step": [2.0, 2.0],  # Specify step for each dimension
        "features": ["presence", "vx"],  # presence and relative speed features
    },
    "simulation_frequency": 15,  # adjust as needed
    "policy_frequency": 5,
    "duration": 40,  # initial episode duration in seconds
    "action": {"type": "ContinuousAction"},  # use the discrete meta-action space
    "offscreen_rendering": True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial):  # noqa: F811
    """Objective for optimizing the policy for discrete action space environment."""
    # Define hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    iterations = trial.suggest_int("iterations", 10, 20)
    # Prepare everything for training
    envs = make_envs(ENV_NAME, NUM_ENVS, config=CONFIG)
    input_dim = envs.observation_space.shape[1]
    output_dim = envs.action_space._shape[1]  # So do I hate it
    # Train the policy
    policy = PolicyNetworkContinuous(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)
    optimized_policy = train(
        policy, envs, optimizer, gamma, iterations, device, SEED, PROGRESS
    )
    # Validate the policy
    envs = make_envs(ENV_NAME, NUM_ENVS, config=CONFIG)
    results = validate(optimized_policy, envs, EVAL_ITERATIONS, device, SEED, PROGRESS)

    return np.array(results["mean_reward"]).mean()


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Continuous policy tuning",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    if DEFAULT:
        study.enqueue_trial({**DEFAULT})
    study.optimize(objective, n_jobs=THREADS, n_trials=TRIALS)
    results = study.trials_dataframe()
    results.to_csv("./results/reinforce/continuous_hp_search.csv")
