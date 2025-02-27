# Required dependencies:
# pip install gym highway-env torch numpy matplotlib

import gymnasium as gym
import highway_env  # Registers the HighwayEnv with Gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time         # For measuring episode duration
import csv          # For saving data to CSV

# Set device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Define the Continuous Policy Network
# ------------------------------
class PolicyNetworkContinuous(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        """
        Initializes a neural network for continuous action outputs.
        The network outputs the mean values for each action dimension.
        A trainable log standard deviation parameter is used to define the Gaussian distribution.
        
        Args:
            input_dim (int): Dimension of the flattened observation.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of continuous actions (here, 2: acceleration and wheel angle).
        """
        super(PolicyNetworkContinuous, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # Initialize a trainable log_std parameter (for each action dimension)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        """
        Returns a Normal distribution for each action dimension.
        """
        x = self.relu(self.fc1(x))
        mean = self.fc2(x)
        std = torch.exp(self.log_std).expand_as(mean)
        # Create a Gaussian distribution for each action component.
        return torch.distributions.Normal(mean, std)

# ------------------------------
# Memory to store trajectories (log probabilities and rewards)
# ------------------------------
class Memory:
    def __init__(self):
        self.log_probs = []
        self.rewards = []
    
    def clear(self):
        self.log_probs = []
        self.rewards = []

# ------------------------------
# REINFORCE Update Function
# ------------------------------
def update_policy(policy, optimizer, memory, gamma=0.99):
    """
    Updates the policy network using the REINFORCE algorithm.
    
    The policy gradient is computed by:
      ∇θ J(θ) ≈ Σₜ (Gₜ * ∇θ log π(aₜ|sₜ)),
    where Gₜ is the discounted return from time step t.
    
    Args:
        policy (nn.Module): The policy network.
        optimizer (torch.optim.Optimizer): Optimizer for the network.
        memory (Memory): Stored trajectories (log_probs, rewards).
        gamma (float): Discount factor.
    """
    R = 0
    returns = []
    # Compute discounted returns backwards
    for r in reversed(memory.rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float).to(device)
    
    # Normalize returns to stabilize training
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    
    policy_loss = []
    for log_prob, R in zip(memory.log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    memory.clear()

# ------------------------------
# Environment Setup for Continuous Actions
# ------------------------------
def make_env_continuous():
    """
    Create and configure the Highway environment for the continuous action space.
    
    In this configuration, the environment expects two action channels:
      - Acceleration (a) in [-5, 5]
      - Wheel angle (δ) in [-π/4, π/4]
      
    The observation is configured as an occupancy grid. The environment's reward is given by:
      R(s,a) = α (v - v_min)/(v_max - v_min) - β * collision + γ (lane index)/(total_lanes)
    """
    
    config = {
        "observation": {
            "type": "OccupancyGrid",  # Can also use "Kinematics" or "TimeToCollision"
            "grid_size": [[-5, 5], [-5, 5]],  # Two dimensions: x from -5 to 5 and y from -5 to 5
            "grid_step": [2.0, 2.0],         # Specify step for each dimension
            "features": ["presence", "vx"]  # Example features: vehicle presence and relative x-speed
        },
        "simulation_frequency": 15,  # Simulation frequency in Hz
        "policy_frequency": 5,       # How often the policy is applied
        "duration": 40,              # Episode duration in seconds
        "action": {"type": "ContinuousAction"}  # Select the continuous action space
    }
    
    env = gym.make("highway-v0", config=config)
    return env

def preprocess_observation(obs):
    """
    Preprocess the observation into a flattened tensor.
    
    This function flattens the observation (which may be a grid or dict) to a one-dimensional vector.
    """
    if isinstance(obs, dict):
        obs = np.concatenate([v.flatten() for v in obs.values()])
    else:
        obs = np.array(obs).flatten()
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------------------
# Training Loop for Continuous REINFORCE Agent
# ------------------------------
def train_agent_continuous(num_episodes=500, hidden_dim=128, learning_rate=1e-3, gamma=0.99):
    """
    Trains the continuous-action REINFORCE agent in the Highway environment.
    
    The agent outputs two continuous values (acceleration and wheel angle) that are
    sampled from a Gaussian distribution parameterized by the policy network.
    Actions are clipped to the valid ranges:
      - Acceleration: [-5, 5]
      - Wheel angle: [-π/4, π/4]
    
    Args:
        num_episodes (int): Total number of training episodes.
        hidden_dim (int): Hidden layer size for the policy network.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor.
    
    Returns:
        policy (nn.Module): Trained policy network.
        episode_rewards (list): List of total rewards per episode.
    """
    env = make_env_continuous()
    
    # List to record per-episode data: (episode, duration, total_reward)
    episode_data = []
    
    # Sample an observation to determine the input dimension
    obs, _ = env.reset()
    processed_obs = preprocess_observation(obs)
    input_dim = processed_obs.shape[1]
    
    # Initialize the continuous policy network
    policy = PolicyNetworkContinuous(input_dim, hidden_dim, output_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    memory = Memory()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Record start time for this episode.
        episode_start = time.time()
        
        obs, _ = env.reset()
        processed_obs = preprocess_observation(obs)
        done = False
        ep_reward = 0
        
        while not done:
            # Get the action distribution from the policy network.
            action_dist = policy(processed_obs)
            # Sample an action from the distribution.
            action = action_dist.sample()  # shape: [1, 2]
            # Compute the log probability of the sampled action.
            log_prob = action_dist.log_prob(action).sum()
            memory.log_probs.append(log_prob)
            
            # Convert action to numpy, clip each component to the valid range.
            action_np = action.cpu().detach().numpy().squeeze()
            a = np.clip(action_np[0], -5.0, 5.0)
            delta = np.clip(action_np[1], -np.pi/4, np.pi/4)
            continuous_action = np.array([a, delta])
            
            # Step the environment with the continuous action.
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            memory.rewards.append(reward)
            ep_reward += reward
            
            # Preprocess the new observation.
            processed_obs = preprocess_observation(obs)
        
        # Compute episode duration using wall-clock time.
        episode_duration = time.time() - episode_start
        
        # Update the policy network after each episode.
        update_policy(policy, optimizer, memory, gamma)
        episode_rewards.append(ep_reward)
        
        # Record the episode data: (episode number, duration, total reward)
        episode_data.append((episode + 1, episode_duration, ep_reward))
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Duration: {episode_duration:.2f}s, Total Reward: {ep_reward:.2f}")
    
    env.close()
    
    # Save episode data to a CSV file.
    with open("episode_data_continuous.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Duration (s)", "Total Reward"])
        for row in episode_data:
            writer.writerow(row)
    
    return policy, episode_rewards

# ------------------------------
# Plotting Training Rewards
# ------------------------------
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode (REINFORCE - Continuous Agent)")
    plt.legend()
    plt.savefig("training_rewards_2.png")
    plt.close()

# ------------------------------
# Main execution block
# ------------------------------
if __name__ == "__main__":
    # Train the continuous-action REINFORCE agent
    trained_policy, rewards = train_agent_continuous(num_episodes=500)
    
    # Plot training progress
    plot_rewards(rewards)
    
    # Instructions:
    # - Install required packages: gym, highway-env, torch, numpy, and matplotlib.
    # - This code configures the Highway environment for continuous actions.
    # - The agent directly outputs two values: acceleration and wheel angle.
    # - The built-in controllers (using the formulas above) handle converting these
    #   values into vehicle dynamics.
    # - The reward function considers vehicle speed, collisions, and lane preference.
    # - Wall-clock duration for each episode is measured and recorded.
    # - Episode data (episode number, duration in seconds, total reward) is saved to
    #   "episode_data_continuous.csv" for later analysis.
    # - To run, save this script (e.g., train_continuous_reinforce.py) and execute with Python.
