# Required dependencies:
# pip install gym highway-env torch numpy

import gymnasium as gym
import highway_env  # Registers the HighwayEnv with Gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Set device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Define the Policy Network
# ------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes a simple fully-connected network.
        
        Args:
            input_dim (int): Dimension of the observation space.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of discrete actions (here, 5 meta-actions).
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # Output probability distribution over actions.
        return self.softmax(x)

# ------------------------------
# Memory to store trajectories
# ------------------------------
class Memory:
    def __init__(self):
        self.log_probs = []
        self.rewards = []
    
    def clear(self):
        self.log_probs = []
        self.rewards = []

# ------------------------------
# REINFORCE update function
# ------------------------------
def update_policy(policy, optimizer, memory, gamma=0.99):
    """
    Update the policy network using the REINFORCE algorithm.
    
    The policy gradient is computed by:
      ∇θ J(θ) ≈ Σ_t (G_t * ∇θ log π(a_t|s_t)),
    where G_t is the discounted return from time step t.
    
    Args:
        policy (PolicyNetwork): The policy network.
        optimizer (torch.optim.Optimizer): Optimizer.
        memory (Memory): Collected trajectories (log_probs, rewards).
        gamma (float): Discount factor.
    """
    # Compute discounted returns
    R = 0
    returns = []
    for r in reversed(memory.rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float).to(device)
    
    # Normalize returns for more stable learning
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    
    policy_loss = []
    for log_prob, R in zip(memory.log_probs, returns):
        # Negative sign because we want to maximize expected return.
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    memory.clear()

# ------------------------------
# Environment and Training Setup
# ------------------------------
def make_env():
    """
    Create and configure the Highway environment for the discrete meta-action space.
    
    We use an occupancy grid observation. Other options (e.g., kinematics table) can be chosen
    by modifying the environment configuration as per:
    https://github.com/Farama-Foundation/HighwayEnv/blob/master/docs/observations/index.md
    
    The environment's reward is computed as:
      R(s,a) = α (v - v_min)/(v_max - v_min) - β * collision + γ (lane_index)/(total_lanes)
    where:
      - v, v_min, v_max are the current, minimum, and maximum speeds.
      - α, β, γ are coefficients.
    """

    # Configure the environment:
    config = {
        "observation": {
            # Use an occupancy grid. The grid size and features can be adjusted.
            "type": "OccupancyGrid",  # or "Kinematics" / "TimeToCollision"
            "grid_size": [[-5, 5], [-5, 5]],  # Two dimensions: x from -5 to 5 and y from -5 to 5
            "grid_step": [2.0, 2.0],         # Specify step for each dimension
            "features": ["presence", "vx"]  # presence and relative speed features
        },
        "simulation_frequency": 15,  # adjust as needed
        "policy_frequency": 5,
        "duration": 40,              # episode duration in seconds
        "action": {'type': 'DiscreteMetaAction'},  # use the discrete meta-action space
    }
    # Create the environment.
    env = gym.make("highway-v0", config=config)
    return env

def preprocess_observation(obs):
    """
    Preprocess the observation to a flattened vector.
    Depending on the observation type (here occupancy grid), flatten it to feed into the network.
    """
    # If observation is a dict or multi-dimensional array, flatten it.
    if isinstance(obs, dict):
        obs = np.concatenate([v.flatten() for v in obs.values()])
    else:
        obs = np.array(obs).flatten()
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------------------
# Training Loop for Task #1
# ------------------------------
def train_agent(num_episodes=500, hidden_dim=128, learning_rate=1e-3, gamma=0.99):
    env = make_env()
    
    # Example: Get the observation dimension from a sample observation
    obs, _ = env.reset()
    processed_obs = preprocess_observation(obs)
    input_dim = processed_obs.shape[1]
    output_dim = 5  # Five meta-actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER

    policy = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    memory = Memory()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        processed_obs = preprocess_observation(obs)
        done = False
        ep_reward = 0
        
        while not done:
            # Get action probabilities from the policy network.
            probs = policy(processed_obs)
            # Sample an action from the probability distribution.
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            # Save log probability for the action.
            memory.log_probs.append(m.log_prob(action))
            
            # Step the environment with the chosen meta-action.
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            # The reward is computed using the formula:
            # R(s,a) = α (v - v_min)/(v_max - v_min) - β * collision + γ (lane index)/(total_lanes)
            # where the internal dynamics (vehicle speed, collisions, lane changes) are handled by the environment.
            memory.rewards.append(reward)
            ep_reward += reward
            
            # Preprocess the new observation.
            processed_obs = preprocess_observation(obs)
        
        # Update the policy after each episode.
        update_policy(policy, optimizer, memory, gamma)
        episode_rewards.append(ep_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {ep_reward:.2f}")
    
    env.close()
    return policy, episode_rewards

# ------------------------------
# Plotting Training Rewards
# ------------------------------
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode (REINFORCE - Discrete Agent)")
    plt.legend()
    plt.savefig("training_rewards_1.png")
    plt.close()

# ------------------------------
# Main execution block
# ------------------------------
if __name__ == "__main__":
    # Train the discrete REINFORCE agent
    trained_policy, rewards = train_agent(num_episodes=500)
    
    # Plot training progress
    plot_rewards(rewards)
    
    # Instructions:
    # - Ensure highway-env and required dependencies are installed.
    # - The environment configuration uses an occupancy grid observation.
    # - The agent learns a policy for selecting one of the 5 meta-actions, which the environment converts
    #   into vehicle acceleration and steering commands using built-in controllers (refer to formulas above).
    # - The reward function incorporates factors such as maximum speed, collision penalties, and the duration
    #   spent in the far right lane.
    # - To run the code, simply execute: python <this_script_name.py>
