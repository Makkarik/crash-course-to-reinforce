# Crash Course to REINFORCE

**Driving through the busy highway with REINFORCE autopilot**

---

## Introduction

This project demonstrates the implementation of the REINFORCE algorithm to train a reinforcement learning agent for autonomous highway driving. The agent learns to navigate through complex traffic scenarios by leveraging vectorized environments, policy networks, and advanced memory management (with padding to handle asynchronous episode terminations). Both discrete and continuous action spaces are supported, and the REINFORCE agent is benchmarked against baseline agents from the [rl-agents](https://github.com/eleurent/rl-agents) library.

## Installation

Before you begin, ensure that you have [Poetry](https://pypi.org/project/poetry/) installed for dependency management. It is recommended that Poetry creates a local virtual environment in your project's folder.

1. **Configure Poetry to use in-project virtual environments:**

    ```bash
    poetry config virtualenvs.in-project true
    ```

2. **Clone the repository and install dependencies:**

    ```bash
    git clone git@github.com:Makkarik/crash-course-to-reinforce.git
    cd crash-course-to-reinforce
    poetry install
    ```

3. **Install additional packages for baselines (if needed):**

    The benchmarking notebooks rely on legacy modules from [rl-agents](https://github.com/eleurent/rl-agents) and [finite-mdp](https://github.com/eleurent/finite-mdp). Activate your virtual environment and install them:

    ```bash
    # For Windows
    .venv/Scripts/activate

    # For Mac/Linux
    source .venv/bin/activate

    pip install git+https://github.com/eleurent/rl-agents
    pip install git+https://github.com/eleurent/finite-mdp
    ```

    **Note:** These installations might downgrade some packages. Restart your kernel if necessary.


## Environment Description

The project uses a highway driving simulation based on [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv), which simulates realistic highway scenarios with complex vehicle dynamics.

### Action Spaces

The environment supports three types of action spaces:

- **Continuous Action Space:**  
  - **Acceleration:** A continuous value in the range \([-5.0,\, 5.0]\) m/s².
  - **Steering (Wheel Angle):** A continuous value in the range \(\left[-\frac{\pi}{4},\, \frac{\pi}{4}\right]\) radians.

- **Discrete Action Space:**  
  A quantized version of the continuous action space, often discretized into 3 levels.

- **Discrete Meta-Action Space:**  
  A set of 5 high-level actions:
    - 0 : LANE_LEFT - Change to the left lane. 
    - 1 : IDLE - Maintain the current lane. 
    - 2 : LANE_RIGHT - Change to the right lane. 
    - 3 : FASTER - Increase speed. 
    - 4 : SLOWER - Decrease speed.

These meta-actions are internally mapped to acceleration and steering commands via an integrated controller.

### Controller and Vehicle Dynamics

The environment’s built-in controllers convert agent actions into vehicle control commands:

- **Longitudinal Controller:**

Uses a proportional control law:

\[
a = K_p \,(v_r - v)
\]

where:
- \(v_r\) is the reference speed (set by the meta-actions),
- \(v\) is the current speed,
- \(K_p = 1/\tau_a\) with \(\tau_a = 0.6\).

- **Lateral Controller:**

Implemented using a proportional-derivative approach split into:

- **Positional Control:**

  \[
  \begin{cases}
  v_{\text{lat},r} = -K_{p,\text{lat}} \Delta_{\text{lat}}, \\
  \Delta \psi_{r} = \arcsin\left(\frac{v_{\text{lat},r}}{v}\right),
  \end{cases}
  \]

- **Heading Control:**

  \[
  \begin{cases}
  \psi_r = \psi_L + \Delta \psi_{r}, \\
  \dot{\psi}_r = K_{p,\psi} (\psi_r - \psi), \\
  \delta = \arcsin\left(\frac{1}{2}\frac{l}{v}\,\dot{\psi}_r\right),
  \end{cases}
  \]

where:
- \(\psi_L\) is the lane heading,
- \(l = 5\, \text{m}\) is the vehicle length,
- \(K_{p,\text{lat}}\) and \(K_{p,\psi}\) are controller gains.

### Kinematics and State Space

The vehicle dynamics follow the bicycle model:

\[
\begin{cases}
\dot{x}=v\cos(\psi+\beta), \\
\dot{y}=v\sin(\psi+\beta), \\
\dot{v}=a, \\
\dot{\psi}=\frac{v}{l}\sin\beta, \\
\beta=\tan^{-1}\left(\frac{1}{2}\tan\delta\right),
\end{cases}
\]

- **State Space:**  
The state includes a 5-dimensional vector for the ego-vehicle and a 7-dimensional vector for each ambient vehicle. Additional parameters (vehicle dimensions, collision flags, etc.) are also part of the state. In meta-action mode, the controller contributes extra features.

### Reward Function

The reward function balances speed, collision avoidance, and lane positioning:

\[
R(s,a) = \alpha\frac{v - v_{\min}}{v_{\max} - v_{\min}} - \beta\,\text{collision} + \gamma\frac{\text{lane index}}{\text{total lanes}}
\]

- **Speed Reward:** Incentivizes higher speeds.
- **Collision Penalty:** Penalizes collisions.
- **Lane Preference:** Rewards staying in the far right lane.

### Observation

Observations are provided as occupancy grids that capture the spatial layout around the ego-vehicle. Each grid cell contains features such as vehicle presence and relative speeds.

## Baselines and Benchmarking

The project benchmarks the REINFORCE agent against established baseline agents provided by [rl-agents](https://github.com/eleurent/rl-agents):

- **Random Agent:** Chooses actions uniformly at random.
- **Value Iteration Agent:** Computes optimal policies based on a finite MDP approximation.

Benchmarking is performed using the `benchmark` function (in `baselines_utils.py`), which runs a series of episodes under various environment configurations (e.g., 1Hz and 5Hz policy frequencies). Results are saved as CSV files and visualized using the `show_metrics` function.

## Training and Evaluation

The core training of the REINFORCE agent is implemented in `agent.py` and is further demonstrated in the `reinforce.ipynb` notebook.

### Key Components

- **Policy Networks:**  
- `PolicyNetworkDiscrete`: For discrete action spaces.
- `PolicyNetworkContinious`: For continuous action spaces.

- **Experience Memory:**  
The `Memory` class stores log probabilities, rewards, and applies a padding mechanism to handle episodes that finish at different times.

- **Training Loop:**  
The `train` function performs:
- Environment resets with seeding.
- Action selection and experience collection.
- Backpropagation after computing gradients based on full trajectories.

- **Validation:**  
The `validate` function runs the agent in a validation environment and returns metrics such as mean reward and episode length.

Refer to the `reinforce.ipynb` notebook for detailed training and evaluation examples.

## Notebooks

- **reinforce.ipynb:**  
Provides a comprehensive walkthrough of the REINFORCE training process, from setting up environments and networks to training and validation.

- **baselines.ipynb:**  
Benchmarks the REINFORCE agent against the Random and Value Iteration baseline agents. Includes visualizations and sample videos of agent performance.

## Results and Observations

- In a 1Hz policy environment, the Value Iteration Agent performs well, reliably reaching the destination.
- In a more challenging 5Hz environment, the Value Iteration Agent struggles due to the increased complexity, while the Random Agent remains ineffective.
- The REINFORCE agent is designed to handle the full trajectory information, using padding to mask terminal states and ensuring robust learning in dynamic settings.

Benchmark figures and performance metrics are generated during the experiments and are available in the results folder.

## Conclusion

This project presents a complete framework for training a REINFORCE-based agent in a realistic highway driving environment. It covers the entire pipeline—from environment setup and policy network design to training, validation, and benchmarking against baseline agents. The work demonstrates the challenges of asynchronous episode termination and highlights the benefits of padding in gradient computation for effective policy learning.

---

Happy learning and safe driving!
