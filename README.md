# EnvSage-RL


https://github.com/user-attachments/assets/2a0d6f4e-89cb-4eee-ba52-6fba883d4e56


LLM-guided reinforcement learning with environment reasoning.

EnvSage-RL introduces a meta-learning layer where a Large Language Model observes reinforcement learning episodes, analyzes environment dynamics, forms hypotheses, and proposes targeted experiments to improve training.

---

# Motivation

Traditional RL agents learn through trial-and-error and often require millions of steps.

EnvSage-RL introduces a reasoning loop:

1. RL agent runs an episode
2. Episode data is analyzed
3. LLM forms hypotheses about environment dynamics
4. LLM proposes controlled experiment episodes
5. Training parameters adapt based on results

Goal: Instead of blindly optimizing rewards, the system attempts to **understand the environment as a dynamic system**.
The same way a human wouldn't blindly press random buttons a millions times when first time playing Minecraft. But rather try and extract the maximum insights, rules, environment dynamics while taking minimal actions. Quality > Quantity.

---

# Key Idea

The LLM acts as a **training strategist** rather than a passive analyzer.

Example:

```
Hypothesis: environment contains constant downward acceleration (gravity).

Experiment:
Run 3 episodes with no actions.

Observation:
State variable 5 decreases consistently.

Conclusion:
Gravity confirmed.
```

The LLM then adjusts training strategies accordingly.

---

# Architecture

```
RL Agent → Episode Logger → Feature Analyzer → LLM Strategist
     ↑                                             ↓
 Trainer ← Strategy Adapter ← Experiment Planner
```

---

# Tech Stack

| Component | Tool |
|---|---|
Language | Python 3.12 |
RL Framework | Stable-Baselines3 |
Environment | Gymnasium |
Deep Learning | PyTorch |
Logging | TensorBoard |
Experiment Tracking | Weights & Biases |
Packaging | uv |

---

# Repository Structure

```
envsage-rl/

envs/
environment wrappers

agent/
rl training logic

analysis/
episode statistics + dynamic feature extraction

reflection/
LLM strategist

experiments/
baseline comparisons

logs/
tensorboard outputs

notebooks/
analysis and visualization
```

---

# Training Workflow

1. Run RL episode
2. Extract trajectory data
3. Compute statistical features
4. LLM analyzes environment dynamics
5. LLM proposes experiment episodes
6. Trainer executes experiments
7. Adjust RL parameters

---

# Feature Extraction

Instead of passing raw trajectory data to the LLM, the system computes structured features.

Examples:

| Feature | Description |
|---|---|
mean velocity | average movement |
reward acceleration | change in reward slope |
state drift | natural system movement |
action sensitivity | state response to actions |

These features allow the LLM to reason about system dynamics.

---

# Example Hypothesis Loop

```
Episode results → LLM reasoning:

Observation:
state[5] decreases every timestep.

Hypothesis:
environment contains constant downward force.

Experiment:
run 3 episodes without actions.

Validation:
state[5] decreases linearly.

Conclusion:
gravity present.
```

---

# Example Environments

- CartPole
- LunarLander-v2
- MiniGrid
- Doom Gym (future)

The system is designed to generalize across environments.

---

# HuggingFace Integration

Training logs and experiment results can be uploaded to HuggingFace.

Possible artifacts:

- training metrics
- reflection logs
- experiment reports
- environment dynamics graphs

---

# TODO

1. Prevent LLM cheating via environment recognition.

Ideas:

- randomize environment names
- hide observation dimensions
- randomize reward scaling
- inject noise into observation order

---

# Future Work

- multi-agent reflection
- causal discovery of environment rules
- automated curriculum learning
- integration with world models
