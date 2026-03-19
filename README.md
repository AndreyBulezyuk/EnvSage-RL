# EnvSage-RL


https://github.com/user-attachments/assets/2a0d6f4e-89cb-4eee-ba52-6fba883d4e56


LLM-guided reinforcement learning with environment reasoning.

EnvSage-RL introduces a meta-learning layer where a Large Language Model observes reinforcement learning episodes, analyzes environment dynamics, forms hypotheses, and proposes targeted experiments to improve training.

---

# Motivation

Author: [Andrey Bulezyuk](https://www.linkedin.com/in/andreybulezyuk/)

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

# Training Workflow

0. For each Episode
1. Run 1 - 50 RL episodes
2. Analyze taken actions, rewards and observations
2.1. Extract Environment Dynamics and Constants
2.2. Create hypothesis about the Environment, it's Dynamics and Constant
2.3. Run 1 or more Experiments for each hypothesis to in/validate it
2.4. Update the Knowledge about the Environments Dynamics and Constants
3. Tune Training Algorithm Parameters

---

# Feature Extraction

Instead of passing raw trajectory data to the LLM, the system computes dynamic features/variables (Wind, Acceleration, Number of People in Frame, etc. ) or constants (e.g: Gravity, Duration of XYZ).

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
state[5] decreases by 0.2 every timestep.

Hypothesis:
environment contains constant downward force.

Experiment:
run 3 episodes without actions.

Validation:
state[5] decreases linearly.

Conclusion:
obs[5] = gravity with -0.2 at each timestep 
```


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
