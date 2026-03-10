import gymnasium as gym
from agent.sage_agent import SageAgent


env = gym.make("LunarLander-v3", render_mode="human")
agent = SageAgent(env)
obs = env.reset()
print(f"Starting observation: {obs}")

total_reward = 0
num_episodes = 1000

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}")
    done = False
    terminated = False
    obs = env.reset()
    total_reward = 0
    while not (done or terminated):
        action = agent.select_action(obs)  
        next_obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        if done :
            print(f"Episode {episode + 1} finished! Total reward: {total_reward}")
env.close()
    