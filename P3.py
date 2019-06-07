from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="/home/arjun/Documents/Udacity/Deep_RL/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
print(states)
state_size = states.shape[1]


import torch
from collections import deque
import matplotlib.pyplot as plt

from agent import Agents

agents = Agents(state_size=state_size,
                action_size=action_size,
                num_agents=num_agents,
                random_seed=0)


def maddpg(n_episodes=5000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agents.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations  # get the next state
            reward = env_info.rewards  # get the reward
            done = env_info.local_done  # see if episode has finished

            agents.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break
        scores_window.append(np.max(score))  # save most recent score
        scores.append(np.max(score))  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5 and len(scores_window) >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agents.actor1_local.state_dict(), 'checkpoint_actor1.pth')
            torch.save(agents.actor2_local.state_dict(), 'checkpoint_actor2.pth')
            torch.save(agents.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

scores = maddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

