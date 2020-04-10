# Import the environment
from unityagents import UnityEnvironment
# import other packages
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import csv
import os
# import the agent
from ddpg_agent import Agent

# define modes
MODE_EXPLORE    = 0
MODE_TRAIN      = 1
MODE_TEST       = 2

# experiment id
EXPERIMENT_ID = 'reacher_ddpg_orig'

# paths
LOGS_DIR = 'logs/'+EXPERIMENT_ID+'/'
WEIGHTS_DIR = 'weights/'+EXPERIMENT_ID+'/'
LOG_FILE = LOGS_DIR+'logs.csv'

# make directories
os.system('mkdir -p '+LOGS_DIR)
os.system('mkdir -p '+WEIGHTS_DIR)

# actor and critic weights
WEIGHTS_ACTOR = WEIGHTS_DIR+'checkpoint_actor.pth'
WEIGHTS_CRITIC = WEIGHTS_DIR+'checkpoint_critic.pth'

# pretrained weights; set to None if not using pretrained weights
PRETRAINED_WEIGHTS_ACTOR = None
PRETRAINED_WEIGHTS_CRITIC = None
# PRETRAINED_WEIGHTS_ACTOR = WEIGHTS_ACTOR
# PRETRAINED_WEIGHTS_CRITIC = WEIGHTS_CRITIC

# Start the reacher unity environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

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
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Instantiate all agents
agent = None
agents = []
if num_agents == 1:
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    # load the Actor weights from file
    if PRETRAINED_WEIGHTS_ACTOR is not None:
        agent.actor_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))
        agent.actor_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))

    if PRETRAINED_WEIGHTS_CRITIC is not None:
        # load the Critic weights from file
        agent.critic_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))
        agent.critic_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))

else:
    for i in range(num_agents):
        agents.append(Agent(state_size=state_size, action_size=action_size, random_seed=2))

        if PRETRAINED_WEIGHTS_ACTOR is not None:
            # load the Actor weights from file
            agents[i].actor_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))
            agents[i].actor_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))

        if PRETRAINED_WEIGHTS_CRITIC is not None:
            # load the Critic weights from file
            agents[i].critic_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))
            agents[i].critic_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))

# explore environment by taking random actions
def explore(n_steps=200):
    # reset the environment and parameters
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = []
    for _ in range(n_steps):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores.append(env_info.rewards)                    # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# Train the agent with DDPG
def ddpg(n_episodes=1000, max_t=500):
    scores = []
    # open log file
    with open(LOG_FILE, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i_episode in range(1, n_episodes+1):
            # reset the environment and parameters
            env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
            agent.reset()
            states = env_info.vector_observations                  # get the current state (for each agent)
            score = 0
            mean_score = []
            for t in range(max_t):
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_state = env_info.vector_observations[0]      # get next state (for each agent)
                reward = env_info.rewards[0]                      # get reward (for each agent)
                done = env_info.local_done[0]                     # see if episode finished
                agent.step(state, actions[0], reward, next_state, done)
                score += env_info.rewards[0]                      # update the score (for each agent)
                state = next_state                                # roll over states to next time step
                if done:                                          # exit loop if episode finished
                    break
                print('\rEpisode {}, Step {}\tScore: {:.3f}'.format(i_episode, t, score), end="")
                mean_score.append(score)

            mean_score = np.mean(mean_score)
            scores.append(mean_score)
            print('\r                                                      ', end="")
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, mean_score))
            writer.writerow({'episode': i_episode, 'score': mean_score})

            torch.save(agent.actor_local.state_dict(), WEIGHTS_ACTOR)
            torch.save(agent.critic_local.state_dict(), WEIGHTS_CRITIC)
            
    return scores

# main function
def main(mode=MODE_EXPLORE):
    if mode == MODE_EXPLORE:
        explore()
    elif mode == MODE_TRAIN:
        if num_agents == 1:
            scores = ddpg()
        else:
            scores = ddpg_multiple_agents()

        # close the environment
        env.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

if __name__ == '__main__':
    main(mode=MODE_TRAIN)