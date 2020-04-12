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
import argparse
# import the agent
from ddpg_agent import Agent

# define modes
MODE_EXPLORE    = 0
MODE_TRAIN      = 1
MODE_TEST       = 2

## ---------------- Config Parameters -------
# experiment id
EXPERIMENT_ID = 'reacher_ddpg_multiple_agents'
SINGLE_AGENT_ENV = 'Reacher_SingleAgent_Linux/Reacher.x86_64'
MULTI_AGENT_ENV = 'Reacher_MultiAgent_Linux/Reacher.x86_64'
# Single vs Multi-Agent Environment
USE_MULTI_AGENT_ENV = False
# Program mode
MODE = MODE_TEST
# Score threshold for deciding if the environment was solved
SCORE_THRES_ENV_SOLVED = 30.0
# pretrained weights; set to None if not using pretrained weights
# PRETRAINED_WEIGHTS_ACTOR = None
# PRETRAINED_WEIGHTS_CRITIC = None
PRETRAINED_WEIGHTS_ACTOR = 'weights/reacher_ddpg_multiple_agents/checkpoint_actor.pth'
PRETRAINED_WEIGHTS_CRITIC = 'weights/reacher_ddpg_multiple_agents/checkpoint_critic.pth'
##-------------------------------------------

# moving average
def moving_average(arr, n=50) :
    idx = [i+1 for i in range(len(arr))]
    for i in range(len(arr)):
        idx[i] = min(n, idx[i])
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / idx

# explore environment by taking random actions
def explore(n_steps=1000):
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
def train_agent(n_episodes=3000, max_t=1000, reset_noise_t=100):
    scores_deque = deque(maxlen=100)
    scores_global = []
    best_score = 0
    # open log file
    with open(LOG_FILE, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i_episode in range(1, n_episodes+1):
            # reset the environment and parameters
            env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
            # reset noise
            agent.reset_noise()
            states = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            mean_score = []
            for t in range(max_t):
                # reset noise every few steps to promote exploration
                if t%reset_noise_t == 0:
                    agent.reset_noise()
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                # update learning from each agent
                for i_agent in range(num_agents):
                    agent.step(states[i_agent], actions[i_agent], rewards[i_agent], next_states[i_agent], dones[i_agent])
                scores += env_info.rewards                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode finished
                    break
                print('\rEpisode {}, Step {}\tScore: {:.3f}'.format(i_episode, t, np.mean(scores)), end="")

            scores_deque.append(np.mean(scores))
            mean_score = np.mean(scores_deque)
            scores_global.append(np.mean(scores))
            print('\r                                                      ', end="")
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, mean_score))
            writer.writerow({'episode': i_episode, 'score': mean_score})

            if mean_score >= best_score:
                print('Best score achieved. Saving model weights.')
                best_score = mean_score
                torch.save(agent.actor_local.state_dict(), WEIGHTS_ACTOR)
                torch.save(agent.critic_local.state_dict(), WEIGHTS_CRITIC)

            # Keep higher threshold during training
            if mean_score > (SCORE_THRES_ENV_SOLVED+5.0):
                print('\nEnvironment solved in {} episodes.\tAverage Score: {:.3f}'.format(i_episode, mean_score))
                break
            
    return scores_global

# Test the agent with DDPG
def test_agent(n_episodes=100, max_t=1000):
    scores_deque = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        # reset the environment and parameters
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        mean_score = []
        for t in range(max_t):
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards[0]                      # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
            print('\rStep {}  \tScore: {:.3f}'.format(t, np.mean(scores)), end="")
            mean_score.append(scores)

        mean_score = np.mean(mean_score)
        scores_deque.append(mean_score)
        print('\rEpisode {} Average Score: {:.3f}\n'.format(i_episode, mean_score), end="")

    if(np.mean(scores_deque) > SCORE_THRES_ENV_SOLVED):
        print('Environment solved.')
    print('Average score over {} episodes: {}'.format(n_episodes, np.mean(scores_deque)))

    # return score
    return scores_deque

# main function
def main(mode=MODE_EXPLORE):
    if mode == MODE_EXPLORE:
        explore()
    elif mode == MODE_TRAIN:
        # if num_agents == 1:
        n_episodes = 2000
        scores = train_agent(n_episodes=n_episodes)

        # close the environment
        env.close()

        # get moving average of score
        score_ma = moving_average(scores, n=5)
        # plot 
        plt.figure(figsize=(16,9))
        plt.title('Agent Performance', fontweight='bold')
        # training losses and iou
        plt.grid(linestyle='-', linewidth='0.2', color='gray')
        plt.plot(range(n_episodes), scores, 'b-', alpha=0.5)
        plt.plot(range(n_episodes), score_ma, 'r-')
        plt.legend(['score','averaged score'], loc='lower right', fancybox=True, framealpha=1., shadow=True, borderpad=1, prop={"size":15})
        plt.ylabel('Score', fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.show()

    elif mode == MODE_TEST:
        n_episodes = 100
        scores = test_agent(n_episodes=n_episodes)

        # close the environment
        env.close()

        # get moving average of score
        score_ma = moving_average(scores, n=20)
        # plot 
        plt.figure(figsize=(16,9))
        plt.title('Agent Performance', fontweight='bold')
        # training losses and iou
        plt.grid(linestyle='-', linewidth='0.2', color='gray')
        plt.plot(range(n_episodes), scores, 'b-', alpha=0.5)
        plt.plot(range(n_episodes), score_ma, 'r-')
        plt.legend(['score','averaged score'], loc='lower right', fancybox=True, framealpha=1., shadow=True, borderpad=1, prop={"size":15})
        plt.ylabel('Score', fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.show()

if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_agent", help="choose the multi-agent environment", action='store_true')
    parser.add_argument("--explore", help="explore the environment", action='store_true')
    parser.add_argument("--train", help="train the DDPG agent", action='store_true')
    parser.add_argument("--test", help="test the DDPG agent", action='store_true')
    args = parser.parse_args()

    # setup the model using config parameters
    if args.multi_agent:
        USE_MULTI_AGENT_ENV = True
    if args.explore:
        MODE = MODE_EXPLORE
    if args.train:
        MODE = MODE_TRAIN
    if args.test:
        MODE = MODE_TEST

    # paths
    LOGS_DIR = 'logs/'+EXPERIMENT_ID+'/'
    WEIGHTS_DIR = 'weights/'+EXPERIMENT_ID+'/'
    LOG_FILE = LOGS_DIR+'logs.csv'

    # Choose agent environment
    AGENT_ENV = None
    if USE_MULTI_AGENT_ENV == True:
        AGENT_ENV = MULTI_AGENT_ENV
    else:
        AGENT_ENV = SINGLE_AGENT_ENV

    # make directories
    os.system('mkdir -p '+LOGS_DIR)
    os.system('mkdir -p '+WEIGHTS_DIR)

    # actor and critic weights
    WEIGHTS_ACTOR = WEIGHTS_DIR+'checkpoint_actor.pth'
    WEIGHTS_CRITIC = WEIGHTS_DIR+'checkpoint_critic.pth'

    # Start the reacher unity environment
    env = UnityEnvironment(file_name=AGENT_ENV)

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
    # if num_agents == 1:
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    # load the Actor weights from file
    if PRETRAINED_WEIGHTS_ACTOR is not None:
        agent.actor_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))
        agent.actor_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_ACTOR, map_location=lambda storage, loc: storage))

    if PRETRAINED_WEIGHTS_CRITIC is not None:
        # load the Critic weights from file
        agent.critic_local.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))
        agent.critic_target.load_state_dict(torch.load(PRETRAINED_WEIGHTS_CRITIC, map_location=lambda storage, loc: storage))

    # run the main function
    main(mode=MODE)