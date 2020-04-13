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
import time
# import the agent
from ddpg_agent import Agent
from agent_env import *

# define modes
MODE_EXPLORE    = 0
MODE_TRAIN      = 1
MODE_TEST       = 2

## ---------------- Config Parameters -------
# experiment id
EXPERIMENT_ID = ''

# number of episodes
N_EPISODES = 2000

# Program mode
MODE = MODE_TEST
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
def explore(n_episodes=100, n_steps=1000):
    # reset the environment and parameters
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = []
    for i_episode in range(1, n_episodes+1):
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

# Train the agent with DDPG
def train_agent(n_episodes=2000, max_t=5000, reset_noise_t=100):
    n_episodes_avg = 100
    scores_deque = deque(maxlen=n_episodes_avg)
    scores_global = []
    best_score = -100.0
    # open log file
    with open(LOG_FILE, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'score', 'avg_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i_episode in range(1, n_episodes+1):
            start_time = time.time()
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
                print('\rEpisode: {}\t| Step: {}\t| Score: {:.3f}'.format(i_episode, t, np.mean(scores)), end="")
            
            time_taken = (time.time() - start_time)
            scores_deque.append(np.mean(scores))
            mean_score = np.mean(scores_deque)
            scores_global.append(np.mean(scores))
            print('\r                                                      ', end="")
            print('\rEpisode: {}\t| Score: {:.3f}\t| Average Score ({} episodes): {:.3f}\t | Time Taken: {:.2f}s'.format(i_episode, np.mean(scores), n_episodes_avg, mean_score, time_taken))
            writer.writerow({'episode': i_episode, 'score': np.mean(scores), 'avg_score': mean_score})

            if np.mean(scores) >= best_score:
                print('Best score achieved. Saving model weights.')
                best_score = np.mean(scores)
                torch.save(agent.actor_local.state_dict(), WEIGHTS_ACTOR)
                torch.save(agent.critic_local.state_dict(), WEIGHTS_CRITIC)

            # Keep higher threshold during training
            if mean_score > (SCORE_THRES_ENV_SOLVED+(0.2*SCORE_THRES_ENV_SOLVED)):
                print('\nEnvironment solved in {} episodes.\tAverage Score: {:.3f}'.format(i_episode, mean_score))
                break
            
    return i_episode, scores_global

# Test the agent with DDPG
def test_agent(n_episodes=100, max_t=5000):
    n_episodes_avg = 100
    scores_deque = deque(maxlen=n_episodes_avg)
    for i_episode in range(1, n_episodes+1):
        # reset the environment and parameters
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
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
            print('\rEpisode: {}\t| Step: {}\t| Score: {:.3f}'.format(i_episode, t, np.mean(scores)), end="")
            mean_score.append(scores)

        mean_score = np.mean(mean_score)
        scores_deque.append(mean_score)
        print('\r                                                      ', end="")
        print('\rEpisode: {}\t| Score: {:.3f}\t| Average Score ({} episodes): {:.3f}'.format(i_episode, mean_score, n_episodes_avg, np.mean(scores_deque)))

    if(np.mean(scores_deque) > SCORE_THRES_ENV_SOLVED):
        print('Environment solved.')
    print('Average score over {} episodes: {}'.format(n_episodes, np.mean(scores_deque)))

    # return score
    return scores_deque

# main function
def main(mode, n_episodes):
    if mode == MODE_EXPLORE:
        explore()
    elif mode == MODE_TRAIN:
        n_episodes, scores = train_agent(n_episodes=n_episodes)

        # close the environment
        env.close()

        # get moving average of score
        score_ma = moving_average(scores, n=100)
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
    parser.add_argument("--env", help="environment to be used", default="tennis")
    parser.add_argument("--explore", help="explore the environment", action='store_true')
    parser.add_argument("--train", help="train the DDPG agent", action='store_true')
    parser.add_argument("--test", help="test the DDPG agent", action='store_true')
    parser.add_argument("--n_episodes", help="number of episodes", default=2000, type=int)
    args = parser.parse_args()

    # setup the model using config parameters
    if args.env == 'reacher_single':
        agent_env_dict = env_dict[IDX_ENV_REACHER_SINGLE_AGENT]
    elif args.env == 'reacher_multi':
        agent_env_dict = env_dict[IDX_ENV_REACHER_MULTI_AGENTS]
    elif args.env == 'crawler':
        agent_env_dict = env_dict[IDX_ENV_CRAWLER]
    elif args.env == 'tennis':
        agent_env_dict = env_dict[IDX_ENV_TENNIS]
    else:
        raise('unknown agent environment')
    
    # Agent environment
    AGENT_ENV = agent_env_dict['env']
    # Prefix
    RUN_ID_PREFIX = agent_env_dict['prefix']
    # Score threshold for deciding if the environment was solved
    SCORE_THRES_ENV_SOLVED = agent_env_dict['solved_score_thres']
    # Pretrained weights
    PRETRAINED_WEIGHTS_ACTOR = agent_env_dict['pretrained_weights_actor']
    PRETRAINED_WEIGHTS_CRITIC = agent_env_dict['pretrained_weights_critic']

    N_EPISODES = args.n_episodes

    if args.explore:
        MODE = MODE_EXPLORE
    if args.train:
        MODE = MODE_TRAIN
    if args.test:
        MODE = MODE_TEST
        N_EPISODES = 100


    # paths
    EXPERIMENT_ID = RUN_ID_PREFIX+EXPERIMENT_ID
    LOGS_DIR = 'logs/'+EXPERIMENT_ID+'/'
    WEIGHTS_DIR = 'weights/'+EXPERIMENT_ID+'/'
    LOG_FILE = LOGS_DIR+'logs.csv'

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

    # Instantiate the agent
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
    main(mode=MODE, n_episodes=N_EPISODES)