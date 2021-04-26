# -*- coding: utf-8 -*-

import os
from datetime import datetime
import AGVEnv
import logging
import argparse
import numpy as np
import torch
from maddpg import MADDPGAgentTrainer
import AGVEnv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
plt.rcParams.update({'font.size': 12})

MINIBATCH_SIZE = 50


# ################## SETTINGS ######################
down_lanes = [i/2.0 for i in [4/2,4+4/2,8+4/2,12+4/2,16+4/2,20+4/2]]
X = [random.choice(down_lanes)*random.randrange(5)+random.randrange(50) for i in range(AGVEnv.Num_AGV) ] # x-coordinate of all cars
Y = 0


width = 0
height = 2000/20

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'sarl_model'

n_neighbor = 1


# Create logger
logger = logging.getLogger("ddpg_multi")
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
dirname = 'runs/{}'.format(now)
if not os.path.exists(dirname):
    os.makedirs(dirname)

filehandler = logging.FileHandler(filename='{}/ddpg_multi.log'.format(dirname))
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
# Uncomment to enable console logger
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.INFO)
logger.addHandler(steamhandler)


reward_ep = []
reward_avg = []

def plot_durations():
    h = plt.figure(1)
    plt.clf()
    ax = h.add_subplot(111)
    durations_reward_avg = torch.FloatTensor(reward_avg)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.plot(durations_reward_avg.numpy(), label = 'DDPG')
    plt.legend(loc='best', prop={'size': 12})
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    plt.pause(0.001)  # pause a bit so that plots are updated

def maddpg(env, num_agents, agent, n_episodes=500, max_t=2000, print_every=50):
    """Train DDPG Agent

    Params    
    ======
        env (object): UAV environment instance
        num_agents (int): number of agents
        agent (DDPGMultiAgent): agent instance
        writer (VisWriter): Visdom visualiser for realtime plots
        n_episodes (int): number of episodes to train the network
        max_t (int): number of timesteps in each episode
        print_every (int): how often to print the progress
    """
    reward_step = []
    for i_episode in range(1, n_episodes+1):
        running_reward = []
        states1 = env.reset()
        states = [states1[i] for i in range(env.No_gNB)]
        #states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        #action_Alpha = np.zeros(num_agents)
        best_maxt = 0
        training_step = 0
        for t in range(max_t):
            # for i in range(num_agents):
            #     action_Alpha[i] = DQN_agent.actAlpha(states[i])
            actions = agent.act(states)
            next_states1,rewards,dones, rates  = env.step(actions)                # send all actions to UAV environment
            running_reward.append(sum(rates))
            next_states = [next_states1[i] for i in range(env.No_gNB)]
            agent.step(states, actions, sum(rewards), next_states, all(dones))
            states = next_states                                            # roll over states to next time step

            if all(dones):
                training_step = t
                break # exit loop if episode finished
        agent.update()
        print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, training_step, np.mean(running_reward)))
        reward_ep.append(np.mean(running_reward))
        reward_avg.append(np.mean(reward_ep[-500:]))
        plot_durations()


    filename = 'results/DDPG_' + '_Reward' 
    f = open(filename, "w")
    f.write("# Reward Avg_reward \n")        # column names
    np.savetxt(f, np.array([reward_ep, reward_avg]).T)
    f.close() 
    return score


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=25, help="Total number of episodes to train")
    parser.add_argument("--max_t", type=int, default=2000, help="Max timestep in a single episode")
    parser.add_argument("--vis", dest="vis", action="store_true", help="Use visdom to visualise training")
    parser.add_argument("--no-vis", dest="vis", action="store_false", help="Do not use visdom to visualise training")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--info", type=str, default="", help="Use this to attach notes to your runs")
    parser.set_defaults(vis=True)

    args = parser.parse_args()

    env = AGVEnv.AGVEnv(X=X)



    #env.new_random_game()  # initialize parameters in env

    # number of agents
    num_agents = 1
    print('Number of agents:', num_agents)

    state = env.observation
    state_shape = env.No_AGV*4
    action_size = env.action_space
    agent = MADDPGAgentTrainer(state_shape, action_size, num_agents, random_seed=48, dirname=dirname, print_every=100, model_path=args.model)

    scores = maddpg(env, num_agents, agent, n_episodes=args.num_episodes, max_t=args.max_t)

if __name__ == "__main__":    
    main()