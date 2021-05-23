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
from logs import Logger


logs = Logger(model_name="AC_RL", data_name="Logging")



plt.rcParams.update({'font.size': 12})

MINIBATCH_SIZE = 50

# ################## SETTINGS ######################
down_lanes = [i/2.0 for i in [4/2,4+4/2,8+4/2,12+4/2,16+4/2,20+4/2]]


Y = 0

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
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.INFO)
logger.addHandler(steamhandler)


reward_ep = []
reward_avg = []
new_reward_ep = []
new_reward_avg = []
h_avg_ep = []
h_avg = []
power_avg_ep = []
power_avg = []
hv2i_avg = []
hv2i_ep = []
adm_ep = []


def rates_plot_durations():
    h = plt.figure(1)
    plt.clf()
    ax = h.add_subplot(111)
    durations_reward_avg = torch.FloatTensor(reward_avg)
    # new_durations_reward_avg = torch.FloatTensor(new_reward_avg)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.plot(durations_reward_avg.numpy(), label = 'T_rates')
    # plt.plot(new_durations_reward_avg.numpy(), label='Rewards')
    plt.legend(loc='best', prop={'size': 12})
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    plt.pause(0.001)  # pause a bit so that plots are updated


def latency_plot_durations():
    h = plt.figure(2)
    plt.clf()
    ax = h.add_subplot(111)
    new_durations_reward_avg = torch.FloatTensor(new_reward_avg)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average latency')
    plt.plot(new_durations_reward_avg.numpy(), label='Rewards')
    plt.legend(loc='best', prop={'size': 12})
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    plt.pause(0.001)  # pause a bit so that plots are updated

def adm_plot_durations():
    h = plt.figure(14)
    plt.clf()
    ax = h.add_subplot(111)
    new_durations_reward_avg = torch.FloatTensor(adm_ep)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average adm')
    plt.plot(new_durations_reward_avg.numpy(), label='adm')
    plt.legend(loc='best', prop={'size': 12})
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    plt.pause(0.001)  # pause a bit so that plots are updated

def h_plot_durations():
    h = plt.figure(3)
    plt.clf()
    ax = h.add_subplot(111)
    new_durations_h_avg = torch.FloatTensor(h_avg)
    # new_durations_h_v2i = torch.FloatTensor(hv2i_ep)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average h-decision')
    plt.plot(new_durations_h_avg.numpy(), label='h')
    # plt.plot(new_durations_h_v2i.numpy(), label='h-V2I')
    plt.legend(loc='best', prop={'size': 12})
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3,2))
    ax.yaxis.set_major_formatter(formatter)
    plt.pause(0.001)  # pause a bit so that plots are updated

def power_plot_durations():
    h = plt.figure(4)
    plt.clf()
    ax = h.add_subplot(111)
    new_durations_h_avg = torch.FloatTensor(power_avg)
    plt.title('Training DDPG')
    plt.xlabel('Episode')
    plt.ylabel('Average power')
    plt.plot(new_durations_h_avg.numpy(), label='Power')
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
    for i_episode in range(1, n_episodes+1):
        running_reward = []
        new_running_reward = []
        new_running_h = []
        new_running_power = []
        running_hv2i = []
        running_adm = []

        states1, adm_rest = env.reset()
        states = [states1[i] for i in range(len(states1))]
        agent.reset()
        score = np.zeros(num_agents)
        #action_Alpha = np.zeros(num_agents)
        best_maxt = 0
        training_step = 0
        for t in range(max_t):
            actions1,actions2 = agent.act(states)
            actions = np.concatenate((actions1,actions2))
            next_states1,rewards,dones, h_all, Power, adm, rates  = env.step(actions,t,adm_rest)                # send all actions to UAV environment
            running_reward.append(np.mean(rates))
            new_running_reward.append(sum(rewards))
            new_running_h.append(np.mean(h_all))
            running_adm.append(np.mean(adm))
            new_running_power.append(np.mean(Power))
            # running_hv2i.append(h_v2i)
            next_states = next_states1
            print(adm)
            agent.step(states, actions, sum(rewards), next_states, all(dones))
            states = next_states  # roll over states to next time step
            if all(dones):
                training_step = t
                # logs.newLog(np.mean(rates), np.mean(rewards),t)
                break # exit loop if episode finished

        agent.update(i_episode)

        print('Episode {} \t avg length: {} \t T_rates: {}'.format(
                i_episode, training_step, np.mean(running_reward)))
        print('Episode {} \t avg length: {} \t Reward: {}'.format(
                i_episode, training_step, np.mean(new_running_reward)))
        print("V2I selection counts : ",len(env.count_v2i))
        print("V2V selection counts : ",len(env.count_v2v))
        print("Avg h_decisn : ", np.mean(new_running_h))
        print("Avg power : ", np.mean(new_running_power))
        # print("hv2i : ", np.mean(running_hv2i))
        print("adm_mean : ", np.mean(running_adm))
        reward_ep.append(np.mean(running_reward))
        new_reward_ep.append(np.mean(new_running_reward))
        # hv2i_ep.append(np.mean(running_hv2i))
        h_avg_ep.append(np.mean(new_running_h))
        power_avg_ep.append(np.mean(new_running_power))
        reward_avg.append(np.mean(reward_ep))
        new_reward_avg.append(np.mean(new_reward_ep))
        h_avg.append(np.mean(h_avg_ep))
        power_avg.append(np.mean(power_avg_ep))
        adm_ep.append(np.mean(running_adm))

         # plot graphs
        rates_plot_durations()
        latency_plot_durations()
        h_plot_durations()
        power_plot_durations()
        adm_plot_durations()
    jaboulouka = input("Press any key to exit")

    filename = 'results/DDPG_' + '_Reward' 
    f = open(filename, "w")
    f.write("# Reward Avg_reward \n")        # column names
    np.savetxt(f, np.array([reward_ep, reward_avg]).T)
    np.savetxt(f, np.array([new_reward_ep, new_reward_avg]).T)
    f.close() 
    return score




def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=500, help="Total number of episodes to train")
    parser.add_argument("--max_t", type=int, default=1000, help="Max timestep in a single episode")
    parser.add_argument("--vis", dest="vis", action="store_true", help="Use visdom to visualise training")
    parser.add_argument("--no-vis", dest="vis", action="store_false", help="Do not use visdom to visualise training")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--info", type=str, default="", help="Use this to attach notes to your runs")
    parser.set_defaults(vis=True)

    args = parser.parse_args()

    env = AGVEnv.AGVEnv()


    #env.new_random_game()  # initialize parameters in env

    # number of agents
    num_agents = env.No_gNB
    print('Number of agents:', num_agents)



    state = env.reset
    state_shape = env.No_AGV*4
    action_size = env.action_space
    agent = MADDPGAgentTrainer(state_shape, action_size, num_agents, random_seed=48, dirname=dirname, print_every=100, model_path=args.model)

    scores = maddpg(env, num_agents, agent, n_episodes=args.num_episodes, max_t=args.max_t)


if __name__ == "__main__":    
    main()