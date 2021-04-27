# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:35:18 2021

@author: TAIHM
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
from scipy.stats import nakagami
import cv2
import random
from scipy import special
import torch
import torch.nn.functional as F


GRID_WIDTH = 200  # meter
GRIG_LENGTH = 200  # meter
P_max_v2v = 5 # Will recheck for max power

min = 0.1
max = 0.5

Num_AGV = 8
list_neighbr = []
duplicate_index = []
GRID_SIZE = 10
RESOLUTION = 10  # pixel
TIME_SLOT = 1  # second
SPEED = 10
gNB_HEIGHT = 10  # m
Q_code = 10  # number of code
M = 1  # Number of antennas
S = 16  # denotes the number of available phase values for each antenna element
f_c = 6e9  # carrier frequency 6GHz
c = 3e8  # speed of light
alpha = 3.76  # pathloss exponent
B = 200e3  # Hz System bandwidth
rho_max = 0.1  # SIR threshold
P_max = 10  # watt constraint transmit power per gNB
n_0 = 1e-14  # dBm noise
epsilon = 10e-5  # decoding error probability threshold
D_k = 160  # bits = 20Bytes packet length
n_k = 1024  # symbols - Channel blocklength
d = {0: 0,
     1: pi / 2,
     2: pi,
     3: 3 * pi / 2}


down_lanes = [i/2.0 for i in [4/2,4+4/2,8+4/2,12+4/2,16+4/2,20+4/2]]


def normalize(value, minTar, maxTar, minVal=-1, maxVal=1):
    return ((value - minVal) / (maxVal - minVal)) * (maxTar - minTar) + minTar
#
#
# def get_grid(coordinates):
#     return [math.ceil(coordinates[0] / GRID_SIZE) - 1, math.ceil(coordinates[1] / GRID_SIZE) - 1]
#
#
# def get_coordinates(grid):
#     return [grid[0] * GRID_SIZE + GRID_SIZE / 2, grid[1] * GRID_SIZE + GRID_SIZE / 2]


class AGV:
    def __init__(self, X):
        self.grid = X
        self.h = [random.randint(30,50) for i in range(Num_AGV)]
        self.rc = [random.randint(20,30) for i in range(Num_AGV)] # in GHz
        self.speed = [random.randint(30,70) for i in range(Num_AGV)]
        self.demand = [min + (max-min)*random.random() for i in range(Num_AGV)]
        self.duplicate_index = duplicate_index
        self.individual_time_limit = [random.randint(20,30) for i in range(Num_AGV)]

    def move(self, i):
        self.grid[i] += self.speed[i]*TIME_SLOT
      #   self.grid = get_grid(self.coordinates)



class AGVEnv(AGV):
    def __init__(self,X):
        AGV.__init__(self,X)
        self.No_AGV = Num_AGV
        self.No_gNB = 1
        self.X = X
        #self.Y = Y
        self.action_space = self.No_AGV * 2
        self.No_ant = M
        self.list_neighbr  = list_neighbr
        self.gNB_pos = [500,0]
        self.AGVs = [AGV(self.X[i]) for i in range(self.No_AGV)]
        self.dist_V2I = [[np.sqrt(pow(self.gNB_pos[0] - self.AGVs[j].grid, 2)) for j in range(self.No_AGV)] ]

        self.dist_V2V = [[np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2))  +0.05for j in range(self.No_AGV)] for i in range(self.No_AGV)]
        self.dist_V2V = np.reshape(self.dist_V2V, (8,8))
        self.g_V2I = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2I[i][j] ** (-alpha) for j in range(self.No_AGV)] for i in
                  range(self.No_gNB)]
       # if self.dist_V2V >= 0 :
        self.g_V2V = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2V[i][j] ** (-alpha) for j in range(self.No_AGV)] for i in
                  range(self.No_AGV)]
    #    else:
    #        self.g_V2V = 0
        self.observation = [self.g_V2I[0][0:Num_AGV], self.g_V2V[0][0:Num_AGV], self.demand, self.individual_time_limit]
        repeated_vel = [(self.speed[i:i + 1] * len(self.speed)) for i in range(len(self.speed))]
        rel_accel = []
        sortedList = []
        for o in range(len(repeated_vel)):
            # print(data2[i])
            for k, l in zip(range(len(repeated_vel)), range(len(repeated_vel))):
                rel_accel.append(self.speed[k] - repeated_vel[o][l])
        new_rel_accel = abs(np.reshape(rel_accel, (8,8)))
        nearest_cars = []
        for t in range(len(new_rel_accel)):
            for j in range(len(new_rel_accel)):
                nearest_cars.append((new_rel_accel[t][j])*0.05 + (self.dist_V2V[t][j])*0.95)
        new_nearest_cars = np.reshape(nearest_cars, (8, 8))


        for i in range(self.No_AGV):
            sort_idx= np.argsort(new_nearest_cars[i])
            sortedList.append(sort_idx)
            list_neighbr.append(sort_idx[1])
        #print(list_neighbr)
        sortedList = np.reshape(sortedList, (8,8))
        #print(sortedList)




    def reset(self):
        """
        Reset the initial value
        """
       # self.X = [(random.choice(down_lanes)*random.randrange(5)+random.randrange(50)) for i in range(self.No_AGV) ]  # x-coordinate of all cars
        self.gNB_pos = [500, 0]
        self.AGVs = [AGV(self.X[i]) for i in range(self.No_AGV)]
        self.dist_V2I = [[np.sqrt(
            pow(self.gNB_pos[1] - self.AGVs[j].grid, 2)  ) for j in range(self.No_AGV)] for i in range(self.No_gNB)]

        self.dist_V2V = [[np.sqrt( pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05 for j in range(self.No_AGV)] for i in range(self.No_AGV)]
        #
        self.g_V2I = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2I[i][j] ** (-alpha) for j in range(self.No_AGV)] for i
                      in
                      range(self.No_gNB)]
        # if self.dist_V2V >= 0 :
        self.g_V2V = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2V[i][j] ** (-alpha) for j in range(self.No_AGV)] for i in range(self.No_AGV)]
        self.g_V2V = np.reshape(self.g_V2V, (8,8))
        self.g_V2I = np.reshape(self.g_V2I, (8))
        v2v_nearest = [self.g_V2V[i][j] for i,j in zip(range(len(self.g_V2V)), list_neighbr)]
        self.observation = [np.reshape((self.g_V2I[:], v2v_nearest[:], self.demand[:], self.individual_time_limit[:]), (self.No_AGV*4))]
        return self.observation

    def display(self):
        # Create a blank image
        board = np.zeros([GRID_WIDTH + 1, GRIG_LENGTH + 1, 3])
        # Color the snake green
        for AGV in self.AGVs:
            board[AGV.grid, AGV.grid] = [0, 255, 0]
            board[AGV.Dest_y, AGV.Dest_x] = [0, 0, 255]
        # Display board
        cv2.imshow("Automated warehouse", np.uint8(board.repeat(RESOLUTION, 0).repeat(RESOLUTION, 1)))
        cv2.waitKey(int(1000 / SPEED))

    def step(self, actions):
        #reward = np.zeros((self.No_AGV))
        penalty = np.zeros((self.No_AGV))
        self.ass = np.zeros(self.No_AGV) # what is self.ass
        power = np.zeros((self.No_AGV))
        codeword = np.zeros((self.No_AGV))
        power[:] = actions[0][0][0:self.No_AGV]
        codeword[:] = actions[0][0][self.No_AGV:2 * self.No_AGV]
        P2 = [F.softmax(torch.tensor(power), dim=-1) for i in range(self.No_AGV)]
        P_V2I = [P2[i].numpy() * P_max for i in range(self.No_AGV)]
        P_V2V = [P2[i].numpy() * P_max_v2v for i in range(self.No_AGV)]
        for i in range(len(codeword)):
            if codeword[i] > 0.5:
                codeword[i]=1
            else:
                codeword[i]=0
        #print(codeword)


        done = [False for i in range(self.No_AGV)]
        ###----------------Calculat positions and channel gains---------------------
      #  speed = random.normal(loc=5, scale=2, size=(self.No_AGV))
        next_state = 0
        # print("grid before")
        # print(self.grid)
        # print(self.grid[0])
        # print("speed before")
        # print(self.speed[0])
        for i in range(self.No_AGV):
            self.move(i)
            #done[i] = True
        # print("grid after")
        # print(self.grid[0])
        #self.display()
        self.dist_V2I = [[np.sqrt(
            pow(self.gNB_pos[1] - self.AGVs[j].grid, 2)  ) for j in range(self.No_AGV)] for i in range(self.No_gNB)]

        self.dist_V2V = [[np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05 for j in range(self.No_AGV)] for i in range(self.No_AGV)]
        #
        self.g_V2I = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2I[i][j] ** (-alpha) for j in range(self.No_AGV)] for i
                      in
                      range(self.No_gNB)]
        # if self.dist_V2V >= 0 :
        self.g_V2V = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2V[i][j] ** (-alpha) for j in range(self.No_AGV)] for i in range(self.No_AGV)]
        self.g_V2V = np.reshape(self.g_V2V, (8,8))
        self.g_V2I = np.reshape(self.g_V2I, (8))
        ####----------------Calculate SINR ------------------------------------------
        # SINR1 = np.zeros((self.No_AGV))
        # Max_cluster = 1
        # for j in range(self.No_AGV):
        #     SINR_max = {}
        #     for i in range(self.No_gNB):
        #         beam = [row[int(Code[i][j])] for row in self.C]
        #         SINR1_temp = (np.absolute(np.sqrt(P[i][j])*np.matmul(beam,np.conj(self.h[i][j]))))**2
        #         SINR_max[i] = SINR1_temp
        #     num_ass_gNB = 0
        #     for key, value in sorted(SINR_max.items(), key=lambda item: item[1],reverse=True):
        #         if num_ass_gNB < Max_cluster:
        #             SINR1[j] += value
        #             self.ass[key][j] = 1
        #             num_ass_gNB +=1

        ################  clustering-----------------------------)
        # SINR1 = np.zeros((self.No_AGV))
        # for j in range(self.No_AGV):
        #     SINR_max = {}
        #     for i in range(self.No_gNB):
        #         beam = [row[int(Code[i][j])] for row in self.C]
        #         SINR_temp = (np.absolute(np.sqrt(P[i][j]) * np.matmul(beam, np.conj(self.h[i][j])))) ** 2
        #         SINR_max[i] = SINR_temp
        #     num_ass_gNB = 0
        #     index = max(SINR_max.keys(), key=(lambda k: SINR_max[k]))
        #     max_sinr = max(SINR_max.items(), key=lambda item: item[1])
        #     SINR1[j] += max_sinr[1]
        #     penalty[j] += P[index][j]
        #     self.ass[index][j] = 1
        #     for key, value in sorted(SINR_max.items(), key=lambda item: item[1], reverse=True):
        #         if key != index and rho_max <= abs(value / max_sinr[1]) <= 1:
        #             SINR1[j] += value
        #             penalty[j] += P[key][j]
        #             self.ass[key][j] = 1
        ###############################################
        SINR_V2V = np.zeros((self.No_AGV))
        SINR_V2I = np.zeros((self.No_AGV))
        Rate_V2I = np.zeros((self.No_AGV))
        Rate_V2V = np.zeros((self.No_AGV))
        reward   = np.zeros((self.No_AGV))
        URLLC_Rate = np.zeros((self.No_AGV))
        Error = np.zeros((self.No_AGV))
        Interference = np.zeros((self.No_AGV))
        # print("gv2i")
        # print(P_V2I)
        # V2I
        for i in range(self.No_AGV):
            SINR_V2I[i] = P_V2I[0][i]*self.g_V2I[i]
        #V2V
            for j in list_neighbr:
                if self.g_V2V[i][j] == math.inf:
                    SINR_V2V[i] = 0
                else:
                    SINR_V2V[i] = P_V2V[0][i] * self.g_V2V[i][j]
        #print(str(len(SINR_V2V)) + "and v2i is" + str(len((SINR_V2I))))
        # SINR_V2I = list(SINR_V2I)
        # SINR_V2V = list(SINR_V2V)
        for i in range(self.No_AGV):
            Rate_V2I[i]= np.log2(1 + SINR_V2I[i]/n_0**2)
            Rate_V2V[i] = np.log2(1 + SINR_V2V[i]/n_0**2)
        for i in range(Num_AGV):
            if codeword[i]==0:
                self.demand[i]-=Rate_V2I[i]/1000
                penalty= Rate_V2I[i]-10
                reward[i] = Rate_V2I[i] - penalty
            else:
                self.demand[i]-=Rate_V2V[i]/1000
                reward[i] = Rate_V2V[i]
            self.individual_time_limit[i]-=1
        for i in range(Num_AGV):
            if self.demand == 0:
                done[i]=True

        #print(reward)


            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # print('Signal['+str(i) + '] = ' + str((SINR1[i])))
            # print('SINR['+str(i) + '] = ' + str((SINR[i])))
            # print('Interference['+str(i) + '] = ' + str((Interference[i])))


        #      Rate = np.log2(1 + SINR[i])
        #     # print('Rate['+str(i) + '] = ' + str((Rate[i])))
        #     x = np.log(2) * np.sqrt(n_k) * (Rate[i] - D_k / n_k)
        #     # print(x)
        #     Error[i] = (1 / 2) * math.erfc(x / np.sqrt(2))
        #     # print('Error['+str(i) + '] = ' + str((Error[i])))
        #     if Error[i] != 0:
        #         URLLC_Rate[i] = Rate[i] - np.sqrt(2) * np.sqrt(
        #             ((1 + SINR[i]) ** 2 - 1) / (n_k * (1 + SINR[i]) ** 2)) * special.erfcinv(2 * Error[i]) / np.log(2)
        #     else:
        #         URLLC_Rate[i] = Rate[i]
        #     reward[i] = URLLC_Rate[i] - penalty[i]
        # # plot_durations_ass(self,self.ass)
        # self.observation = [np.reshape(self.h[i], (self.No_AGV * self.No_ant)) for i in range(self.No_gNB)]
        # next_state = self.observation
        v2v_nearest = [self.g_V2V[i][j] for i, j in zip(range(len(self.g_V2V)), list_neighbr)]
        self.observation = [np.reshape((self.g_V2I[:], v2v_nearest[:], self.demand[:], self.individual_time_limit[:]), (self.No_AGV*4))]
        next_state = self.observation
        return next_state, reward, done, reward


# import matplotlib.ticker as mticker
# plt.close("all")
# env = AGVEnv()
# states = env.observation
# startTime = time.time()
# rewards_episode = []
# rewards_avg = []
# def plot_durations():
#     g=plt.figure(2)
#     plt.clf()
#     ax = g.add_subplot(111)
#     durations_reward = torch.FloatTensor(rewards_episode)
#     durations_reward_avg = torch.FloatTensor(rewards_avg)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.plot(durations_reward.numpy(), label = 'reward')
#     plt.plot(durations_reward_avg.numpy(), label = 'average reward')
#     plt.legend(loc='best', prop={'size': 12})
#     formatter = mticker.ScalarFormatter(useMathText=True)
#     formatter.set_powerlimits((-3,2))
#     ax.yaxis.set_major_formatter(formatter)
#     plt.pause(0.001)  # pause a bit so that plots are updated

def plot_durations_ass(env, ass):
    plt.figure(1)
    plt.clf()
    X_gNB = [env.gNB_pos[i][0] for i in range(4)]
    Y_gNB = [env.gNB_pos[i][1] for i in range(4)]
    X_AGV = [env.AGVs[i].grid[0] for i in range(env.No_AGV)]
    Y_AGV = [env.AGVs[i].grid[1] for i in range(env.No_AGV)]
    plt.plot(X_gNB, Y_gNB, 'g^', markersize=12, label='gNB')
    plt.plot(X_AGV, Y_AGV, 'rs', markersize=6, label='AGV')
    for i in range(env.No_gNB):
        for j in range(env.No_AGV):
            if ass[i][j]:
                plt.plot([env.gNB_pos[i][0], env.AGVs[j].grid[0]], [env.gNB_pos[i][1], env.AGVs[j].grid[1]], '--',
                         color='royalblue')
    plt.ylim([0, GRIG_LENGTH])
    plt.xlim([0, GRID_WIDTH])
    plt.legend(loc='best', frameon=True)
    plt.ylabel('Y (m)')
    plt.xlabel('X (m)')
    plt.grid(True)
    plt.pause(1)  # pause a bit so that plots are updated
    plt.show()

# while True:
#     actions = [np.random.rand(env.No_AGV*3) for i in range(env.No_gNB)]
#     cluster = np.zeros((env.No_gNB,env.No_AGV))
#     for i in range(env.No_gNB):
#         for j in range(env.No_AGV):
#             if env.dis[i][j] <= 150:
#                 actions[i][j] = 1
#             else:
#                 actions[i][j] = 0
#         cluster[i] = actions[i][0:env.No_AGV]
#     ass = [np.round(cluster[i]) for i in range(env.No_gNB)]
#     plot_durations_ass(ass)
#     next_state, rewards, done, Error = env.step(actions)
#     rewards_episode.append(np.min(rewards))
#     rewards_avg.append(np.mean(rewards_episode[-100:]))
#     plot_durations()
#     if all(done):
#         break
# executionTime = (time.time() - startTime)
# print(executionTime)




