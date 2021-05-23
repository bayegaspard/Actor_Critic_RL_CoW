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
import pandas as pd



GRID_WIDTH = 200  # meter
GRIG_LENGTH = 200  # meter
P_max_v2v = 5 # Will recheck for max power

Num_AGV = 8
list_neighbr = []
duplicate_index = []
dist_V2V = []
dist_V2V_step = []
index_list = []


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
    def __init__(self,X_cord,h,rc,speed,demand,individual_time,S):
        self.grid = X_cord
        self.h = h#random.randint(100,500)
        self.rc = rc#random.randint(20,30)  # in GHz
        self.speed = speed#random.randint(1, 5)
        # self.list_neighbor = list_neighbr
        # self.acceleration = [random.randint(10, 15)]
        self.demand = demand#random.randint(100,200)
        self.duplicate_index = duplicate_index
        self.individual_time_limit = individual_time#random.randint(200,300)
        self.S = S


      #   self.grid = get_grid(self.coordinates)

# (self,X_cord,h,rc,speed,demand,individual_time)

class AGVEnv():
    def __init__(self):
        self.No_AGV = Num_AGV
        self.No_gNB = 1
        # demand = [1,2,3,4,5,6,7,8]
        # h=[1,2,3,4,5,6,7,8]
        # rc = [2,3,1,9,1,5,3,7]
        # individual_time = [2,3,4,5,7,9,8,1]
        # X = [3,4,7,5,9,6,7,3]
        # speed = [1,2,3,4,5,6,7,8]
        self.action_space = self.No_AGV * 4
        self.No_ant = M
        self.gNB_pos = [0,0]
        self.count_v2i = []
        self.count_v2v = []


        # self.t_delay[]

        # self.AGVs = [AGV(X[i],h[i],rc[i],speed[i],demand[i],individual_time[i]) for i in range(Num_AGV)]


    def move(self):
        for i in range(Num_AGV):
            self.AGVs[i].grid += self.AGVs[i].speed*TIME_SLOT


    def reset(self):
        """
        Reset the initial value
        """
        self.list_nb = list_neighbr
        self.dist_V2V = dist_V2V
        self.dist_V2V_step = dist_V2V_step
        self.index_list = index_list # We need to change this name
        self.dist_V2I = dist_V2V
        adn_ind = np.zeros((Num_AGV))
        # demand = [1/40,2/40,3/40,4/40,5/40,6/40,7/40,8/40]
        # demand = [2, 2, 2, 2, 2, 2, 2, 2]
        # h=[10,10,10,10,10,10,10,10]
        # rc = [5,5,5,5,5,5,5,5]
        # individual_time = [20,20,20,20,20,20,20,20]
        # X = [3,4,7,5,9,6,2,8]
        # speed = [1,1,1,1,1,1,1,1]

        h = [random.randint(100,120) for i in range(self.No_AGV)]
        rc = [random.randint(10,12) for i in range(self.No_AGV)]
        individual_time = [random.randint(5,7) for i in range(self.No_AGV)]
        X = [random.randint(1, 10) for i in range(self.No_AGV)]
        speed = [random.randint(1, 5) for i in range(self.No_AGV)]
        demand = [random.randint(1,2) for i in range(self.No_AGV)]
        S = list(demand)
        global h_v2i
        h_v2i = (sum(h)/Num_AGV)/2

        for i in range(Num_AGV):
            self.AGVs = [AGV(X[i],h[i],rc[i],speed[i],demand[i],individual_time[i],S[i]) for i in range(Num_AGV)]
        repeated_speed = [[self.AGVs[i].speed] * Num_AGV for i in range(Num_AGV)]
        repeated_speed = np.reshape(repeated_speed,(Num_AGV,Num_AGV))
        relative_speed = []
        relative_dist = []
        repeated_dist = [[self.AGVs[i].grid] * Num_AGV for i in range(Num_AGV)]
        repeated_dist = np.reshape(repeated_dist, (Num_AGV, Num_AGV))
        for i in range(Num_AGV):
            for k,l in zip(range(Num_AGV),range(Num_AGV)):
                relative_speed.append(self.AGVs[k].speed - repeated_speed[i][l])
        for i in range(Num_AGV):
            for k,l in zip(range(Num_AGV),range(Num_AGV)):
                # relative_dist.append(self.AGVs[k].grid - repeated_dist[i][l])
                relative_dist.append(np.sqrt(pow(self.AGVs[k].grid - repeated_dist[i][l], 2)) + 0.5)

        relative_speed = abs(np.reshape(relative_speed,(Num_AGV,Num_AGV)))
        relative_dist = abs(np.reshape(relative_dist,(Num_AGV,Num_AGV)))
        relative_speed = pd.DataFrame(relative_speed)
        relative_dist = pd.DataFrame(relative_dist)



        latency = np.zeros((Num_AGV,Num_AGV))
        c_demand = (pd.DataFrame(demand)*pd.DataFrame(rc))
        df = c_demand
        h_pd = 1/pd.DataFrame(h)
        for i in range(len(c_demand)):
            for j in range(Num_AGV):
                latency[i][j] = (demand[i]*rc[i])/h[j]

        compute_dist_speed = ((relative_speed*0.3) + (relative_dist*0.3) +(latency+0.3)) #        compute_dist_speed = ((relative_speed*0.3) + (relative_dist*0.3)) #

        compute_dist_speed = compute_dist_speed.values.tolist()
        compute_dist_speed = np.reshape(compute_dist_speed,(Num_AGV,Num_AGV))

        for i in range(Num_AGV):
            sortedidx = np.argsort(compute_dist_speed[i])
            if latency[i][sortedidx[1]] < individual_time[i] and sortedidx[1] not in self.list_nb :
                self.list_nb.append(sortedidx[1]) # We have to fix repetition of indeces
            elif latency[i][sortedidx[2]] < individual_time[i] and sortedidx[2] not in self.list_nb:
                self.list_nb.append(sortedidx[2])
                # self.list_nb.append(1000000)
            elif latency[i][sortedidx[3]] < individual_time[i]  and sortedidx[3] not in self.list_nb:
                self.list_nb.append(sortedidx[3])
            elif latency[i][sortedidx[4]] < individual_time[i] and sortedidx[4] not in self.list_nb :
                self.list_nb.append(sortedidx[4]) # We have to fix repetition of indeces
            elif latency[i][sortedidx[5]] < individual_time[i]  and sortedidx[5] not in self.list_nb:
                self.list_nb.append(sortedidx[5])
            elif latency[i][sortedidx[6]] < individual_time[i]  and sortedidx[6] not in self.list_nb:
                self.list_nb.append(sortedidx[6])
            elif latency[i][sortedidx[7]] < individual_time[i]  and sortedidx[7] not in self.list_nb:
                self.list_nb.append(sortedidx[7])
            else:
                self.list_nb.append(1000000)

        All_time_limit_reset = np.zeros((self.No_AGV))
        All_demand_reset = np.zeros((self.No_AGV))

        for i in range(Num_AGV):
            All_time_limit_reset[i]=self.AGVs[i].individual_time_limit
            All_demand_reset[i]= self.AGVs[i].demand
            if self.list_nb[i]==1000000:
                index_list.append(i)
        self.list_nb = self.list_nb[:Num_AGV]
            # index_list.append(self.list_nb[i].index(1000000))
        self.dist_V2I = [np.sqrt(pow(self.gNB_pos[0] - self.AGVs[j].grid, 2)) for j in range(self.No_AGV)]

        # for i, j in zip(range(self.No_AGV), self.list_nb):

        if index_list == []:
               self.dist_V2V = [np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05 for i, j in zip(range(self.No_AGV), self.list_nb)]
        else:
            for i, j in zip(range(self.No_AGV), self.list_nb):
                if i in index_list:
                    self.dist_V2V.append(1000000)
                else:
                    self.dist_V2V.append(np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05)

        # self.dist_V2V = self.dist_V2V[:Num_AGV]
        # self.dist_V2V = [np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) +0.05 for i,j in zip(range(self.No_AGV) ,self.list_nb)] # check list of neighbors
        self.g_V2V = [[pow(c / (4 * pi * f_c), 2) * self.dist_V2V[j] ** (-alpha) for j in range(self.No_AGV)]]
        self.g_V2I = [pow(c / (4 * pi * f_c), 2) * self.dist_V2I[j] ** (-alpha) for j in range(self.No_AGV)]
        self.observation = [self.g_V2I[:],self.g_V2V[0][:], All_time_limit_reset[:], All_demand_reset[:]]
        self.observation=np.reshape(self.observation,(Num_AGV*4))
        return self.observation , adn_ind


    # def display(self):
    #     # Create a blank image
    #     board = np.zeros([GRID_WIDTH + 1, GRIG_LENGTH + 1, 3])
    #     # Color the snake green
    #     for AGV in self.AGVs:
    #         board[AGV.grid, AGV.grid] = [0, 255, 0]
    #         board[AGV.Dest_y, AGV.Dest_x] = [0, 0, 255]
    #     # Display board
    #     cv2.imshow("Automated warehouse", np.uint8(board.repeat(RESOLUTION, 0).repeat(RESOLUTION, 1)))
    #     cv2.waitKey(int(1000 / SPEED))

    def step(self, actions,t, adm_rest):
        power = np.zeros((2*self.No_AGV))
        codeword = np.zeros((self.No_AGV))
        power = actions[0][:]
        codeword = actions[1][:Num_AGV]
        h_dcsn = actions[1][Num_AGV:2*Num_AGV]
        P2 = [(torch.tensor(power[i])) for i in range(2*self.No_AGV)]
        P_V2I = [P2[i].numpy() * P_max for i in range(2*self.No_AGV)]
        P_V2V = [P2[i].numpy() * P_max_v2v for i in range(2*self.No_AGV)]
        for i in range(len(codeword)):
            if codeword[i] > 0.3:
                codeword[i]=1
            else:
                codeword[i]=0

        done = [False for i in range(Num_AGV)]
        self.move()
        self.dist_V2I_step = [np.sqrt(pow(self.gNB_pos[0] - self.AGVs[j].grid, 2)  ) + 0.05 for j in range(self.No_AGV)]

        if self.index_list == []:
               self.dist_V2V_step = [np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05 for i, j in zip(range(self.No_AGV), self.list_nb)]
        else:
            for i, j in zip(range(self.No_AGV), self.list_nb):
                if i in self.index_list:
                    self.dist_V2V_step.append(1000000)
                else:
                    self.dist_V2V_step.append(np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) + 0.05)
                # self.dist_V2V_step_loop[i] = self.dist_V2V_step[i]


        self.dist_V2V_step = self.dist_V2V[:Num_AGV]
        # self.dist_V2V_step = [np.sqrt(pow(self.AGVs[i].grid - self.AGVs[j].grid, 2)) +0.05 for i,j in zip(range(self.No_AGV) ,self.list_nb)]
        self.g_V2I_step = [pow(c / (4 * pi * f_c), 2) * self.dist_V2I_step[j] ** (-alpha) for j in range(self.No_AGV)]
        self.g_V2V_step = [pow(c / (4 * pi * f_c), 2) * self.dist_V2V_step[j] ** (-alpha) for j in range(self.No_AGV)]
        self.g_V2I_step = np.reshape(self.g_V2I_step, (Num_AGV))
        SINR_V2V = np.zeros((self.No_AGV))
        SINR_V2I = np.zeros((self.No_AGV))
        Rate_V2I = np.zeros((self.No_AGV))
        Rate_V2V = np.zeros((self.No_AGV))
        reward_UL   = np.zeros((self.No_AGV))
        reward_EXE   = np.zeros((self.No_AGV))
        reward = np.zeros((self.No_AGV))
        new_total_delay = np.zeros((self.No_AGV))
        l_V2I = np.zeros((self.No_AGV))
        l_V2V = np.zeros((self.No_AGV))

        T_Rate = np.zeros((self.No_AGV))
        Power = np.zeros((self.No_AGV))
        h_step = np.zeros((self.No_AGV))
        h_all = np.zeros((self.No_AGV))


        for i in range(Num_AGV):
            SINR_V2I[i] = P_V2I[i]*self.g_V2I_step[i]
            SINR_V2V[i] = P_V2V[i] * self.g_V2V_step[i]

        for i in range(Num_AGV):
            Rate_V2I[i]= np.log2(1 + SINR_V2I[i]/n_0**2)
            Rate_V2V[i] = np.log2(1 + SINR_V2V[i]/n_0**2)
            Rate_V2I[i] = Rate_V2I[i]/1000
            Rate_V2V[i] = Rate_V2V[i]/1000
            if self.AGVs[i].demand < Rate_V2I[i] :
                 Rate_V2I[i] = self.AGVs[i].demand
                 Rate_V2V[i] = self.AGVs[i].demand

            if self.list_nb[i]==1000000:
                self.AGVs[i].demand -= Rate_V2I[i]

                if (self.AGVs[i].individual_time_limit)*0.5 > t:
                    reward_UL[i] += 1
                    Power[i] = P_max * P_V2I[i]
                    T_Rate[i] = Rate_V2I[i]
                    self.count_v2i.append(0)
                else:
                    reward_UL[i] = 0
                    Power[i] = 0
                    T_Rate[i] = 0
                    adm_rest[i] = 1
                    done[i] = True

            else:
                if codeword[i] == 1:
                    if (self.AGVs[i].individual_time_limit)*0.5 > t:
                        reward_UL[i] += 1
                        Power[i] = P_max * P_V2I[i]
                        T_Rate[i] = Rate_V2I[i]
                        self.count_v2i.append(0)
                        print("met")
                    else:
                        reward_UL[i] = 0
                        Power[i] = 0
                        T_Rate[i] = 0
                        adm_rest[i]=1
                        done[i] = True

                else:
                    if  (self.AGVs[i].individual_time_limit)*0.5 > t:
                        reward_UL[i] += 2
                        Power[i] = P_max * P_V2V[i]
                        T_Rate[i] = Rate_V2V[i]
                        self.count_v2v.append(0)
                    else:
                        reward_UL[i] = 0
                        Power[i] = 0
                        T_Rate[i] = 0
                        adm_rest[i]=1
                        done[i] = True


        # Execution step
        for i in range(Num_AGV):
            if self.AGVs[i].demand == 0 or adm_rest[i]==0:
                done[i] = True
                if self.list_nb[i] == 1000000:
                    l_V2I[i] = (self.AGVs[i].rc * self.AGVs[i].S) / h_v2i
                    total_delay = t + l_V2I[i]
                    new_total_delay[i] = l_V2I[i]
                    h_all[i] = h_v2i
                   # Power[i] = P_max * P_V2I[i]
                    if total_delay > (self.AGVs[i].individual_time_limit)*0.5:
                        # if new_total_delay[i] > self.AGVs[i].individual_time_limit:
                        penalty = -0.5*t
                        # penalty = 50*t
                        reward_EXE[i] = 1/total_delay - penalty
                        adm_rest[i] = 1
                    else:
                        # reward_EXE[i] = 1 / new_total_delay[i]
                        reward_EXE[i] = 1 / total_delay
                else:
                    if codeword[i] == 1:
                        l_V2I[i] = (self.AGVs[i].rc * self.AGVs[i].S) / h_v2i
                        total_delay = t + l_V2I[i]
                        new_total_delay[i] = l_V2I[i]
                        h_all[i] = h_v2i
                       # Power[i] = P_max * P_V2I[i]
                        if total_delay > (self.AGVs[i].individual_time_limit)*0.5:
                        #if new_total_delay[i] > self.AGVs[i].individual_time_limit:
                            penalty = -5*t
                            reward_EXE[i] = 1/total_delay - penalty
                            adm_rest[i][i] = 1
                        else:
                            #reward_EXE[i] = 1 / new_total_delay[i]
                            reward_EXE[i] = 1 / total_delay

                    else:
                        h_step[i] = self.AGVs[self.list_nb[i]].h
                        l_V2V[i] = (self.AGVs[i].rc * self.AGVs[i].S) / h_step[i]
                        h_all[i] = h_step[i]
                        total_delay = t + l_V2V[i]
                        new_total_delay[i] = l_V2V[i]
                        # Power[i] = P_max_v2v * P_V2V[i]
                        if total_delay > (self.AGVs[i].individual_time_limit)*0.5:
                        #if new_total_delay[i] > self.AGVs[i].individual_time_limit:
                            penalty = -2*t
                            reward_EXE[i] = 1/total_delay - penalty
                            adm_rest[i][i] = 1
                        else:
                            reward_EXE[i] = 1 / total_delay
                            #reward_EXE[i] = 1 / new_total_delay[i]



        All_time_limit_step = np.zeros((Num_AGV))
        All_demand_step = np.zeros((Num_AGV))


        for i in range(Num_AGV):
            All_time_limit_step[i] = self.AGVs[i].individual_time_limit - new_total_delay[i] - t
            # self.AGVs[i].individual_time_limit = All_time_limit_step[i]
            All_demand_step[i] = self.AGVs[i].demand
            reward[i] = reward_UL[i] + reward_EXE[i]
        self.observation_step = [self.g_V2I_step[:],self.g_V2V_step[:], All_time_limit_step[:], All_demand_step[:]]
        self.observation_step = np.reshape(self.observation_step,(Num_AGV*4))
        next_state = self.observation_step
        return next_state, reward, done, h_all, Power, adm_rest, T_Rate

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





