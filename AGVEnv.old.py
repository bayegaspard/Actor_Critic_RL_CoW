from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import special
import torch
import torch.nn.functional as F
import random
import maddpg_trainer


RESOLUTION = 10 #pixel
SPEED = 10
GRID_WIDTH = 2000/20 # height of highway
P_max = 10 #watt constraint transmit power
Q_code = 1
D_max = 30 # dmax = 30m


def normalize(value, minTar, maxTar,minVal=-1,maxVal =1 ):
 	return ((value- minVal)/(maxVal - minVal))*(maxTar - minTar) + minTar

class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # center of the grids
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)




class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity, acceleration):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.acceleration = acceleration
        self.neighbors = []
        self.destinations = []
        self.h = []
        self.rc = []
        self.y = []



class Environ:

    def __init__(self, down_lane, width, height, n_veh, n_neighbor):
        self.down_lanes = down_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

       # self.demand = {self.rc:30, self.size:2 , self.th :5 } # demand by the SE is 30GHz 2M, and a delay of 5ms
       # self.y = [0, 1]  # control variable 1 for v2v and 0 for V2I
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []

        self.V2I_power_dB = 23  # dBm
        #self.V2V_h_list = [2,6,9,20]
        self.V2V_power_dB_List = [23, 15, 5, -100]
        #self.V2V_multi_action = self.V2V_h_list + self.V2V_power_dB_List # the power levels # continuous distribution (decisision variables)
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.h = 30
        self.rc= random.sample(range(10, 35), 1)
        self.n_RB = n_veh
        self.n_Veh = n_veh
        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        # self.bandwidth = 1500
        self.demand_size = int((4 * 190 + 300) * 8 * 2)  # V2V payload: 1060 Bytes every 100 ms
        # self.demand_size = 20

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_vehicles(self, start_position, start_direction, start_velocity, start_acceleration):
       return self.vehicles.append(Vehicle(start_position, start_direction, start_velocity, start_acceleration))


    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh ))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

    def add_new_vehicles_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd' # velocity: 30 ~ 70 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(30, 70), np.random.randint(1,6))

            # ind = np.random.randint(0, len(self.down_lanes))
            # start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            # start_direction = 'd' # velocity: # velocity: 30 ~ 70 m/s, random
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(30, 70), np.random.randint(1,6))
            #
            # ind = np.random.randint(0, len(self.down_lanes))
            # start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            # start_direction = 'd' # velocity: # velocity: 30 ~ 70 m/s, random
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(30, 70), np.random.randint(1,6))
            #
            # ind = np.random.randint(0, len(self.down_lanes))
            # start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            # start_direction = 'd' # velocity: # velocity: 30 ~ 70 m/s, random
            # self.add_new_vehicles(start_position, start_direction, np.random.randint(30, 70), np.random.randint(1,6))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([(c.velocity*self.time_slow + 0.5*c.acceleration*self.time_slow**2) for c in self.vehicles])
        #self.delta_distance = np.min(self.delta_distance)

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow + 0.5*self.vehicles[i].acceleration*self.time_slow**2
            change_direction = False
            if self.vehicles[i].direction == 'd':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.down_lanes)):

                    if (self.vehicles[i].position[1] <= self.down_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.down_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[1])), self.down_lanes[j]]
                            self.vehicles[i].direction = 'd'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[1] <= self.down_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.down_lanes[j] - self.vehicles[i].position[1])), self.down_lanes[j]]
                                self.vehicles[i].direction = 'd'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance

            # if it comes to an exit
            #if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[1] > self.height): # or (self.vehicles[i].position[0] > self.width)


            i += 1

    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].h = self.h #GHz
            self.vehicles[i].rc = self.rc
            self.vehicles[i].neighbors = []
            self.vehicles[i].y = []
          #  self.vehicles[i]. =
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)
        Acc = [c.acceleration for c in self.vehicles]
        repeated_accl = [(Acc[i:i + 1] * len(Acc)) for i in range(len(Acc))]
        rel_accel = []

        for o in range(len(repeated_accl)):
            # print(data2[i])
            for k, l in zip(range(len(repeated_accl)), range(len(repeated_accl))):
                rel_accel.append(Acc[k] - repeated_accl[o][l])
        new_rel_accel = np.reshape(rel_accel, (4,4))
        nearest_cars = []
        for t in range(len(new_rel_accel)):
            for j in range(len(new_rel_accel)):
                nearest_cars.append((new_rel_accel[t][j] + Distance[t][j]) * 0.5)
        new_nearest_cars = np.reshape(nearest_cars, (4, 4))
        list_neighbr = []
        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(new_nearest_cars[:, i])
            for j in range(self.n_Veh-1):
                if np.logical_and((self.vehicles[j].h > self.vehicles[i].rc[0]), (Distance[j][j] < D_max)):
                    self.vehicles[i].neighbors.append(sort_idx[j + 1])
                    self.vehicles[i].y.append(1)
                    list_neighbr.append(sort_idx[j + 1])
                    destination = self.vehicles[i].neighbors
                    self.vehicles[i].destinations = destination
                else:
                    self.vehicles[i].y.append(0)
        #ourDestination = min(list_neighbr)
        #print(ourDestination)


    def renew_channel(self):
        """ Renew slow fading channel """

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing



    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape))/ math.sqrt(2))

    def Compute_Performance_Reward_Train(self, actions_multi):

        actions = actions_multi[:, :, 0]  # the channel_selection_part
        power_selection = actions_multi[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_multi_action[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links))] = -1 # inactive links will not transmit regardless of selected power levels
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_multi_action[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_multi_action[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_multi_action[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))

        self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0 # eliminate negative demands

        self.individual_time_limit -= self.time_fast

        reward_elements = V2V_Rate/10
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate, reward_elements

    def Compute_Performance_Reward_Test_rand(self, actions_multi):
        """ for random baseline computation """

        actions = actions_multi[:, :, 0]  # the channel_selection_part
        power_selection = actions_multi[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_multi_action[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_random = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference_random))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_multi_action[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_multi_action[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_multi_action[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_random = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference_random))

        self.demand_rand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast

        self.active_links_rand[np.multiply(self.active_links_rand, self.demand_rand <= 0)] = 0 # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate

    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_multi_action[power_selection[i, j]]
                                                                                   - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)


    def act_for_training(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)

        lambdda = 0.
        reward = lambdda * np.sum(V2I_Rate) / (self.n_Veh * 10) + (1 - lambdda) * np.sum(reward_elements) / (self.n_Veh * self.n_neighbor)

        return reward

    def act_for_testing(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate

    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate



        # random baseline
        #self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        #self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        #self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')


    # def display(self):
    #     # Create a blank image
    #     board = np.zeros([GRID_WIDTH+1, GRIG_LENGTH+1, 3])
    #     # Color the snake green
    #     for i in :
    #         board[AGV.grid[1], AGV.grid[0]] = [0, 255, 0]
    #         board[AGV.Dest_y, AGV.Dest_x] = [0, 0, 255]
    #     # Display board
    #     cv2.imshow("Automated warehouse", np.bool(board.repeat(RESOLUTION, 0).repeat(RESOLUTION, 1)))
    #     cv2.waitKey(int(1000/SPEED))

    def reset(self,):
        """
        Reset the initial value
        """
        self.new_random_game()
       # # self.Dest_X = randomdom.sample(range(GRID_WIDTH), self.No_AGV)
       #  #self.Dest_Y = rd.sample(range(GRIG_LENGTH), self.No_AGV)
       #  #self.X = rd.sample(range(GRID_WIDTH), self.No_AGV)
       #  #self.AGVs = [AGV(self.X[i],0, self.Dest_X[i], self.Dest_Y[i]) for i in range(self.No_AGV)]
       #  self.dis = [[np.sqrt(pow(self.gNB_pos[i][0] - self.AGVs[j].grid[0],2) + pow(self.gNB_pos[i][1] - self.AGVs[j].grid[1],2)
       #              + gNB_HEIGHT**2) for j in range(self.No_AGV)] for i in range(self.No_gNB)]
       #  #pow(c/(4*pi*f_c),2)
       #  self.g = [[pow(c/(4*pi*f_c),2)*self.dis[i][j]**(-alpha) for j in range(self.No_AGV)] for i in range(self.No_gNB)]
       #  self.h = [[[self.g[i][j]*nakagami.rvs(2)*np.exp(1j*np.random.rand()*2*pi) for k in range(self.No_ant)]
       #              for j in range(self.No_AGV)]
       #            for i in range(self.No_gNB)]
       #  self.observation = [np.reshape(self.h[i], (self.No_AGV*self.No_ant)) for i in range(self.No_gNB)]
       #  return self.observation
        obersvation = [np.concatenate((self.V2I_channels_with_fastfading, np.reshape(self.V2I_channels_with_fastfading, -1), self.V2V_interference, np.asarray([self.V2I_abs]), self.V2V_abs, self.time_remaining, self.load_remaining, np.asarray([1, 0.02])))]
        return obersvation
    #     # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))

    def step(self, actions):
        reward = np.zeros((self.n_Veh))
        #penalty = np.zeros((self.No_AGV))
        self.ass = np.zeros((self.n_Veh,self.n_Veh))
        power = np.zeros((self.n_Veh, self.n_Veh))
        codeword = np.zeros((self.n_Veh, self.n_Veh))

        #actions = list(np.array(actions).reshape(4, 16))
        print("this is it")
        print(actions)
        for i in range(2*self.n_Veh):
            power[i] = actions[0:]
            codeword[i] = actions[2*self.n_Veh:4*self.n_Veh]

        P2 = [F.softmax(torch.tensor(power[i]), dim=-1) for i in range(self.n_RB)]
        P = [P2[i].numpy()*P_max for i in range(self.n_RB)]

        Code = [np.round(normalize(codeword[i],0,Q_code-1)) for i in range(self.n_Veh)]

        done = [False for i in range(self.n_Veh)]
        ###----------------Calculat positions and channel gains---------------------
        speed = self.add_new_vehicles(self.start_position, self.start_direction, self.start_velocity, self.start_acceleration)
        next_state = 0
        for i in range(self.n_Veh):

            v2irate, v2vrate, rewardele = self.Compute_Performance_Reward_Train(actions)
            done[i] = True
        #self.display()

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

# def plot_durations_ass(env,ass):
#     plt.figure(1)
#     plt.clf()
#     X_gNB = [env.gNB_pos[i][0] for i in range(4)]
#     Y_gNB = [env.gNB_pos[i][1] for i in range(4)]
#     X_AGV = [env.AGVs[i].grid[0] for i in range(env.No_AGV)]
#     Y_AGV = [env.AGVs[i].grid[1] for i in range(env.No_AGV)]
#     plt.plot(X_gNB,Y_gNB, 'g^',  markersize = 12, label = 'gNB')
#     plt.plot(X_AGV,Y_AGV, 'rs',  markersize = 6, label = 'AGV')
#     for i in range(env.No_gNB):
#         for j in range(env.No_AGV):
#             if ass[i][j]:
#                 plt.plot([env.gNB_pos[i][0],env.AGVs[j].grid[0]],[env.gNB_pos[i][1],env.AGVs[j].grid[1]],'--', color= 'royalblue')
#     plt.ylim([0, GRIG_LENGTH])
#     plt.xlim([0, GRID_WIDTH])
#     plt.legend(loc='best', frameon=True)
#     plt.ylabel('Y (m)')
#     plt.xlabel('X (m)')
#     plt.grid(True)
#     plt.pause(1)  # pause a bit so that plots are updated
#     plt.show()

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




