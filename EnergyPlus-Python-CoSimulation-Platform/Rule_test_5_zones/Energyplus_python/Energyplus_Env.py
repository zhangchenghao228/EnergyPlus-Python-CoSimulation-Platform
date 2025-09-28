import Energyplus
from queue import Queue, Full, Empty
import numpy as np
import itertools
import matplotlib.pyplot as plt
import wandb
import time
import os
from distutils.util import strtobool
import argparse
import pandas as pd
import math


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    # 用户可以通过它来控制是否使用Weights and Biases（一个实验跟踪工具）来跟踪实验
    parser.add_argument("--wandb-project-name", type=str, default="TEST",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="shandong111",
                        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    # fmt: on
    return args


name = "EnergyPlus_MADDPG"


class EnergyPlusEnvironment:
    def __init__(self) -> None:
        self.count = 0  # 这个用来记录时间
        self.T_MIN = 18
        self.T_MAX = 24
        self.last_obs_copy = {}
        args = parse_args()
        run_name = f"Circle__{args.exp_name}__{args.seed}__{int(time.time())}"
        # self.wandb = wandb.init(
        #     project=args.wandb_project_name,
        #     entity=args.wandb_entity,
        #     sync_tensorboard=True,
        #     config=vars(args),
        #     name=run_name,
        #     monitor_gym=False,
        #     save_code=True,
        # )
        self.episode = -1
        self.timestep = 0
        self.obs_copy = {}
        self.last_obs = {}  # 这是一个空字典，因为energyplus返回的obs是一个字典
        self.obs_queue: Queue = None  # this queue and the energyplus's queue is the same obj,其实下面这个函数传递的是一个队列
        self.act_queue: Queue = None  # this queue and the energyplus's queue is the same obj，这个注释是什么意思
        self.energyplus: Energyplus.EnergyPlus = Energyplus.EnergyPlus(None, None)

        self.observation_space_size = len(self.energyplus.variables) + len(self.energyplus.meters)
        self.num_agnets = 5
        self.temps_name = ["zone_air_temp_" + str(i + 1) for i in range(self.num_agnets)]
        self.occups_name = ["people_" + str(i + 1) for i in range(self.num_agnets)]
        self.Relative_Humidity_name = ["zone_air_Relative_Humidity_" + str(i + 1) for i in range(self.num_agnets)]
        self.PPD_name = ["PPD_" + str(i + 1) for i in range(self.num_agnets)]
        self.heating_setpoint_name = ["zone_heating_setpoint_" + str(i + 1) for i in range(self.num_agnets)]
        self.cooling_setpoint_name = ["zone_cooling_setpoint_" + str(i + 1) for i in range(self.num_agnets)]
        self.total_energy = 0
        self.total_temp_penalty = [0] * self.num_agnets
        self.total_reward = [0] * self.num_agnets
        self.total_ppd = [0] * self.num_agnets

        # get the indoor/outdoor temperature series
        self.indoor_temps = []
        self.outdoor_temp = []
        # get the setpoint series
        self.setpoints = []
        # get the energy series
        self.energy = []
        # get the occupancy situation
        self.occup_count = []
        self.relative_humidity = []
        self.humditys = []
        self.windspeed = []
        self.winddirection = []
        self.Direct_Solar_Radiation = []
        self.Diffuse_Solar_Radiation = []
        self.PPD = []
        self.heatingpoint = []
        self.coolingpoint = []

        '''仿真时间信息'''
        self.week = 1
        self.day_hour = 0
        '''仿真时间信息'''

        '''VAV能耗信息'''
        self.VAV_energy = 0
        self.VAV_count = 0
        self.total_energy_copy = 0
        '''VAV能耗信息'''
        # self.day_count_hist = []
        self.day_hour_hist = []
        self.week_hist = []
        self.ppd1_hist = []
        self.ppd2_hist = []
        self.ppd3_hist = []
        self.ppd4_hist = []
        self.ppd5_hist = []
        self.ppd6_hist = []
        self.occ1_hist = []
        self.occ2_hist = []
        self.occ3_hist = []
        self.occ4_hist = []
        self.occ5_hist = []
        self.occ6_hist = []
        self.heating1_setpoint_hist = []
        self.heating2_setpoint_hist = []
        self.heating3_setpoint_hist = []
        self.heating4_setpoint_hist = []
        self.heating5_setpoint_hist = []
        self.heating6_setpoint_hist = []
        self.cooling1_setpoint_hist = []
        self.cooling2_setpoint_hist = []
        self.cooling3_setpoint_hist = []
        self.cooling4_setpoint_hist = []
        self.cooling5_setpoint_hist = []
        self.cooling6_setpoint_hist = []
        self.temperature1 = []
        self.temperature2 = []
        self.temperature3 = []
        self.temperature4 = []
        self.temperature5 = []
        self.temperature6 = []
        self.energy_hist = []

        self.day_count = 0

    # return the first observation
    def reset(self, file_suffix="defalut"):
        '''因为程序要进行多段episode，energyplus会重复运行多次，所以要将下面的变量置为0'''
        self.day_count = 0
        self.day_count = self.day_count + 1
        self.VAV_energy = 0
        self.VAV_count = 0

        self.total_temp_penalty = [0] * self.num_agnets
        self.total_energy = 0
        self.total_reward = [0] * self.num_agnets
        self.indoor_temps.clear()
        self.outdoor_temp.clear()
        self.setpoints.clear()
        self.energy.clear()
        self.occup_count.clear()

        self.relative_humidity.clear()
        self.humditys.clear()
        self.windspeed.clear()
        self.winddirection.clear()
        self.Direct_Solar_Radiation.clear()
        self.Diffuse_Solar_Radiation.clear()
        self.PPD.clear()
        self.heatingpoint.clear()
        self.coolingpoint.clear()

        self.energyplus.stop()
        self.episode += 1
        '''因为程序要进行多段episode，energyplus会重复运行多次，所以要将下面的变量置为0'''

        if self.energyplus is not None:
            self.energyplus.stop()

        self.obs_queue = Queue(maxsize=1)  # 这里是一个队列，可以理解为在传递一个地址
        self.act_queue = Queue(maxsize=1)  # 这里是一个队列，可以理解为在传递一个地址

        self.energyplus = Energyplus.EnergyPlus(
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
        )

        self.energyplus.start(file_suffix)

        obs = self.obs_queue.get()  # obs是一个字典

        self.last_obs = obs

        self.VAV_energy = self.VAV_energy + (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000
        self.VAV_count = self.VAV_count + 1
        self.energy_hist.append((self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000)
        '''其实这个地方的值没有什么用处，可能这里只是为了画图，从而存储这些值'''
        self.indoor_temps.append([obs[x] for x in self.temps_name])
        self.occup_count.append([obs[x] for x in self.occups_name])
        self.relative_humidity.append([obs[x] for x in self.Relative_Humidity_name])
        self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])

        '''获取仿真的时间信息'''
        self.week, self.day_hour = self.energyplus.get_time_information()
        '''获取仿真的时间信息'''
        self.day_hour_hist.append(self.day_hour)
        self.week_hist.append(self.week)

        # 下面这是将一个字典变为一个列表，这个列表是step()函数的返回值，里面存放了需要的状态
        to_log = {"zone_air_temp_1": obs["zone_air_temp_1"],
                  "zone_air_temp_2": obs["zone_air_temp_2"],
                  "zone_air_temp_3": obs["zone_air_temp_3"],
                  "zone_air_temp_4": obs["zone_air_temp_4"],
                  "zone_air_temp_5": obs["zone_air_temp_5"],
                  }
        self.count = self.count + 1
        day_count = 0
        zone1_obs = [self.day_hour / 24, self.week / 7,
                     (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 11000000,
                     (obs["zone_air_temp_1"] - 15) / (30 - 15), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 11, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [self.day_hour / 24, self.week / 7,
                     (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 11000000,
                     (obs["zone_air_temp_2"] - 15) / (30 - 15), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 5,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [self.day_hour / 24, self.week / 7,
                     (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 11000000,
                     (obs["zone_air_temp_3"] - 15) / (30 - 15), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 11,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [self.day_hour / 24, self.week / 7,
                     (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 11000000,
                     (obs["zone_air_temp_4"] - 15) / (30 - 15), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 5,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [self.day_hour / 24, self.week / 7,
                     (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 11000000,
                     (obs["zone_air_temp_5"] - 15) / (30 - 15), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 20,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]


        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100])
        self.ppd1_hist.append(obs["PPD_1"])
        self.ppd2_hist.append(obs["PPD_2"])
        self.ppd3_hist.append(obs["PPD_3"])
        self.ppd4_hist.append(obs["PPD_4"])
        self.ppd5_hist.append(obs["PPD_5"])
        self.temperature1.append(obs["zone_air_temp_1"])
        self.temperature2.append(obs["zone_air_temp_2"])
        self.temperature3.append(obs["zone_air_temp_3"])
        self.temperature4.append(obs["zone_air_temp_4"])
        self.temperature5.append(obs["zone_air_temp_5"])
        self.occ1_hist.append(obs["people_1"])
        self.occ2_hist.append(obs["people_2"])
        self.occ3_hist.append(obs["people_3"])
        self.occ4_hist.append(obs["people_4"])
        self.occ5_hist.append(obs["people_5"])
        self.heating1_setpoint_hist.append(obs["zone_heating_setpoint_1"])
        self.heating2_setpoint_hist.append(obs["zone_heating_setpoint_2"])
        self.heating3_setpoint_hist.append(obs["zone_heating_setpoint_3"])
        self.heating4_setpoint_hist.append(obs["zone_heating_setpoint_4"])
        self.heating5_setpoint_hist.append(obs["zone_heating_setpoint_5"])
        self.cooling1_setpoint_hist.append(obs["zone_cooling_setpoint_1"])
        self.cooling2_setpoint_hist.append(obs["zone_cooling_setpoint_2"])
        self.cooling3_setpoint_hist.append(obs["zone_cooling_setpoint_3"])
        self.cooling4_setpoint_hist.append(obs["zone_cooling_setpoint_4"])
        self.cooling5_setpoint_hist.append(obs["zone_cooling_setpoint_5"])
        return obs_test, self.week, self.day_hour, PPD, day_count

    # predict next observation
    def step(self, action):
        self.timestep += 1  # 这个为什么要加1
        self.day_count = self.day_count + 1
        done = False
        if self.energyplus.failed():
            raise RuntimeError(f"E+ failed {self.energyplus.sim_results['exit_code']}")

        if self.energyplus.simulation_complete:
            done = True
            obs = self.last_obs
        else:
            timeout = 3
            try:
                self.VAV_count = self.VAV_count + 1
                self.VAV_energy = self.VAV_energy + (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000
                print('-----------------------------------------------')
                print('VAV_energy: ', self.VAV_energy)
                print('-----------------------------------------------')
                if self.VAV_count == 8926:
                    self.total_energy_copy = self.VAV_energy
                keys_order = [
                    "zone_cooling_setpoint_1",
                    "zone_heating_setpoint_1",
                    "zone_cooling_setpoint_2",
                    "zone_heating_setpoint_2",
                    "zone_cooling_setpoint_3",
                    "zone_heating_setpoint_3",
                    "zone_cooling_setpoint_4",
                    "zone_heating_setpoint_4",
                    "zone_cooling_setpoint_5",
                    "zone_heating_setpoint_5",
                    # "zone_cooling_setpoint_6",
                    # "zone_heating_setpoint_6"
                ]
                zone_setpoint = []
                for key in keys_order:
                    zone_setpoint.append(self.last_obs[key])
                zone_setpoint_array = np.array(zone_setpoint)

                one_d_list = list(itertools.chain(*action))
                one_d_list = np.array(one_d_list)

                # 将神经网络的输出值映射到15-30
                action_result = one_d_list

                action_result = action_result.tolist()

                self.setpoints.append(action_result)  # 将神经网络输出的-1至1的数值转换为19-24之间的数值
                '''这里相当于是在传递神经网络输出的索引值'''
                start_time = time.time()
                self.act_queue.put(action_result, timeout=timeout)  # timeout指定此操作等待的时间，这个接收的是一个1维的numpy数组
                self.last_obs_copy = self.last_obs
                self.obs_copy = self.last_obs
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                end_time = time.time()
                print('env_time: ', end_time - start_time)
            except(Full, Empty):
                done = True
                self.obs_copy = self.last_obs
                obs = self.last_obs
                self.last_obs_copy = self.last_obs
            '''上面这个函数用于捕获异常'''
        reward, reward_local, reward_global = self.get_reward  # 这是一个标量
        if 8 <= self.day_hour < 21 and obs["people_1"] != 0:
            PPD1 = obs["PPD_1"]
        else:
            PPD1 = 0
        if 8 <= self.day_hour < 21 and obs["people_2"] != 0:
            PPD2 = obs["PPD_2"]
        else:
            PPD2 = 0
        if 8 <= self.day_hour < 21 and obs["people_3"] != 0:
            PPD3 = obs["PPD_3"]
        else:
            PPD3 = 0
        if 8 <= self.day_hour < 21 and obs["people_4"] != 0:
            PPD4 = obs["PPD_4"]
        else:
            PPD4 = 0
        if 8 <= self.day_hour < 21 and obs["people_5"] != 0:
            PPD5 = obs["PPD_5"]
        else:
            PPD5 = 0
        to_log = {"zone_cooling_setpoint_1": obs["zone_cooling_setpoint_1"],
                  "zone_cooling_setpoint_2": obs["zone_cooling_setpoint_2"],
                  "zone_cooling_setpoint_3": obs["zone_cooling_setpoint_3"],
                  "zone_cooling_setpoint_4": obs["zone_cooling_setpoint_4"],
                  "zone_cooling_setpoint_5": obs["zone_cooling_setpoint_5"],
                  "zone_heating_setpoint_1": obs["zone_heating_setpoint_1"],
                  "zone_heating_setpoint_2": obs["zone_heating_setpoint_2"],
                  "zone_heating_setpoint_3": obs["zone_heating_setpoint_3"],
                  "zone_heating_setpoint_4": obs["zone_heating_setpoint_4"],
                  "zone_heating_setpoint_5": obs["zone_heating_setpoint_5"],
                  "zone_air_temp_1": obs["zone_air_temp_1"],
                  "zone_air_temp_2": obs["zone_air_temp_2"],
                  "zone_air_temp_3": obs["zone_air_temp_3"],
                  "zone_air_temp_4": obs["zone_air_temp_4"],
                  "zone_air_temp_5": obs["zone_air_temp_5"],
                  # "zone_air_temp_6": obs["zone_air_temp_6"],
                  "VAV_ENERGY_total": self.total_energy_copy,
                  "VAV_ENERGY": self.VAV_energy,
                  "VAV_ENERGY_TEMP": (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000,
                  "hour": self.day_hour,
                  "PPD1": PPD1,
                  "PPD2": PPD2,
                  "PPD3": PPD3,
                  "PPD4": PPD4,
                  "PPD5": PPD5,
                  }
        self.count = self.count + 1
        # self.wandb.log(to_log, step=self.count)
        obs_vec = np.array(list(obs.values()))  # 这是一个列表
        self.week, self.day_hour = self.energyplus.get_time_information()
        self.day_hour_hist.append(self.day_hour)
        self.week_hist.append(self.week)
        day_count = self.day_count % 96
        zone1_obs = [self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] + 15) / (21 + 15),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, obs["elec_hvac"] / 56000000,
                     (obs["zone_air_temp_1"] - 10) / (30 - 10), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 6, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] + 15) / (21 + 15),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, obs["elec_hvac"] / 56000000,
                     (obs["zone_air_temp_2"] - 10) / (30 - 10), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 13,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] + 15) / (21 + 15),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, obs["elec_hvac"] / 56000000,
                     (obs["zone_air_temp_3"] - 10) / (30 - 10), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 5,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] + 15) / (21 + 15),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, obs["elec_hvac"] / 56000000,
                     (obs["zone_air_temp_4"] - 10) / (30 - 10), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 6,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] + 15) / (21 + 15),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, obs["elec_hvac"] / 56000000,
                     (obs["zone_air_temp_5"] - 10) / (30 - 10), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 13,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]

        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100])
        # 更新数据的过程
        self.ppd1_hist.append(obs["PPD_1"])
        self.ppd2_hist.append(obs["PPD_2"])
        self.ppd3_hist.append(obs["PPD_3"])
        self.ppd4_hist.append(obs["PPD_4"])
        self.ppd5_hist.append(obs["PPD_5"])
        self.temperature1.append(obs["zone_air_temp_1"])
        self.temperature2.append(obs["zone_air_temp_2"])
        self.temperature3.append(obs["zone_air_temp_3"])
        self.temperature4.append(obs["zone_air_temp_4"])
        self.temperature5.append(obs["zone_air_temp_5"])
        # self.temperature6.append(obs["zone_air_temp_6"])
        self.occ1_hist.append(obs["people_1"])
        self.occ2_hist.append(obs["people_2"])
        self.occ3_hist.append(obs["people_3"])
        self.occ4_hist.append(obs["people_4"])
        self.occ5_hist.append(obs["people_5"])
        self.heating1_setpoint_hist.append(obs["zone_heating_setpoint_1"])
        self.heating2_setpoint_hist.append(obs["zone_heating_setpoint_2"])
        self.heating3_setpoint_hist.append(obs["zone_heating_setpoint_3"])
        self.heating4_setpoint_hist.append(obs["zone_heating_setpoint_4"])
        self.heating5_setpoint_hist.append(obs["zone_heating_setpoint_5"])
        # self.heating6_setpoint_hist.append(obs["zone_heating_setpoint_6"])
        self.cooling1_setpoint_hist.append(obs["zone_cooling_setpoint_1"])
        self.cooling2_setpoint_hist.append(obs["zone_cooling_setpoint_2"])
        self.cooling3_setpoint_hist.append(obs["zone_cooling_setpoint_3"])
        self.cooling4_setpoint_hist.append(obs["zone_cooling_setpoint_4"])
        self.cooling5_setpoint_hist.append(obs["zone_cooling_setpoint_5"])
        self.energy_hist.append((self.last_obs["elec_hvac"] + self.last_obs["elec_heat"])/3600000)


        if done == True:
            # 将所有数据合并成一个字典，方便生成 DataFrame
            data = {
                "hour": self.day_hour_hist,
                "Rule_PPD_1": self.ppd1_hist,
                "Rule_PPD_2": self.ppd2_hist,
                "Rule_PPD_3": self.ppd3_hist,
                "Rule_PPD_4": self.ppd4_hist,
                "Rule_PPD_5": self.ppd5_hist,
                "Rule_People_1": self.occ1_hist,
                "Rule_People_2": self.occ2_hist,
                "Rule_People_3": self.occ3_hist,
                "Rule_People_4": self.occ4_hist,
                "Rule_People_5": self.occ5_hist,
                "Rule_temperature1": self.temperature1,
                "Rule_temperature2": self.temperature2,
                "Rule_temperature3": self.temperature3,
                "Rule_temperature4": self.temperature4,
                "Rule_temperature5": self.temperature5,
                # "temperature6": self.temperature6,
                "Rule_heating1": self.heating1_setpoint_hist,
                "Rule_heating2": self.heating2_setpoint_hist,
                "Rule_heating3": self.heating3_setpoint_hist,
                "Rule_heating4": self.heating4_setpoint_hist,
                "Rule_heating5": self.heating5_setpoint_hist,
                # "heating6": self.heating6_setpoint_hist,
                "Rule_cooling1": self.cooling1_setpoint_hist,
                "Rule_cooling2": self.cooling2_setpoint_hist,
                "Rule_cooling3": self.cooling3_setpoint_hist,
                "Rule_cooling4": self.cooling4_setpoint_hist,
                "Rule_cooling5": self.cooling5_setpoint_hist,
                # "cooling6": self.cooling6_setpoint_hist,
                "Rule_energy": self.energy_hist,
            }

            # 创建 DataFrame 并带上表头
            df = pd.DataFrame(data)
            # 创建 ExcelWriter
            with pd.ExcelWriter("Rule_data.xlsx") as writer:
                # 保存完整数据到一个 sheet
                df.to_excel(writer, sheet_name="All_data", index=False)

                # 按每96行分割数据，并保存到不同的 sheets
                chunk_size = 96
                num_chunks = math.ceil(len(df) / chunk_size)
                for i in range(num_chunks):
                    start_row = i * chunk_size
                    end_row = start_row + chunk_size
                    chunk_df = df.iloc[start_row:end_row]
                    sheet_name = f'Day_{i + 1}'
                    chunk_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"数据已保存到 MATD3_data.xlsx，其中包含 'All_Data' 和 {num_chunks} 个分割的 sheets。")

        return obs_test, reward_local, reward_global, done, self.week, self.day_hour, PPD, day_count

    '''下面这个@property不能删除'''

    @property
    def get_reward(self):
        PPD_thres = 0.2
        w_e = 0.6
        w_c = 0.4
        reward = []  # 存放5个agent的奖励
        reward_local = []
        reward_global = []
        # according to the meters and variables to compute
        obs = self.last_obs  # 这个是在获取状态，这个状态是一个字典
        '''这个函数用于判断每个区域是否有人'''
        occups_vals = []
        for occup in self.occups_name:
            occups_vals.append(obs[occup])
        '''这个函数取得每个区域的PPD值'''
        PPD_vals = []
        for PPD in self.PPD_name:
            PPD_vals.append(obs[PPD] / 100)
        '''这个函数取得每个区域的PPD值'''
        '''计算c(t)值'''
        c_result = []
        for PPD_copy in PPD_vals:
            if PPD_copy > PPD_thres:
                c_result.append(1)
            else:
                # ppd_temp = self.calculate_reward(PPD_copy)
                c_result.append(PPD_copy)
        '''这个值对于5个智能体都是一样的，是不是应该寻找一个替代值'''
        # TODO find a good function to evaluate the temperature reward
        energy = (obs["elec_hvac"] + obs["elec_heat"]) / 20000000 / 2.5  # 将电能消耗量从瓦特秒转换为千瓦时,这个值可能是在1以下
        for o, c in zip(occups_vals, c_result):
            if o == 0:
                r = -w_e * energy
                reward_local_temp = 0
                reward_global_temp = -w_e * energy
            else:
                r = -w_e * energy - w_c * c
                reward_local_temp = - w_c * c
                reward_global_temp = -w_e * energy
            reward.append(r)
            reward_local.append(reward_local_temp)
            reward_global.append(reward_global_temp)
        return reward, reward_local, reward_global

    def close(self):
        if self.energyplus is not None:
            self.energyplus.stop()

    def render(self):
        # get the indoor/outdoor temperature series
        zone_temp = []
        for i in range(5):
            zone_temp.append(np.array(self.indoor_temps)[:, i])

        # get occupancy
        zone_occupy = []
        for i in range(5):
            zone_occupy.append(np.array(self.occup_count)[:, i])
        # get the setpoint series
        sp_series = []
        for i in range(0, 10, 2):
            sp_series.append(np.array(self.setpoints)[:, i])
        # get the energy series
        x = range(len(self.setpoints))

        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("temperature (℃)")
            plt.plot(x, zone_temp[i], label=f"zone_{i + 1}_temperature")
        plt.legend()
        plt.show()

        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("setpoint (℃)")
            plt.plot(x, sp_series[i], label=f"zone_{i + 1}_setpoint")
        plt.legend()
        plt.show()
        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("occupancy")
            plt.plot(x, zone_occupy[i], label=f"zone_{i + 1}_people_occupant_count ")
        plt.legend()
        plt.show()

        plt.plot(x, self.energy)
        plt.title("energy cost")
        plt.xlabel("timestep")
        plt.ylabel("energy cost (kwh)")
        plt.show()

        plt.plot(x, self.outdoor_temp)
        plt.title("outdoor temperature")
        plt.xlabel("timestep")
        plt.ylabel("temperature (℃)")
        plt.show()
