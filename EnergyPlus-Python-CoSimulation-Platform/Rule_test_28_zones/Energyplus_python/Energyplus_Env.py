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
        self.num_agnets =27
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
        self.ppd7_hist = []
        self.ppd8_hist = []
        self.ppd9_hist = []
        self.ppd10_hist = []
        self.ppd11_hist = []
        self.ppd12_hist = []
        self.ppd13_hist = []
        self.ppd14_hist = []
        self.ppd15_hist = []
        self.ppd16_hist = []
        self.ppd17_hist = []
        self.ppd18_hist = []
        self.ppd19_hist = []
        self.ppd20_hist = []
        self.ppd21_hist = []
        self.ppd22_hist = []
        self.ppd23_hist = []
        self.ppd24_hist = []
        self.ppd25_hist = []
        self.ppd26_hist = []
        self.ppd27_hist = []
        self.occ1_hist = []
        self.occ2_hist = []
        self.occ3_hist = []
        self.occ4_hist = []
        self.occ5_hist = []
        self.occ6_hist = []
        self.occ7_hist = []
        self.occ8_hist = []
        self.occ9_hist = []
        self.occ10_hist = []
        self.occ11_hist = []
        self.occ12_hist = []
        self.occ13_hist = []
        self.occ14_hist = []
        self.occ15_hist = []
        self.occ16_hist = []
        self.occ17_hist = []
        self.occ18_hist = []
        self.occ19_hist = []
        self.occ20_hist = []
        self.occ21_hist = []
        self.occ22_hist = []
        self.occ23_hist = []
        self.occ24_hist = []
        self.occ25_hist = []
        self.occ26_hist = []
        self.occ27_hist = []

        self.heating1_setpoint_hist = []
        self.heating2_setpoint_hist = []
        self.heating3_setpoint_hist = []
        self.heating4_setpoint_hist = []
        self.heating5_setpoint_hist = []
        self.heating6_setpoint_hist = []
        self.heating7_setpoint_hist = []
        self.heating8_setpoint_hist = []
        self.heating9_setpoint_hist = []
        self.heating10_setpoint_hist = []
        self.heating11_setpoint_hist = []
        self.heating12_setpoint_hist = []
        self.heating13_setpoint_hist = []
        self.heating14_setpoint_hist = []
        self.heating15_setpoint_hist = []
        self.heating16_setpoint_hist = []
        self.heating17_setpoint_hist = []
        self.heating18_setpoint_hist = []
        self.heating19_setpoint_hist = []
        self.heating20_setpoint_hist = []
        self.heating21_setpoint_hist = []
        self.heating22_setpoint_hist = []
        self.heating23_setpoint_hist = []
        self.heating24_setpoint_hist = []
        self.heating25_setpoint_hist = []
        self.heating26_setpoint_hist = []
        self.heating27_setpoint_hist = []



        self.cooling1_setpoint_hist = []
        self.cooling2_setpoint_hist = []
        self.cooling3_setpoint_hist = []
        self.cooling4_setpoint_hist = []
        self.cooling5_setpoint_hist = []
        self.cooling6_setpoint_hist = []
        self.cooling7_setpoint_hist = []
        self.cooling8_setpoint_hist = []
        self.cooling9_setpoint_hist = []
        self.cooling10_setpoint_hist = []
        self.cooling11_setpoint_hist = []
        self.cooling12_setpoint_hist = []
        self.cooling13_setpoint_hist = []
        self.cooling14_setpoint_hist = []
        self.cooling15_setpoint_hist = []
        self.cooling16_setpoint_hist = []
        self.cooling17_setpoint_hist = []
        self.cooling18_setpoint_hist = []
        self.cooling19_setpoint_hist = []
        self.cooling20_setpoint_hist = []
        self.cooling21_setpoint_hist = []
        self.cooling22_setpoint_hist = []
        self.cooling23_setpoint_hist = []
        self.cooling24_setpoint_hist = []
        self.cooling25_setpoint_hist = []
        self.cooling26_setpoint_hist = []
        self.cooling27_setpoint_hist = []



        self.temperature1 = []
        self.temperature2 = []
        self.temperature3 = []
        self.temperature4 = []
        self.temperature5 = []
        self.temperature6 = []
        self.temperature7 = []
        self.temperature8 = []
        self.temperature9 = []
        self.temperature10 = []
        self.temperature11 = []
        self.temperature12 = []
        self.temperature13 = []
        self.temperature14 = []
        self.temperature15 = []
        self.temperature16 = []
        self.temperature17 = []
        self.temperature18 = []
        self.temperature19 = []
        self.temperature20 = []
        self.temperature21 = []
        self.temperature22 = []
        self.temperature23 = []
        self.temperature24 = []
        self.temperature25 = []
        self.temperature26 = []
        self.temperature27 = []

        self.reward_hist = []
        self.day_count = 0
        self.day_count_hist = []
        self.energy_hist = []
        self.global_hist = []

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
        zone1_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_1"] - 15) / (30 - 15), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 11, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_2"] - 15) / (30 - 15), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 5,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_3"] - 15) / (30 - 15), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 11,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_4"] - 15) / (30 - 15), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 5,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_5"] - 15) / (30 - 15), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 20,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]
        zone6_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_6"] - 15) / (30 - 15), (obs["zone_heating_setpoint_6"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_6"] - 15) / (32 - 15),
                     obs["people_6"] / 20,
                     obs["zone_air_Relative_Humidity_6"] / 100
                     ]
        zone7_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_7"] - 15) / (30 - 15), (obs["zone_heating_setpoint_7"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_7"] - 15) / (32 - 15),
                     obs["people_7"] / 20,
                     obs["zone_air_Relative_Humidity_7"] / 100
                     ]
        zone8_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_8"] - 15) / (30 - 15), (obs["zone_heating_setpoint_8"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_8"] - 15) / (32 - 15),
                     obs["people_8"] / 20,
                     obs["zone_air_Relative_Humidity_8"] / 100
                     ]
        zone9_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_9"] - 15) / (30 - 15), (obs["zone_heating_setpoint_9"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_9"] - 15) / (32 - 15),
                     obs["people_9"] / 20,
                     obs["zone_air_Relative_Humidity_9"] / 100
                     ]
        zone10_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_10"] - 15) / (30 - 15), (obs["zone_heating_setpoint_10"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_10"] - 15) / (32 - 15),
                     obs["people_10"] / 20,
                     obs["zone_air_Relative_Humidity_10"] / 100
                     ]
        
        zone11_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_11"] - 15) / (30 - 15), (obs["zone_heating_setpoint_11"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_11"] - 15) / (32 - 15),
                     obs["people_11"] / 11, obs["zone_air_Relative_Humidity_11"] / 100
                     ]
        zone12_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_12"] - 15) / (30 - 15), (obs["zone_heating_setpoint_12"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_12"] - 15) / (32 - 15),
                     obs["people_12"] / 5,
                     obs["zone_air_Relative_Humidity_12"] / 100
                     ]
        zone13_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_13"] - 15) / (30 - 15), (obs["zone_heating_setpoint_13"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_13"] - 15) / (32 - 15),
                     obs["people_13"] / 11,
                     obs["zone_air_Relative_Humidity_13"] / 100
                     ]
        zone14_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_14"] - 15) / (30 - 15), (obs["zone_heating_setpoint_14"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_14"] - 15) / (32 - 15),
                     obs["people_14"] / 5,
                     obs["zone_air_Relative_Humidity_14"] / 100
                     ]
        zone15_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_15"] - 15) / (30 - 15), (obs["zone_heating_setpoint_15"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_15"] - 15) / (32 - 15),
                     obs["people_15"] / 20,
                     obs["zone_air_Relative_Humidity_15"] / 100
                     ]
        zone16_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_16"] - 15) / (30 - 15), (obs["zone_heating_setpoint_16"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_16"] - 15) / (32 - 15),
                     obs["people_16"] / 20,
                     obs["zone_air_Relative_Humidity_16"] / 100
                     ]
        zone17_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_17"] - 15) / (30 - 15), (obs["zone_heating_setpoint_17"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_17"] - 15) / (32 - 15),
                     obs["people_17"] / 20,
                     obs["zone_air_Relative_Humidity_17"] / 100
                     ]
        zone18_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_18"] - 15) / (30 - 15), (obs["zone_heating_setpoint_18"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_18"] - 15) / (32 - 15),
                     obs["people_18"] / 20,
                     obs["zone_air_Relative_Humidity_18"] / 100
                     ]
        zone19_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_19"] - 15) / (30 - 15), (obs["zone_heating_setpoint_19"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_19"] - 15) / (32 - 15),
                     obs["people_19"] / 20,
                     obs["zone_air_Relative_Humidity_19"] / 100
                     ]
        zone20_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_20"] - 15) / (30 - 15), (obs["zone_heating_setpoint_20"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_20"] - 15) / (32 - 15),
                     obs["people_20"] / 20,
                     obs["zone_air_Relative_Humidity_20"] / 100
                     ]
        

        zone21_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_21"] - 15) / (30 - 15), (obs["zone_heating_setpoint_21"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_21"] - 15) / (32 - 15),
                     obs["people_21"] / 11, obs["zone_air_Relative_Humidity_21"] / 100
                     ]
        zone22_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_22"] - 15) / (30 - 15), (obs["zone_heating_setpoint_22"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_22"] - 15) / (32 - 15),
                     obs["people_22"] / 5,
                     obs["zone_air_Relative_Humidity_22"] / 100
                     ]
        zone23_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_23"] - 15) / (30 - 15), (obs["zone_heating_setpoint_23"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_23"] - 15) / (32 - 15),
                     obs["people_23"] / 11,
                     obs["zone_air_Relative_Humidity_23"] / 100
                     ]
        zone24_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_24"] - 15) / (30 - 15), (obs["zone_heating_setpoint_24"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_24"] - 15) / (32 - 15),
                     obs["people_24"] / 5,
                     obs["zone_air_Relative_Humidity_24"] / 100
                     ]
        zone25_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_25"] - 15) / (30 - 15), (obs["zone_heating_setpoint_25"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_25"] - 15) / (32 - 15),
                     obs["people_25"] / 20,
                     obs["zone_air_Relative_Humidity_25"] / 100
                     ]
        zone26_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_26"] - 15) / (30 - 15), (obs["zone_heating_setpoint_26"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_26"] - 15) / (32 - 15),
                     obs["people_26"] / 20,
                     obs["zone_air_Relative_Humidity_26"] / 100
                     ]
        zone27_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_27"] - 15) / (30 - 15), (obs["zone_heating_setpoint_27"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_27"] - 15) / (32 - 15),
                     obs["people_27"] / 20,
                     obs["zone_air_Relative_Humidity_27"] / 100
                     ]


        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs, zone6_obs, zone7_obs, zone8_obs, zone9_obs, zone10_obs,
                             zone11_obs, zone12_obs, zone13_obs, zone14_obs, zone15_obs, zone16_obs, zone17_obs, zone18_obs, zone19_obs, zone20_obs,
                             zone21_obs, zone22_obs, zone23_obs, zone24_obs, zone25_obs, zone26_obs, zone27_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100, obs["PPD_6"] / 100,
             obs["PPD_7"] / 100, obs["PPD_8"] / 100, obs["PPD_9"] / 100, obs["PPD_10"] / 100, obs["PPD_11"] / 100, obs["PPD_12"] / 100, 
             obs["PPD_13"] / 100, obs["PPD_14"] / 100, obs["PPD_15"] / 100, obs["PPD_16"] / 100, obs["PPD_17"] / 100, obs["PPD_18"] / 100, 
             obs["PPD_19"] / 100, obs["PPD_20"] / 100, obs["PPD_21"] / 100, obs["PPD_22"] / 100, obs["PPD_23"] / 100, obs["PPD_24"] / 100, 
             obs["PPD_25"] / 100, obs["PPD_26"] / 100, obs["PPD_27"] / 100])

        self.ppd1_hist.append(obs["PPD_1"])
        self.ppd2_hist.append(obs["PPD_2"])
        self.ppd3_hist.append(obs["PPD_3"])
        self.ppd4_hist.append(obs["PPD_4"])
        self.ppd5_hist.append(obs["PPD_5"])
        self.ppd6_hist.append(obs["PPD_6"])
        self.ppd7_hist.append(obs["PPD_7"])
        self.ppd8_hist.append(obs["PPD_8"])
        self.ppd9_hist.append(obs["PPD_9"])
        self.ppd10_hist.append(obs["PPD_10"])
        self.ppd11_hist.append(obs["PPD_11"])
        self.ppd12_hist.append(obs["PPD_12"])
        self.ppd13_hist.append(obs["PPD_13"])
        self.ppd14_hist.append(obs["PPD_14"])
        self.ppd15_hist.append(obs["PPD_15"])
        self.ppd16_hist.append(obs["PPD_16"])
        self.ppd17_hist.append(obs["PPD_17"])
        self.ppd18_hist.append(obs["PPD_18"])
        self.ppd19_hist.append(obs["PPD_19"])
        self.ppd20_hist.append(obs["PPD_20"])
        self.ppd21_hist.append(obs["PPD_21"])
        self.ppd22_hist.append(obs["PPD_22"])
        self.ppd23_hist.append(obs["PPD_23"])
        self.ppd24_hist.append(obs["PPD_24"])
        self.ppd25_hist.append(obs["PPD_25"])
        self.ppd26_hist.append(obs["PPD_26"])
        self.ppd27_hist.append(obs["PPD_27"])


        self.temperature1.append(obs["zone_air_temp_1"])
        self.temperature2.append(obs["zone_air_temp_2"])
        self.temperature3.append(obs["zone_air_temp_3"])
        self.temperature4.append(obs["zone_air_temp_4"])
        self.temperature5.append(obs["zone_air_temp_5"])
        self.temperature6.append(obs["zone_air_temp_6"])
        self.temperature7.append(obs["zone_air_temp_7"])
        self.temperature8.append(obs["zone_air_temp_8"])
        self.temperature9.append(obs["zone_air_temp_9"])
        self.temperature10.append(obs["zone_air_temp_10"])
        self.temperature11.append(obs["zone_air_temp_11"])
        self.temperature12.append(obs["zone_air_temp_12"])
        self.temperature13.append(obs["zone_air_temp_13"])
        self.temperature14.append(obs["zone_air_temp_14"])
        self.temperature15.append(obs["zone_air_temp_15"])
        self.temperature16.append(obs["zone_air_temp_16"])
        self.temperature17.append(obs["zone_air_temp_17"])
        self.temperature18.append(obs["zone_air_temp_18"])
        self.temperature19.append(obs["zone_air_temp_19"])
        self.temperature20.append(obs["zone_air_temp_20"])
        self.temperature21.append(obs["zone_air_temp_21"])
        self.temperature22.append(obs["zone_air_temp_22"])
        self.temperature23.append(obs["zone_air_temp_23"])
        self.temperature24.append(obs["zone_air_temp_24"])
        self.temperature25.append(obs["zone_air_temp_25"])
        self.temperature26.append(obs["zone_air_temp_26"])
        self.temperature27.append(obs["zone_air_temp_27"])


        self.occ1_hist.append(obs["people_1"])
        self.occ2_hist.append(obs["people_2"])
        self.occ3_hist.append(obs["people_3"])
        self.occ4_hist.append(obs["people_4"])
        self.occ5_hist.append(obs["people_5"])
        self.occ6_hist.append(obs["people_6"])
        self.occ7_hist.append(obs["people_7"])
        self.occ8_hist.append(obs["people_8"])
        self.occ9_hist.append(obs["people_9"])
        self.occ10_hist.append(obs["people_10"])
        self.occ11_hist.append(obs["people_11"])
        self.occ12_hist.append(obs["people_12"])
        self.occ13_hist.append(obs["people_13"])
        self.occ14_hist.append(obs["people_14"])
        self.occ15_hist.append(obs["people_15"])
        self.occ16_hist.append(obs["people_16"])
        self.occ17_hist.append(obs["people_17"])
        self.occ18_hist.append(obs["people_18"])
        self.occ19_hist.append(obs["people_19"])
        self.occ20_hist.append(obs["people_20"])
        self.occ21_hist.append(obs["people_21"])
        self.occ22_hist.append(obs["people_22"])
        self.occ23_hist.append(obs["people_23"])
        self.occ24_hist.append(obs["people_24"])
        self.occ25_hist.append(obs["people_25"])
        self.occ26_hist.append(obs["people_26"])
        self.occ27_hist.append(obs["people_27"])



        self.heating1_setpoint_hist.append(obs["zone_heating_setpoint_1"])
        self.heating2_setpoint_hist.append(obs["zone_heating_setpoint_2"])
        self.heating3_setpoint_hist.append(obs["zone_heating_setpoint_3"])
        self.heating4_setpoint_hist.append(obs["zone_heating_setpoint_4"])
        self.heating5_setpoint_hist.append(obs["zone_heating_setpoint_5"])
        self.heating6_setpoint_hist.append(obs["zone_heating_setpoint_6"])
        self.heating7_setpoint_hist.append(obs["zone_heating_setpoint_7"])
        self.heating8_setpoint_hist.append(obs["zone_heating_setpoint_8"])
        self.heating9_setpoint_hist.append(obs["zone_heating_setpoint_9"])
        self.heating10_setpoint_hist.append(obs["zone_heating_setpoint_10"])
        self.heating11_setpoint_hist.append(obs["zone_heating_setpoint_11"])
        self.heating12_setpoint_hist.append(obs["zone_heating_setpoint_12"])
        self.heating13_setpoint_hist.append(obs["zone_heating_setpoint_13"])
        self.heating14_setpoint_hist.append(obs["zone_heating_setpoint_14"])
        self.heating15_setpoint_hist.append(obs["zone_heating_setpoint_15"])
        self.heating16_setpoint_hist.append(obs["zone_heating_setpoint_16"])
        self.heating17_setpoint_hist.append(obs["zone_heating_setpoint_17"])
        self.heating18_setpoint_hist.append(obs["zone_heating_setpoint_18"])
        self.heating19_setpoint_hist.append(obs["zone_heating_setpoint_19"])
        self.heating20_setpoint_hist.append(obs["zone_heating_setpoint_20"])
        self.heating21_setpoint_hist.append(obs["zone_heating_setpoint_21"])
        self.heating22_setpoint_hist.append(obs["zone_heating_setpoint_22"])
        self.heating23_setpoint_hist.append(obs["zone_heating_setpoint_23"])
        self.heating24_setpoint_hist.append(obs["zone_heating_setpoint_24"])
        self.heating25_setpoint_hist.append(obs["zone_heating_setpoint_25"])
        self.heating26_setpoint_hist.append(obs["zone_heating_setpoint_26"])
        self.heating27_setpoint_hist.append(obs["zone_heating_setpoint_27"])






        self.cooling1_setpoint_hist.append(obs["zone_cooling_setpoint_1"])
        self.cooling2_setpoint_hist.append(obs["zone_cooling_setpoint_2"])
        self.cooling3_setpoint_hist.append(obs["zone_cooling_setpoint_3"])
        self.cooling4_setpoint_hist.append(obs["zone_cooling_setpoint_4"])
        self.cooling5_setpoint_hist.append(obs["zone_cooling_setpoint_5"])
        self.cooling6_setpoint_hist.append(obs["zone_cooling_setpoint_6"])
        self.cooling7_setpoint_hist.append(obs["zone_cooling_setpoint_7"])
        self.cooling8_setpoint_hist.append(obs["zone_cooling_setpoint_8"])
        self.cooling9_setpoint_hist.append(obs["zone_cooling_setpoint_9"])
        self.cooling10_setpoint_hist.append(obs["zone_cooling_setpoint_10"])
        self.cooling11_setpoint_hist.append(obs["zone_cooling_setpoint_11"])
        self.cooling12_setpoint_hist.append(obs["zone_cooling_setpoint_12"])
        self.cooling13_setpoint_hist.append(obs["zone_cooling_setpoint_13"])
        self.cooling14_setpoint_hist.append(obs["zone_cooling_setpoint_14"])
        self.cooling15_setpoint_hist.append(obs["zone_cooling_setpoint_15"])
        self.cooling16_setpoint_hist.append(obs["zone_cooling_setpoint_16"])
        self.cooling17_setpoint_hist.append(obs["zone_cooling_setpoint_17"])
        self.cooling18_setpoint_hist.append(obs["zone_cooling_setpoint_18"])
        self.cooling19_setpoint_hist.append(obs["zone_cooling_setpoint_19"])
        self.cooling20_setpoint_hist.append(obs["zone_cooling_setpoint_20"])
        self.cooling21_setpoint_hist.append(obs["zone_cooling_setpoint_21"])
        self.cooling22_setpoint_hist.append(obs["zone_cooling_setpoint_22"])
        self.cooling23_setpoint_hist.append(obs["zone_cooling_setpoint_23"])
        self.cooling24_setpoint_hist.append(obs["zone_cooling_setpoint_24"])
        self.cooling25_setpoint_hist.append(obs["zone_cooling_setpoint_25"])
        self.cooling26_setpoint_hist.append(obs["zone_cooling_setpoint_26"])
        self.cooling27_setpoint_hist.append(obs["zone_cooling_setpoint_27"])





        self.day_count_hist.append(self.day_count % 96)
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
                    "zone_cooling_setpoint_6",
                    "zone_heating_setpoint_6",
                    "zone_cooling_setpoint_7",
                    "zone_heating_setpoint_7",
                    "zone_cooling_setpoint_8",
                    "zone_heating_setpoint_8",
                    "zone_cooling_setpoint_9",
                    "zone_heating_setpoint_9",
                    "zone_cooling_setpoint_10",
                    "zone_heating_setpoint_10",

                    "zone_cooling_setpoint_11",
                    "zone_heating_setpoint_11",
                    "zone_cooling_setpoint_12",
                    "zone_heating_setpoint_12",
                    "zone_cooling_setpoint_13",
                    "zone_heating_setpoint_13",
                    "zone_cooling_setpoint_14",
                    "zone_heating_setpoint_14",
                    "zone_cooling_setpoint_15",
                    "zone_heating_setpoint_15",
                    "zone_cooling_setpoint_16",
                    "zone_heating_setpoint_16",
                    "zone_cooling_setpoint_17",
                    "zone_heating_setpoint_17",
                    "zone_cooling_setpoint_18",
                    "zone_heating_setpoint_18",
                    "zone_cooling_setpoint_19",
                    "zone_heating_setpoint_19",
                    "zone_cooling_setpoint_20",
                    "zone_heating_setpoint_20",


                    "zone_cooling_setpoint_21",
                    "zone_heating_setpoint_21",
                    "zone_cooling_setpoint_22",
                    "zone_heating_setpoint_22",
                    "zone_cooling_setpoint_23",
                    "zone_heating_setpoint_23",
                    "zone_cooling_setpoint_24",
                    "zone_heating_setpoint_24",
                    "zone_cooling_setpoint_25",
                    "zone_heating_setpoint_25",
                    "zone_cooling_setpoint_26",
                    "zone_heating_setpoint_26",
                    "zone_cooling_setpoint_27",
                    "zone_heating_setpoint_27"
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
        # if 8 <= self.day_hour < 21 and obs["people_1"] != 0:
        #     PPD1 = obs["PPD_1"]
        # else:
        #     PPD1 = 0
        # if 8 <= self.day_hour < 21 and obs["people_2"] != 0:
        #     PPD2 = obs["PPD_2"]
        # else:
        #     PPD2 = 0
        # if 8 <= self.day_hour < 21 and obs["people_3"] != 0:
        #     PPD3 = obs["PPD_3"]
        # else:
        #     PPD3 = 0
        # if 8 <= self.day_hour < 21 and obs["people_4"] != 0:
        #     PPD4 = obs["PPD_4"]
        # else:
        #     PPD4 = 0
        # if 8 <= self.day_hour < 21 and obs["people_5"] != 0:
        #     PPD5 = obs["PPD_5"]
        # else:
        #     PPD5 = 0
        # to_log = {"zone_cooling_setpoint_1": obs["zone_cooling_setpoint_1"],
        #           "zone_cooling_setpoint_2": obs["zone_cooling_setpoint_2"],
        #           "zone_cooling_setpoint_3": obs["zone_cooling_setpoint_3"],
        #           "zone_cooling_setpoint_4": obs["zone_cooling_setpoint_4"],
        #           "zone_cooling_setpoint_5": obs["zone_cooling_setpoint_5"],
        #           "zone_heating_setpoint_1": obs["zone_heating_setpoint_1"],
        #           "zone_heating_setpoint_2": obs["zone_heating_setpoint_2"],
        #           "zone_heating_setpoint_3": obs["zone_heating_setpoint_3"],
        #           "zone_heating_setpoint_4": obs["zone_heating_setpoint_4"],
        #           "zone_heating_setpoint_5": obs["zone_heating_setpoint_5"],
        #           "zone_air_temp_1": obs["zone_air_temp_1"],
        #           "zone_air_temp_2": obs["zone_air_temp_2"],
        #           "zone_air_temp_3": obs["zone_air_temp_3"],
        #           "zone_air_temp_4": obs["zone_air_temp_4"],
        #           "zone_air_temp_5": obs["zone_air_temp_5"],
        #           # "zone_air_temp_6": obs["zone_air_temp_6"],
        #           "VAV_ENERGY_total": self.total_energy_copy,
        #           "VAV_ENERGY": self.VAV_energy,
        #           "VAV_ENERGY_TEMP": (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000,
        #           "hour": self.day_hour,
        #           "PPD1": PPD1,
        #           "PPD2": PPD2,
        #           "PPD3": PPD3,
        #           "PPD4": PPD4,
        #           "PPD5": PPD5,
        #           }
        self.count = self.count + 1
        # self.wandb.log(to_log, step=self.count)
        obs_vec = np.array(list(obs.values()))  # 这是一个列表
        self.week, self.day_hour = self.energyplus.get_time_information()
        self.day_hour_hist.append(self.day_hour)
        self.week_hist.append(self.week)
        day_count = self.day_count % 96
        zone1_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_1"] - 15) / (30 - 15), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 11, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_2"] - 15) / (30 - 15), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 5,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_3"] - 15) / (30 - 15), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 11,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_4"] - 15) / (30 - 15), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 5,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_5"] - 15) / (30 - 15), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 20,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]
        zone6_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_6"] - 15) / (30 - 15), (obs["zone_heating_setpoint_6"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_6"] - 15) / (32 - 15),
                     obs["people_6"] / 20,
                     obs["zone_air_Relative_Humidity_6"] / 100
                     ]
        zone7_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_7"] - 15) / (30 - 15), (obs["zone_heating_setpoint_7"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_7"] - 15) / (32 - 15),
                     obs["people_7"] / 20,
                     obs["zone_air_Relative_Humidity_7"] / 100
                     ]
        zone8_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_8"] - 15) / (30 - 15), (obs["zone_heating_setpoint_8"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_8"] - 15) / (32 - 15),
                     obs["people_8"] / 20,
                     obs["zone_air_Relative_Humidity_8"] / 100
                     ]
        zone9_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_9"] - 15) / (30 - 15), (obs["zone_heating_setpoint_9"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_9"] - 15) / (32 - 15),
                     obs["people_9"] / 20,
                     obs["zone_air_Relative_Humidity_9"] / 100
                     ]
        zone10_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_10"] - 15) / (30 - 15), (obs["zone_heating_setpoint_10"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_10"] - 15) / (32 - 15),
                     obs["people_10"] / 20,
                     obs["zone_air_Relative_Humidity_10"] / 100
                     ]
        
        zone11_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_11"] - 15) / (30 - 15), (obs["zone_heating_setpoint_11"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_11"] - 15) / (32 - 15),
                     obs["people_11"] / 11, obs["zone_air_Relative_Humidity_11"] / 100
                     ]
        zone12_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_12"] - 15) / (30 - 15), (obs["zone_heating_setpoint_12"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_12"] - 15) / (32 - 15),
                     obs["people_12"] / 5,
                     obs["zone_air_Relative_Humidity_12"] / 100
                     ]
        zone13_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_13"] - 15) / (30 - 15), (obs["zone_heating_setpoint_13"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_13"] - 15) / (32 - 15),
                     obs["people_13"] / 11,
                     obs["zone_air_Relative_Humidity_13"] / 100
                     ]
        zone14_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_14"] - 15) / (30 - 15), (obs["zone_heating_setpoint_14"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_14"] - 15) / (32 - 15),
                     obs["people_14"] / 5,
                     obs["zone_air_Relative_Humidity_14"] / 100
                     ]
        zone15_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_15"] - 15) / (30 - 15), (obs["zone_heating_setpoint_15"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_15"] - 15) / (32 - 15),
                     obs["people_15"] / 20,
                     obs["zone_air_Relative_Humidity_15"] / 100
                     ]
        zone16_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_16"] - 15) / (30 - 15), (obs["zone_heating_setpoint_16"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_16"] - 15) / (32 - 15),
                     obs["people_16"] / 20,
                     obs["zone_air_Relative_Humidity_16"] / 100
                     ]
        zone17_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_17"] - 15) / (30 - 15), (obs["zone_heating_setpoint_17"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_17"] - 15) / (32 - 15),
                     obs["people_17"] / 20,
                     obs["zone_air_Relative_Humidity_17"] / 100
                     ]
        zone18_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_18"] - 15) / (30 - 15), (obs["zone_heating_setpoint_18"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_18"] - 15) / (32 - 15),
                     obs["people_18"] / 20,
                     obs["zone_air_Relative_Humidity_18"] / 100
                     ]
        zone19_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_19"] - 15) / (30 - 15), (obs["zone_heating_setpoint_19"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_19"] - 15) / (32 - 15),
                     obs["people_19"] / 20,
                     obs["zone_air_Relative_Humidity_19"] / 100
                     ]
        zone20_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_20"] - 15) / (30 - 15), (obs["zone_heating_setpoint_20"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_20"] - 15) / (32 - 15),
                     obs["people_20"] / 20,
                     obs["zone_air_Relative_Humidity_20"] / 100
                     ]
        

        zone21_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_21"] - 15) / (30 - 15), (obs["zone_heating_setpoint_21"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_21"] - 15) / (32 - 15),
                     obs["people_21"] / 11, obs["zone_air_Relative_Humidity_21"] / 100
                     ]
        zone22_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_22"] - 15) / (30 - 15), (obs["zone_heating_setpoint_22"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_22"] - 15) / (32 - 15),
                     obs["people_22"] / 5,
                     obs["zone_air_Relative_Humidity_22"] / 100
                     ]
        zone23_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_23"] - 15) / (30 - 15), (obs["zone_heating_setpoint_23"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_23"] - 15) / (32 - 15),
                     obs["people_23"] / 11,
                     obs["zone_air_Relative_Humidity_23"] / 100
                     ]
        zone24_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_24"] - 15) / (30 - 15), (obs["zone_heating_setpoint_24"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_24"] - 15) / (32 - 15),
                     obs["people_24"] / 5,
                     obs["zone_air_Relative_Humidity_24"] / 100
                     ]
        zone25_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_25"] - 15) / (30 - 15), (obs["zone_heating_setpoint_25"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_25"] - 15) / (32 - 15),
                     obs["people_25"] / 20,
                     obs["zone_air_Relative_Humidity_25"] / 100
                     ]
        zone26_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_26"] - 15) / (30 - 15), (obs["zone_heating_setpoint_26"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_26"] - 15) / (32 - 15),
                     obs["people_26"] / 20,
                     obs["zone_air_Relative_Humidity_26"] / 100
                     ]
        zone27_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_27"] - 15) / (30 - 15), (obs["zone_heating_setpoint_27"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_27"] - 15) / (32 - 15),
                     obs["people_27"] / 20,
                     obs["zone_air_Relative_Humidity_27"] / 100
                     ]


        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs, zone6_obs, zone7_obs, zone8_obs, zone9_obs, zone10_obs,
                             zone11_obs, zone12_obs, zone13_obs, zone14_obs, zone15_obs, zone16_obs, zone17_obs, zone18_obs, zone19_obs, zone20_obs,
                             zone21_obs, zone22_obs, zone23_obs, zone24_obs, zone25_obs, zone26_obs, zone27_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100, obs["PPD_6"] / 100,
             obs["PPD_7"] / 100, obs["PPD_8"] / 100, obs["PPD_9"] / 100, obs["PPD_10"] / 100, obs["PPD_11"] / 100, obs["PPD_12"] / 100, 
             obs["PPD_13"] / 100, obs["PPD_14"] / 100, obs["PPD_15"] / 100, obs["PPD_16"] / 100, obs["PPD_17"] / 100, obs["PPD_18"] / 100, 
             obs["PPD_19"] / 100, obs["PPD_20"] / 100, obs["PPD_21"] / 100, obs["PPD_22"] / 100, obs["PPD_23"] / 100, obs["PPD_24"] / 100, 
             obs["PPD_25"] / 100, obs["PPD_26"] / 100, obs["PPD_27"] / 100])
        # 更新数据的过程
        # self.ppd1_hist.append(obs["PPD_1"])
        # self.ppd2_hist.append(obs["PPD_2"])
        # self.ppd3_hist.append(obs["PPD_3"])
        # self.ppd4_hist.append(obs["PPD_4"])
        # self.ppd5_hist.append(obs["PPD_5"])
        # self.temperature1.append(obs["zone_air_temp_1"])
        # self.temperature2.append(obs["zone_air_temp_2"])
        # self.temperature3.append(obs["zone_air_temp_3"])
        # self.temperature4.append(obs["zone_air_temp_4"])
        # self.temperature5.append(obs["zone_air_temp_5"])
        # # self.temperature6.append(obs["zone_air_temp_6"])
        # self.occ1_hist.append(obs["people_1"])
        # self.occ2_hist.append(obs["people_2"])
        # self.occ3_hist.append(obs["people_3"])
        # self.occ4_hist.append(obs["people_4"])
        # self.occ5_hist.append(obs["people_5"])
        # self.heating1_setpoint_hist.append(obs["zone_heating_setpoint_1"])
        # self.heating2_setpoint_hist.append(obs["zone_heating_setpoint_2"])
        # self.heating3_setpoint_hist.append(obs["zone_heating_setpoint_3"])
        # self.heating4_setpoint_hist.append(obs["zone_heating_setpoint_4"])
        # self.heating5_setpoint_hist.append(obs["zone_heating_setpoint_5"])
        # # self.heating6_setpoint_hist.append(obs["zone_heating_setpoint_6"])
        # self.cooling1_setpoint_hist.append(obs["zone_cooling_setpoint_1"])
        # self.cooling2_setpoint_hist.append(obs["zone_cooling_setpoint_2"])
        # self.cooling3_setpoint_hist.append(obs["zone_cooling_setpoint_3"])
        # self.cooling4_setpoint_hist.append(obs["zone_cooling_setpoint_4"])
        # self.cooling5_setpoint_hist.append(obs["zone_cooling_setpoint_5"])
        # self.energy_hist.append((self.last_obs["elec_hvac"] + self.last_obs["elec_heat"])/3600000)


        # if done == True:
        #     # 将所有数据合并成一个字典，方便生成 DataFrame
        #     data = {
        #         "hour": self.day_hour_hist,
        #         "Rule_PPD_1": self.ppd1_hist,
        #         "Rule_PPD_2": self.ppd2_hist,
        #         "Rule_PPD_3": self.ppd3_hist,
        #         "Rule_PPD_4": self.ppd4_hist,
        #         "Rule_PPD_5": self.ppd5_hist,
        #         "Rule_People_1": self.occ1_hist,
        #         "Rule_People_2": self.occ2_hist,
        #         "Rule_People_3": self.occ3_hist,
        #         "Rule_People_4": self.occ4_hist,
        #         "Rule_People_5": self.occ5_hist,
        #         "Rule_temperature1": self.temperature1,
        #         "Rule_temperature2": self.temperature2,
        #         "Rule_temperature3": self.temperature3,
        #         "Rule_temperature4": self.temperature4,
        #         "Rule_temperature5": self.temperature5,
        #         # "temperature6": self.temperature6,
        #         "Rule_heating1": self.heating1_setpoint_hist,
        #         "Rule_heating2": self.heating2_setpoint_hist,
        #         "Rule_heating3": self.heating3_setpoint_hist,
        #         "Rule_heating4": self.heating4_setpoint_hist,
        #         "Rule_heating5": self.heating5_setpoint_hist,
        #         # "heating6": self.heating6_setpoint_hist,
        #         "Rule_cooling1": self.cooling1_setpoint_hist,
        #         "Rule_cooling2": self.cooling2_setpoint_hist,
        #         "Rule_cooling3": self.cooling3_setpoint_hist,
        #         "Rule_cooling4": self.cooling4_setpoint_hist,
        #         "Rule_cooling5": self.cooling5_setpoint_hist,
        #         # "cooling6": self.cooling6_setpoint_hist,
        #         "Rule_energy": self.energy_hist,
        #     }

        #     # 创建 DataFrame 并带上表头
        #     df = pd.DataFrame(data)
        #     # 创建 ExcelWriter
        #     with pd.ExcelWriter("Rule_data.xlsx") as writer:
        #         # 保存完整数据到一个 sheet
        #         df.to_excel(writer, sheet_name="All_data", index=False)

        #         # 按每96行分割数据，并保存到不同的 sheets
        #         chunk_size = 96
        #         num_chunks = math.ceil(len(df) / chunk_size)
        #         for i in range(num_chunks):
        #             start_row = i * chunk_size
        #             end_row = start_row + chunk_size
        #             chunk_df = df.iloc[start_row:end_row]
        #             sheet_name = f'Day_{i + 1}'
        #             chunk_df.to_excel(writer, sheet_name=sheet_name, index=False)

        #     print(f"数据已保存到 MATD3_data.xlsx，其中包含 'All_Data' 和 {num_chunks} 个分割的 sheets。")

        return obs_test, reward_local, reward_global, done, self.week, self.day_hour, PPD, day_count

    '''下面这个@property不能删除'''

    @property
    def get_reward(self):
        PPD_thres = 0.2
        w_e = 0.8
        w_c = 0.2
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
        energy = (obs["elec_hvac"] + obs["elec_heat"]) / 20000000 / 2.5 / 4 # 将电能消耗量从瓦特秒转换为千瓦时,这个值可能是在1以下
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
