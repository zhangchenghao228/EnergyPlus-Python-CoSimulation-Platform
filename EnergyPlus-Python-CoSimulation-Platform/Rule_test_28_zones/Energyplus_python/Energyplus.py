# energyplus library
import sys
import time

sys.path.insert(0, r"D:\Energyplus")
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange

import numpy as np
import csv

import threading

from queue import Queue, Empty, Full
from typing import Dict, Any, Tuple, Optional, List

idf_file = r"E:\modify paper\28zone\28_zone_model\Untitled.idf"
epw_file = r"D:\code\1211summer\CHN_SD_Jinan.Tsinan.548230_TMYx.2004-2018\CHN_SD_Jinan.Tsinan.548230_TMYx.2004-2018.epw"
idd_file = r"D:\Energyplus\Energy+.idd"

# data = []
class EnergyPlus:
    '''
    obs_queue是存放观察值的，queue是队列的意思
    act_queue是存放动作值的，queue是队列的意思
    action_space是动作空间，这个是离散的动作空间
    get_action_func就是如何根据神经网络或者其他规则获取action_apace里面的值，queue是队列的意思
    '''

    def __init__(self, obs_queue: Queue = Queue(1), act_queue: Queue = Queue(1)) -> None:

        # for RL
        self.next_obs1 = None
        self.obs_queue = obs_queue
        self.act_queue = act_queue

        # for energyplus
        self.energyplus_api = EnergyPlusAPI()
        self.dx: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread = None

        # energyplus running states
        self.energyplus_state = None  # 用于存储与EnergyPlus的状态信息
        self.initialized = False  # 指示初始化是否已完成
        self.simulation_complete = False  # 指示仿真是否已经完成
        self.warmup_complete = False  # 指示warmup是否已经完成，每次仿真开始之前都要进行一次warmup
        self.warmup_queue = Queue()  # 是一个队列，用于存储或管理与warmup相关的数据，创建了一个空的线程安全队列
        self.progress_value: int = 0  # 表示模拟的进度
        self.sim_results: Dict[str, Any] = {}  # 存放energyplus的仿真结果

        # request variables to be available during runtime，python向energyplus请求的变量在运行时是否可用，因为energyplus不是所有的变量的是可访问的，在访问变量之前需要向energyplus注册python想要访问的变量
        self.request_variable_complete = False

        # get the variable names csv
        self.has_csv = False

        # variables, meters, actuators
        # look up in the csv file that get_available_data_csv() generate
        # or look up the html file
        '''
        space1-1 都是idf文件里面自定义的名字
        html文件里面也有，可以一个一个试
        csv文件里面也有，csv文件里面可以查看所有的variables、meters和actuators
        '''
        # variables
        self.variables = {
            "zone_air_temp_1": ("Zone Air Temperature", "THERMAL ZONE: ZONE1"),
            "zone_air_temp_2": ("Zone Air Temperature", "THERMAL ZONE: ZONE2"),
            "zone_air_temp_3": ("Zone Air Temperature", "THERMAL ZONE: ZONE3"),
            "zone_air_temp_4": ("Zone Air Temperature", "THERMAL ZONE: ZONE4"),
            "zone_air_temp_5": ("Zone Air Temperature", "THERMAL ZONE: ZONE5"),
            "zone_air_temp_6": ("Zone Air Temperature", "THERMAL ZONE: ZONE6"),
            "zone_air_temp_7": ("Zone Air Temperature", "THERMAL ZONE: ZONE7"),
            "zone_air_temp_8": ("Zone Air Temperature", "THERMAL ZONE: ZONE8"),
            "zone_air_temp_9": ("Zone Air Temperature", "THERMAL ZONE: ZONE9"),
            "zone_air_temp_10": ("Zone Air Temperature", "THERMAL ZONE: ZONE10"),
            "zone_air_temp_11": ("Zone Air Temperature", "THERMAL ZONE: ZONE11"),
            "zone_air_temp_12": ("Zone Air Temperature", "THERMAL ZONE: ZONE12"),
            "zone_air_temp_13": ("Zone Air Temperature", "THERMAL ZONE: ZONE13"),
            "zone_air_temp_14": ("Zone Air Temperature", "THERMAL ZONE: ZONE14"),
            "zone_air_temp_15": ("Zone Air Temperature", "THERMAL ZONE: ZONE15"),
            "zone_air_temp_16": ("Zone Air Temperature", "THERMAL ZONE: ZONE16"),
            "zone_air_temp_17": ("Zone Air Temperature", "THERMAL ZONE: ZONE17"),
            "zone_air_temp_18": ("Zone Air Temperature", "THERMAL ZONE: ZONE18"),
            "zone_air_temp_19": ("Zone Air Temperature", "THERMAL ZONE: ZONE19"),
            "zone_air_temp_20": ("Zone Air Temperature", "THERMAL ZONE: ZONE20"),
            "zone_air_temp_21": ("Zone Air Temperature", "THERMAL ZONE: ZONE21"),
            "zone_air_temp_22": ("Zone Air Temperature", "THERMAL ZONE: ZONE22"),
            "zone_air_temp_23": ("Zone Air Temperature", "THERMAL ZONE: ZONE23"),
            "zone_air_temp_24": ("Zone Air Temperature", "THERMAL ZONE: ZONE24"),
            "zone_air_temp_25": ("Zone Air Temperature", "THERMAL ZONE: ZONE25"),
            "zone_air_temp_26": ("Zone Air Temperature", "THERMAL ZONE: ZONE26"),
            "zone_air_temp_27": ("Zone Air Temperature", "THERMAL ZONE: ZONE27"),
        
            "people_1": ("Zone People Occupant Count", "THERMAL ZONE: ZONE1"),
            "people_2": ("Zone People Occupant Count", "THERMAL ZONE: ZONE2"),
            "people_3": ("Zone People Occupant Count", "THERMAL ZONE: ZONE3"),
            "people_4": ("Zone People Occupant Count", "THERMAL ZONE: ZONE4"),
            "people_5": ("Zone People Occupant Count", "THERMAL ZONE: ZONE5"),
            "people_6": ("Zone People Occupant Count", "THERMAL ZONE: ZONE6"),
            "people_7": ("Zone People Occupant Count", "THERMAL ZONE: ZONE7"),
            "people_8": ("Zone People Occupant Count", "THERMAL ZONE: ZONE8"),
            "people_9": ("Zone People Occupant Count", "THERMAL ZONE: ZONE9"),
            "people_10": ("Zone People Occupant Count", "THERMAL ZONE: ZONE10"),
            "people_11": ("Zone People Occupant Count", "THERMAL ZONE: ZONE11"),
            "people_12": ("Zone People Occupant Count", "THERMAL ZONE: ZONE12"),
            "people_13": ("Zone People Occupant Count", "THERMAL ZONE: ZONE13"),
            "people_14": ("Zone People Occupant Count", "THERMAL ZONE: ZONE14"),
            "people_15": ("Zone People Occupant Count", "THERMAL ZONE: ZONE15"),
            "people_16": ("Zone People Occupant Count", "THERMAL ZONE: ZONE16"),
            "people_17": ("Zone People Occupant Count", "THERMAL ZONE: ZONE17"),
            "people_18": ("Zone People Occupant Count", "THERMAL ZONE: ZONE18"),
            "people_19": ("Zone People Occupant Count", "THERMAL ZONE: ZONE19"),
            "people_20": ("Zone People Occupant Count", "THERMAL ZONE: ZONE20"),
            "people_21": ("Zone People Occupant Count", "THERMAL ZONE: ZONE21"),
            "people_22": ("Zone People Occupant Count", "THERMAL ZONE: ZONE22"),
            "people_23": ("Zone People Occupant Count", "THERMAL ZONE: ZONE23"),
            "people_24": ("Zone People Occupant Count", "THERMAL ZONE: ZONE24"),
            "people_25": ("Zone People Occupant Count", "THERMAL ZONE: ZONE25"),
            "people_26": ("Zone People Occupant Count", "THERMAL ZONE: ZONE26"),
            "people_27": ("Zone People Occupant Count", "THERMAL ZONE: ZONE27"),
            



            "zone_air_Relative_Humidity_1": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE1"),
            "zone_air_Relative_Humidity_2": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE2"),
            "zone_air_Relative_Humidity_3": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE3"),
            "zone_air_Relative_Humidity_4": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE4"),
            "zone_air_Relative_Humidity_5": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE5"),
            "zone_air_Relative_Humidity_6": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE6"),
            "zone_air_Relative_Humidity_7": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE7"),
            "zone_air_Relative_Humidity_8": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE8"),
            "zone_air_Relative_Humidity_9": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE9"),
            "zone_air_Relative_Humidity_10": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE10"),
            "zone_air_Relative_Humidity_11": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE11"),
            "zone_air_Relative_Humidity_12": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE12"),
            "zone_air_Relative_Humidity_13": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE13"),
            "zone_air_Relative_Humidity_14": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE14"),
            "zone_air_Relative_Humidity_15": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE15"),
            "zone_air_Relative_Humidity_16": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE16"),
            "zone_air_Relative_Humidity_17": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE17"),
            "zone_air_Relative_Humidity_18": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE18"),
            "zone_air_Relative_Humidity_19": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE19"),
            "zone_air_Relative_Humidity_20": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE20"),
            "zone_air_Relative_Humidity_21": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE21"),
            "zone_air_Relative_Humidity_22": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE22"),
            "zone_air_Relative_Humidity_23": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE23"),
            "zone_air_Relative_Humidity_24": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE24"),
            "zone_air_Relative_Humidity_25": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE25"),
            "zone_air_Relative_Humidity_26": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE26"),
            "zone_air_Relative_Humidity_27": ("Zone Air Relative Humidity", "THERMAL ZONE: ZONE27"),


            'outdoor_air_drybulb_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            'Outdoor_Air_Relative_Humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
            'Outdoor_Wind_Speed': ('Site Wind Speed', 'Environment'),
            'Outdoor_Wind_Direction': ('Site Wind Direction', 'Environment'),
            'Outdoor_Direct_Solar_Radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
            'Outdoor_Diffuse_Solar_Radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
            
            
            
            'PPD_1': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE1 SPACE HALL PEOPLE'),
            'PPD_2': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE2 SPACE HALL PEOPLE'),
            'PPD_3': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE3 SPACE HALL PEOPLE'),
            'PPD_4': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE4 SPACE HALL PEOPLE'),
            'PPD_5': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE5 SPACE HALL PEOPLE'),
            'PPD_6': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE6 SPACE HALL PEOPLE'),
            'PPD_7': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE7 SPACE HALL PEOPLE'),
            'PPD_8': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE8 SPACE HALL PEOPLE'),
            'PPD_9': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE9 SPACE HALL PEOPLE'),
            'PPD_10': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE10 SPACE HALL PEOPLE'),
            'PPD_11': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE11 SPACE HALL PEOPLE'),
            'PPD_12': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE12 SPACE HALL PEOPLE'),
            'PPD_13': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE13 SPACE HALL PEOPLE'),
            'PPD_14': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE14 SPACE HALL PEOPLE'),
            'PPD_15': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE15 SPACE HALL PEOPLE'),
            'PPD_16': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE16 SPACE HALL PEOPLE'),
            'PPD_17': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE17 SPACE HALL PEOPLE'),
            'PPD_18': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE18 SPACE HALL PEOPLE'),
            'PPD_19': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE19 SPACE HALL PEOPLE'),
            'PPD_20': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE20 SPACE HALL PEOPLE'),
            'PPD_21': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE21 SPACE HALL PEOPLE'),
            'PPD_22': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE22 SPACE HALL PEOPLE'),
            'PPD_23': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE23 SPACE HALL PEOPLE'),
            'PPD_24': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE24 SPACE HALL PEOPLE'),
            'PPD_25': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE25 SPACE HALL PEOPLE'),
            'PPD_26': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE26 SPACE HALL PEOPLE'),
            'PPD_27': ('Zone Thermal Comfort Fanger Model PPD', 'ZONE27 SPACE HALL PEOPLE'),



            "zone_heating_setpoint_1": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE1"),
            "zone_heating_setpoint_2": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE2"),
            "zone_heating_setpoint_3": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE3"),
            "zone_heating_setpoint_4": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE4"),
            "zone_heating_setpoint_5": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE5"),
            "zone_heating_setpoint_6": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE6"),
            "zone_heating_setpoint_7": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE7"),
            "zone_heating_setpoint_8": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE8"),
            "zone_heating_setpoint_9": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE9"),
            "zone_heating_setpoint_10": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE10"),
            "zone_heating_setpoint_11": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE11"),
            "zone_heating_setpoint_12": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE12"),
            "zone_heating_setpoint_13": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE13"),
            "zone_heating_setpoint_14": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE14"),
            "zone_heating_setpoint_15": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE15"),
            "zone_heating_setpoint_16": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE16"),
            "zone_heating_setpoint_17": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE17"),
            "zone_heating_setpoint_18": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE18"),
            "zone_heating_setpoint_19": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE19"),
            "zone_heating_setpoint_20": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE20"),
            "zone_heating_setpoint_21": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE21"),
            "zone_heating_setpoint_22": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE22"),
            "zone_heating_setpoint_23": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE23"),
            "zone_heating_setpoint_24": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE24"),
            "zone_heating_setpoint_25": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE25"),
            "zone_heating_setpoint_26": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE26"),
            "zone_heating_setpoint_27": ("Zone Thermostat Heating Setpoint Temperature", "THERMAL ZONE: ZONE27"),

            "zone_cooling_setpoint_1": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE1"),
            "zone_cooling_setpoint_2": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE2"),
            "zone_cooling_setpoint_3": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE3"),
            "zone_cooling_setpoint_4": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE4"),
            "zone_cooling_setpoint_5": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE5"),
            "zone_cooling_setpoint_6": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE6"),
            "zone_cooling_setpoint_7": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE7"),
            "zone_cooling_setpoint_8": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE8"),
            "zone_cooling_setpoint_9": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE9"),
            "zone_cooling_setpoint_10": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE10"),
            "zone_cooling_setpoint_11": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE11"),
            "zone_cooling_setpoint_12": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE12"),
            "zone_cooling_setpoint_13": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE13"),
            "zone_cooling_setpoint_14": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE14"),
            "zone_cooling_setpoint_15": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE15"),
            "zone_cooling_setpoint_16": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE16"),
            "zone_cooling_setpoint_17": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE17"),
            "zone_cooling_setpoint_18": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE18"),
            "zone_cooling_setpoint_19": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE19"),
            "zone_cooling_setpoint_20": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE20"),
            "zone_cooling_setpoint_21": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE21"),
            "zone_cooling_setpoint_22": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE22"),
            "zone_cooling_setpoint_23": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE23"),
            "zone_cooling_setpoint_24": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE24"),
            "zone_cooling_setpoint_25": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE25"),
            "zone_cooling_setpoint_26": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE26"),
            "zone_cooling_setpoint_27": ("Zone Thermostat Cooling Setpoint Temperature", "THERMAL ZONE: ZONE27"),
            
            # "damper_pos": ("Zone Air Terminal VAV Damper Position","SPACE5-1 VAV REHEAT")#区域空气终端变风量阻尼器位置，看来应该是可以控制VAV终端
        }
        # Heating Coil NaturalGas Energy
        # Cooling Coil Electricity Energy
        self.var_handles: Dict[str, int] = {}

        # meters
        self.meters = {
            # "transfer_cool_1" : "Cooling:EnergyTransfer:Zone:SPACE1-1",
            # "transfer_heat_1" : "Heating:EnergyTransfer:Zone:SPACE1-1",
            # "transfer_cool_2" : "Cooling:EnergyTransfer:Zone:SPACE2-1",
            # "transfer_heat_2" : "Heating:EnergyTransfer:Zone:SPACE2-1",
            # "transfer_cool_3" : "Cooling:EnergyTransfer:Zone:SPACE3-1",
            # "transfer_heat_3" : "Heating:EnergyTransfer:Zone:SPACE3-1",
            # "transfer_cool_4" : "Cooling:EnergyTransfer:Zone:SPACE4-1",
            # "transfer_heat_4" : "Heating:EnergyTransfer:Zone:SPACE4-1",
            # "transfer_cool_5" : "Cooling:EnergyTransfer:Zone:SPACE5-1",
            # "transfer_heat_5" : "Heating:EnergyTransfer:Zone:SPACE5-1",

            # https://unmethours.com/question/55005/hvac-energy-consumption/
            "elec_hvac": "Electricity:HVAC",
            "elec_heat": "Electricity:Plant",
            # "elec_heat": "Electricity:Plant"
            # "elec_heating" : "Heating:Electricity",
            # "elec_cooling": "Cooling:Electricity",
            # "gas_heating" : "Heating:NaturalGas"
        }
        self.meter_handles: Dict[str, int] = {}

        # actuators
        self.actuators = {
            "cooling_1": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE1"
            ),
            "heating_1": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE1"
            ),
            "cooling_2": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE2"
            ),
            "heating_2": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE2"
            ),
            "cooling_3": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE3"
            ),
            "heating_3": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE3"
            ),
            "cooling_4": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE4"
            ),
            "heating_4": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE4"
            ),
            "cooling_5": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE5"
            ),
            "heating_5": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE5"
            ),
            "cooling_6": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE6"
            ),
            "heating_6": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE6"
            ),
            "cooling_7": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE7"
            ),
            "heating_7": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE7"
            ),
            "cooling_8": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE8"
            ),
            "heating_8": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE8"
            ),
            "cooling_9": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE9"
            ),
            "heating_9": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE9"
            ),
            "cooling_10": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE10"
            ),
            "heating_10": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE10"
            ),
            "cooling_11": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE11"
            ),
            "heating_11": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE11"
            ),
            "cooling_12": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE12"
            ),
            "heating_12": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE12"
            ),
            "cooling_13": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE13"
            ),
            "heating_13": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE13"
            ),
            "cooling_14": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE14"
            ),
            "heating_14": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE14"
            ),
            "cooling_15": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE15"
            ),
            "heating_15": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE15"
            ),
            "cooling_16": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE16"
            ),
            "heating_16": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE16"
            ),
            "cooling_17": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE17"
            ),
            "heating_17": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE17"
            ),
            "cooling_18": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE18"
            ),
            "heating_18": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE18"
            ),
            "cooling_19": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE19"
            ),
            "heating_19": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE19"
            ),
            "cooling_20": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE20"
            ),
            "heating_20": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE20"
            ),
            "cooling_21": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE21"
            ),
            "heating_21": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE21"
            ),
            "cooling_22": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE22"
            ),
            "heating_22": (
                "Zone Temperature Control",
                "Heating Setpoint",
                "THERMAL ZONE: ZONE22"
            ),
            "cooling_23": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE23"
            ),
            "heating_23": (
                "Zone Temperature Control",
                "Heating Setpoint", 
                "THERMAL ZONE: ZONE23"
            ),
            "cooling_24": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE24"
            ),
            "heating_24": (
                "Zone Temperature Control",
                "Heating Setpoint", 
                "THERMAL ZONE: ZONE24"
            ),
            "cooling_25": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE25"
            ),
            "heating_25": (
                "Zone Temperature Control",
                "Heating Setpoint", 
                "THERMAL ZONE: ZONE25"
            ),
            "cooling_26": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE26"
            ),
            "heating_26": (
                "Zone Temperature Control",
                "Heating Setpoint", 
                "THERMAL ZONE: ZONE26"
            ),
            "cooling_27": (
                "Zone Temperature Control",
                "Cooling Setpoint",
                "THERMAL ZONE: ZONE27"
            ),
            "heating_27": (
                "Zone Temperature Control",
                "Heating Setpoint", 
                "THERMAL ZONE: ZONE27"
            ),
        }
        self.actuator_handles: Dict[str, int] = {}  # 这是存放句柄的字典

        # self.action_space = action_space
        # self.action_space_size = 0
        # if self.action_space is not None :
        # self.action_space_size = len(self.action_space)
        # self.get_action_func = get_action_func

    def get_time_information(self):
        week_day = self.dx.day_of_week(self.energyplus_state)
        day_hour = self.dx.hour(self.energyplus_state)
        return week_day, day_hour

    def start(self, suffix="defalut"):
        self.energyplus_state = self.energyplus_api.state_manager.new_state()  # 返回EnergyPlus的状态对应的指针，指针指向的内容应该就是EnergyPlus存放的状态，这里边的内容应该会不断的变化，这表示了Energyplus的运行进程
        runtime = self.energyplus_api.runtime

        '''因为energyplus中并不是所有变量都是可以请求的，这里是注册一下需要请求的变量'''
        '''Parameters:state – An active EnergyPlus “state” that is returned from a call to api.state_manager.new_state().
            variable_name – The name of the variable to retrieve, e.g. “Site Outdoor Air DryBulb Temperature”, or “Fan Air Mass Flow Rate”
            variable_key – The instance of the variable to retrieve, e.g. “Environment”, or “Main System Fan
        '''
        # request the variable，因为energyplus中并不是所有变量都是可以请求的，这里是注册一下需要请求的变量
        if not self.request_variable_complete:
            for key, var in self.variables.items():
                self.dx.request_variable(self.energyplus_state, var[0], var[1])
                self.request_variable_complete = True
        '''因为energyplus中并不是所有变量都是可以请求的，这里是注册以后需要请求的变量'''

        # register callback used to track simulation progress,这个函数的作用是检查仿真的进行过程，输入参数是进程值
        def report_progress(progress: int) -> None:
            self.progress_value = progress

        runtime.callback_progress(self.energyplus_state, report_progress)

        # register callback used to signal warmup complete
        def _warmup_complete(state: Any) -> None:
            self.warmup_complete = True
            self.warmup_queue.put(True)

        runtime.callback_after_new_environment_warmup_complete(self.energyplus_state, _warmup_complete)

        # register callback used to collect observations and send actions
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # # register callback used to send actions
        # runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)
        # register callback used to send actions
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            # print(f"running EnergyPlus with args: {cmd_args}")
            '''#这个地方设置为TRUE则控制窗口会显示energyplus的模拟仿真过程'''
            self.energyplus_api.runtime.set_console_output_status(state=state, print_output=False)  # 这一行程序不重要
            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)  # 这个是要启动EnergyPlus运行

        '''创建线程，调用_run_energyplus函数，开始一个EnergyPlus的模拟，args是需要传入的参数'''
        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(suffix),
                self.energyplus_state,
                self.sim_results
            )
        )
        '''启动线程'''
        self.energyplus_exec_thread.start()

    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True  # 模拟完成
            self._flush_queues()  # 将self.obs_queue与self.act_queue队列清空，这应该是和线程相关
            self.energyplus_exec_thread.join()  # 模拟结束，关闭线程
            self.energyplus_exec_thread = None  # energyplus执行线程结束,置为None
            self.energyplus_api.runtime.clear_callbacks()  # 这个用于清理已经注册的所有回调函数,因为此线程已经结束，所以要把所有的回调函数清空
            self.energyplus_api.state_manager.delete_state(
                self.energyplus_state)  # 该函数用于删除现有状态实例，释放内存，也就是要把self.energyplus这个指针给清空

    def _collect_obs(self, state_argument):
        # print("jinru_collect_obs")
        # print(state_argument)
        if self.simulation_complete or not self._init_callback(state_argument):
            # print("jinru_collect_obs_return")
            return
        '''上面函数的意思是仿真结束之后返回空'''
        self.next_obs = {
            **{
                key: self.dx.get_variable_value(state_argument, handle)
                for key, handle in self.var_handles.items()
            }
        }
        # **{}这是解包语法，用于将字典中的键值对解包到新的字典中，self.next_obs是一个字典
        # add the meters such as electricity
        for key, handle in self.meter_handles.items():
            self.next_obs[key] = self.dx.get_meter_value(state_argument, handle)
        # if full, it will block the entire simulation
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
            "zone_heating_setpoint_5"
        ]
        zone_setpoint1 = []
        for key in keys_order:
            zone_setpoint1.append(self.next_obs[key])
        zone_setpoint_array1 = np.array(zone_setpoint1)
        # print('zone_setpoint_array_obs', zone_setpoint_array1)
        # print(f"obs: {self.next_obs}")
        self.obs_queue.put(self.next_obs)  # 将其放到obs_queue队列中，这是将一个字典放到obs_queue队列中
        while self.act_queue.empty():
            # print('success')
            # pass
            time.sleep(0.01)  # 添加100毫秒的延迟，避免频繁的CPU空转
        # print("success_put")

    # def obs(self, state_argument):
    #     # print("jinru_collect_obs")
    #     # if self.simulation_complete or not self._init_callback(state_argument):
    #     #     print("jinru_collect_obs_return")
    #     #     return
    #     '''上面函数的意思是仿真结束之后返回空'''
    #     self.next_obs1 = {
    #         **{
    #             key: self.dx.get_variable_value(state_argument, handle)
    #             for key, handle in self.var_handles.items()
    #         }
    #     }
    #     # **{}这是解包语法，用于将字典中的键值对解包到新的字典中，self.next_obs是一个字典
    #     # add the meters such as electricity
    #     for key, handle in self.meter_handles.items():
    #         self.next_obs1[key] = self.dx.get_meter_value(state_argument, handle)
    #     # if full, it will block the entire simulation
    #     keys_order = [
    #         "zone_cooling_setpoint_1",
    #         "zone_heating_setpoint_1",
    #         "zone_cooling_setpoint_2",
    #         "zone_heating_setpoint_2",
    #         "zone_cooling_setpoint_3",
    #         "zone_heating_setpoint_3",
    #         "zone_cooling_setpoint_4",
    #         "zone_heating_setpoint_4",
    #         "zone_cooling_setpoint_5",
    #         "zone_heating_setpoint_5"
    #     ]
    #     zone_setpoint1 = []
    #     for key in keys_order:
    #         zone_setpoint1.append(self.next_obs1[key])
    #     zone_setpoint_array1 = np.array(zone_setpoint1)
    #     # print('zone_setpoint_array_obs1', zone_setpoint_array1)
    #     return zone_setpoint_array1
    #     # print(f"obs: {self.next_obs}")
    #     # self.obs_queue.put(self.next_obs)  # 将其放到obs_queue队列中，这是将一个字典放到obs_queue队列中
    #     # print("success_put")

    def _send_actions(self, state_argument):
        # print(state_argument)
        # time.sleep(1)
        # current_time = self.dx.current_time(state_argument)
        # print(current_time)
        # print("_send_actions")
        if self.simulation_complete or not self._init_callback(state_argument):
            return
        if self.act_queue.empty():
            # print("_send_actions return")
            return
        # print("_send_actions_success----------------------------")
        '''这个是为什么,为什么要有这个action_idx'''
        # softmax后是给的是一个投票，是index
        # action_idx = self.act_queue.get()
        action_idx = self.act_queue.get()
        # print('action_idx: ', action_idx)
        # actions = self.get_action_func(self.action_space, action_idx)
        actions = action_idx  # 这个地方说明actions应该是一个一维的numpy数组
        '''这个函数是向energyplus输入动作'''
        for i in range(len(self.actuator_handles)):
            # Effective heating set-point higher than effective cooling set-point err
            self.dx.set_actuator_value(
                state=state_argument,
                actuator_handle=list(self.actuator_handles.values())[i],
                actuator_value=actions[i]
            )

    '''此函数的用途是将self.obs_queue与self.act_queue队列清空'''

    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()

    '''此函数的用途是将self.obs_queue与self.act_queue队列清空'''

    def make_eplus_args(self, suffix="default"):
        args = [
            "-i",
            idd_file,
            "-w",
            epw_file,
            "-d",
            "res",
            "-p",
            suffix,
            "-x",
            "-r",
            idf_file,
        ]
        return args

    """initialize EnergyPlus handles and checks if simulation runtime is ready，表示energyplus是否已经初始化完成"""

    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) \
                           and not self.dx.warmup_flag(state_argument)
        return self.initialized

    # self.dx.warmup_flag(state_argument)返回值为1的时候表示energyplus正在warmup
    # self._init_handles(state_argument)意思是energyplus是否已经初始化完成
    """initialize EnergyPlus handles and checks if simulation runtime is ready，表示energyplus是否已经初始化完成"""

    '''这个函数用于初始化energyplus的句柄'''

    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        '''初始话句柄用于与energyplus运行时的交互'''
        if not self.initialized:
            if not self.dx.api_data_fully_ready(state_argument):
                return False
            # 上面这个函数的意思是否数据交换API已经准备好
            '''get_variable_handle的作用是获取运行模拟中输出变量的句柄'''
            # store the handles so that we do not need get the hand every callback
            self.var_handles = {
                key: self.dx.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }
            '''获取并保存句柄'''
            self.meter_handles = {
                key: self.dx.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }
            '''获取并保存句柄'''
            self.actuator_handles = {
                key: self.dx.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }
            '''获取并保存句柄'''
            '''因为句柄等于-1表示上述操作没有找到句柄，下面是打印错误操作，当没有找到对应的句柄时，说明variables、meters、actuators中是有错误的'''
            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    print("Error! there is -1 in handle! check the variable names in the var.csv")

                    print("variables:")
                    for k in self.var_handles:
                        print(self.var_handles[k])

                    print("meters:")
                    for k in self.meter_handles:
                        print(self.meter_handles[k])

                    print("actuators")
                    for k in self.actuator_handles:
                        print(k)

                    self.get_available_data_csv(state_argument)
                    exit(1)

            self.initialized = True

        return True

    '''这个函数用于初始化energyplus的句柄'''
    # get the name and key for handles
    '''这个是以易于解析的 CSV 格式列出所有应用程序接口数据内容,通过这个csv文件可以查看能够交互的所有内容'''

    def get_available_data_csv(self, state):
        if self.has_csv:
            return
        else:
            available_data = self.dx.list_available_api_data_csv(self.energyplus_state).decode("utf-8")
            lines = available_data.split('\n')
            with open("var.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for line in lines:
                    fields = line.split(',')
                    writer.writerow(fields)

            self.has_csv = True

    '''这个是以易于解析的 CSV 格式列出所有应用程序接口数据内容,通过这个csv文件可以查看能够交互的所有内容'''

    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0
