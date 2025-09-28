import Energyplus_Env
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 获取参数
    num_agent = 5
    Energyplus_env = Energyplus_Env.EnergyPlusEnvironment()
    action = [[30, 15], [30, 15], [30, 15], [30, 15], [30, 15]]
    s, week, hour, PPD, Day_count = Energyplus_env.reset()
    reward_hist = []
    reward_list = []
    reward_hist_copy = []
    reward_list_copy = []
    agent_reward = [0] * num_agent
    agent_reward_local = [0] * num_agent
    agent_reward_global = [0] * num_agent
    max_reward = 100000
    hist = []
    PPD1 = []
    PPD2 = []
    PPD3 = []
    PPD4 = []
    PPD5 = []
    PPD_total = []
    # PPD6 = []
    PPD1_10_violate_count = 0
    PPD2_10_violate_count = 0
    PPD3_10_violate_count = 0
    PPD4_10_violate_count = 0
    PPD5_10_violate_count = 0
    # PPD6_10_violate_count = 0
    PPD1_15_violate_count = 0
    PPD2_15_violate_count = 0
    PPD3_15_violate_count = 0
    PPD4_15_violate_count = 0
    PPD5_15_violate_count = 0
    # PPD6_15_violate_count = 0
    occupy_count = 0
    agent_reward_copy = [0] * num_agent
    max_reward_copy = 100000
    hist_copy = []
    done_flag = 0
    while not done_flag:
        if week == 7 or week == 1:
            action = [[30.0, 15.0], [30.0, 15.0], [30.0, 15.0], [30.0, 15.0], [30.0, 15.0]]
        else:
            if 32 <= Day_count <= 83:
                action1 = [25, 24]
            else:
                action1 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action2 = [25, 24]
            else:
                action2 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action3 = [25, 24]
            else:
                action3 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action4 = [25, 24]
            else:
                action4 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action5 = [25, 24]
            else:
                action5 = [30.0, 15.0]
            action = [action1, action2, action3, action4, action5]
            print('--------------------------------------')
            print('action ', action)
            print('--------------------------------------')

        s_next, r_local, r_global, done_flag, week, hour, PPD, Day_count = Energyplus_env.step(action)

        s = s_next
        r_local_array = np.array(r_local)
        r_global_array = np.array(r_global)
        r = r_local_array + r_global_array

        reward = r

        for i in range(num_agent):
            agent_reward[i] += r[i]
            agent_reward_local[i] += r_local_array[i]
            agent_reward_global[i] += r_global_array[i]

        if s[0][8] != 0:
            occupy_count = occupy_count + 1
            PPD1.append(PPD[0].astype(np.float64))
            PPD_total.append(PPD[0].astype(np.float64))
        if s[1][8] != 0:
            PPD2.append(PPD[1].astype(np.float64))
            PPD_total.append(PPD[1].astype(np.float64))
        if s[2][8] != 0:
            PPD3.append(PPD[2].astype(np.float64))
            PPD_total.append(PPD[2].astype(np.float64))
        if s[3][8] != 0:
            PPD4.append(PPD[3].astype(np.float64))
            PPD_total.append(PPD[3].astype(np.float64))
        if s[4][8] != 0:
            PPD5.append(PPD[4].astype(np.float64))
            PPD_total.append(PPD[4].astype(np.float64))
        if s[0][8] != 0 and PPD[0] > 0.1:
            PPD1_10_violate_count = PPD1_10_violate_count + 1
        if s[0][8] != 0 and PPD[0] > 0.2:
            PPD1_15_violate_count = PPD1_15_violate_count + 1
        if s[1][8] != 0 and PPD[1] > 0.1:
            PPD2_10_violate_count = PPD2_10_violate_count + 1
        if s[1][8] != 0 and PPD[1] > 0.2:
            PPD2_15_violate_count = PPD2_15_violate_count + 1
        if s[2][8] != 0 and PPD[2] > 0.1:
            PPD3_10_violate_count = PPD3_10_violate_count + 1
        if s[2][8] != 0 and PPD[2] > 0.2:
            PPD3_15_violate_count = PPD3_15_violate_count + 1
        if s[3][8] != 0 and PPD[3] > 0.1:
            PPD4_10_violate_count = PPD4_10_violate_count + 1
        if s[3][8] != 0 and PPD[3] > 0.2:
            PPD4_15_violate_count = PPD4_15_violate_count + 1
        if s[4][8] != 0 and PPD[4] > 0.1:
            PPD5_10_violate_count = PPD5_10_violate_count + 1
        if s[4][8] != 0 and PPD[4] > 0.2:
            PPD5_15_violate_count = PPD5_15_violate_count + 1
        reward = np.sum(reward_list) / 96 * 1
        reward_hist.append(reward)

    ave_r = [0] * num_agent
    for i in range(num_agent):
        ave_r[i] = agent_reward[i]
    print("r1:%.2f r2:%.2f r3:%.2f r4:%.2f r5:%.2f " %
          (agent_reward[0], agent_reward[1], agent_reward[2], agent_reward[3], agent_reward[4]))
    print("r_local_1:%.2f r_local_2:%.2f r_local_3:%.2f r_local_4:%.2f r_local_5:%.2f" %
          (agent_reward_local[0], agent_reward_local[1], agent_reward_local[2], agent_reward_local[3],
           agent_reward_local[4]))
    print("r_global_1:%.2f r_global_2:%.2f r_global_3:%.2f r_global_4:%.2f r_global_5:%.2f " %
          (agent_reward_global[0], agent_reward_global[1], agent_reward_global[2], agent_reward_global[3],
           agent_reward_global[4]))

    st = time.time()
    hist.append(agent_reward)
    total_reward = (agent_reward[0] + agent_reward[1] + agent_reward[2] + agent_reward[3] +
                    agent_reward[4])
    total_reward_local = (
            agent_reward_local[0] + agent_reward_local[1] + agent_reward_local[2] + agent_reward_local[3] +
            agent_reward_local[4])
    total_reward_global = (
            agent_reward_global[0] + agent_reward_global[1] + agent_reward_global[2] + agent_reward_global[3] +
            agent_reward_global[4])
    PPD1_reward = sum(PPD1)
    PPD2_reward = sum(PPD2)
    PPD3_reward = sum(PPD3)
    PPD4_reward = sum(PPD4)
    PPD5_reward = sum(PPD5)
    # PPD6_reward = sum(PPD6)
    print("*******************************")
    print("total reward:", total_reward)
    print("total local reward:", total_reward_local)
    print("total global reward:", total_reward_global)
    # print("*******************************")
    print("PPD1_reward_average:", PPD1_reward / occupy_count)
    print("PPD2_reward_average:", PPD2_reward / occupy_count)
    print("PPD3_reward_average:", PPD3_reward / occupy_count)
    print("PPD4_reward_average:", PPD4_reward / occupy_count)
    print("PPD5_reward_average:", PPD5_reward / occupy_count)
    print("5_Zone_PPD_average", (PPD1_reward / occupy_count + PPD2_reward / occupy_count + PPD3_reward / occupy_count +
                                 PPD4_reward / occupy_count + PPD5_reward / occupy_count) / 5 * 100)
    # print("PPD6_reward_average:", PPD6_reward / occupy_count)
    print("PPD1_10_violate_count", PPD1_10_violate_count)
    print("PPD2_10_violate_count", PPD2_10_violate_count)
    print("PPD3_10_violate_count", PPD3_10_violate_count)
    print("PPD4_10_violate_count", PPD4_10_violate_count)
    print("PPD5_10_violate_count", PPD5_10_violate_count)
    # print("PPD6_10_violate_count", PPD6_10_violate_count)
    print("PPD1_15_violate_count", PPD1_15_violate_count)
    print("PPD2_15_violate_count", PPD2_15_violate_count)
    print("PPD3_15_violate_count", PPD3_15_violate_count)
    print("PPD4_15_violate_count", PPD4_15_violate_count)
    print("PPD5_15_violate_count", PPD5_15_violate_count)
    # print("PPD6_15_violate_count", PPD6_15_violate_count)
    print("occupy_count: ", occupy_count)
    print("**************************************************************")
    to_log1 = {
        "agent_reward_1": agent_reward[0],
        "agent_reward_2": agent_reward[1],
        "agent_reward_3": agent_reward[2],
        "agent_reward_4": agent_reward[3],
        "agent_reward_5": agent_reward[4],
        # "agent_reward_6": agent_reward[5],
        "total_reward": total_reward,
        'PPD1': PPD1_reward,
        'PPD2': PPD2_reward,
        'PPD3': PPD3_reward,
        'PPD4': PPD4_reward,
        'PPD5': PPD5_reward,
        # 'PPD6': PPD6_reward,
    }
    # 将列表转换为 NumPy 数组
    PPD1_array = np.array(PPD1)
    PPD2_array = np.array(PPD2)
    PPD3_array = np.array(PPD3)
    PPD4_array = np.array(PPD4)
    PPD5_array = np.array(PPD5)
    PPD_total_array = np.array(PPD_total)
    # 计算均值
    mean_value1 = np.mean(PPD1_array)
    mean_value2 = np.mean(PPD2_array)
    mean_value3 = np.mean(PPD3_array)
    mean_value4 = np.mean(PPD4_array)
    mean_value5 = np.mean(PPD5_array)
    mean_value_total = np.mean(PPD_total_array)
    # 计算标准差
    std_dev1 = np.std(PPD1_array)
    std_dev2 = np.std(PPD2_array)
    std_dev3 = np.std(PPD3_array)
    std_dev4 = np.std(PPD4_array)
    std_dev5 = np.std(PPD5_array)
    std_dev__total = np.std(PPD_total_array)
    print("PPD1均值:", mean_value1 * 100)
    print("PPD1标准差:", std_dev1 * 100)
    print("PPD2均值:", mean_value2 * 100)
    print("PPD2标准差:", std_dev2 * 100)
    print("PPD3均值:", mean_value3 * 100)
    print("PPD3标准差:", std_dev3 * 100)
    print("PPD4均值:", mean_value4 * 100)
    print("PPD4标准差:", std_dev4 * 100)
    print("PPD5均值:", mean_value5 * 100)
    print("PPD5标准差:", std_dev5 * 100)
    print("PPD_total均值:", mean_value_total*100)
    print("PPD_total标准差:", std_dev__total*100)
    Zone_PPD_mean_value_average = (mean_value1 * 100 + mean_value2 * 100 + mean_value3 * 100 + mean_value4 * 100 + mean_value5 * 100) / 5
    Zone_PPD_std_dev_average = (std_dev1 * 100 + std_dev2 * 100 + std_dev3 * 100 + std_dev4 * 100 + std_dev5 * 100) / 5
    # 创建一个字典来存储数据
    data_PPD_statistics = {
        'PPD': ['PPD1', 'PPD2', 'PPD3', 'PPD4', 'PPD5', '5Zone_PPD_average'],
        '均值': [mean_value1 * 100, mean_value2 * 100, mean_value3 * 100, mean_value4 * 100, mean_value5 * 100,
                 mean_value_total*100],
        '标准差': [std_dev1 * 100, std_dev2 * 100, std_dev3 * 100, std_dev4 * 100, std_dev5 * 100,
                   std_dev__total*100]
    }
    # 将字典转换为 DataFrame
    df = pd.DataFrame(data_PPD_statistics)

    # 导出 DataFrame 到 Excel 文件
    df.to_excel('Rule_PPD_statistics.xlsx', index=False)

    # Energyplus_env.wandb.log(to_log1, step=1)
    df2 = pd.DataFrame(hist)
    df2.to_excel('reward.xlsx', index=False)


