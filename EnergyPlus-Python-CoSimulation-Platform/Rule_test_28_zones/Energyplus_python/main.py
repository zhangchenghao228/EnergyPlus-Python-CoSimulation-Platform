import Energyplus_Env
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 获取参数
    num_agent = 27
    Energyplus_env = Energyplus_Env.EnergyPlusEnvironment()
    action = [[30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15]
              ]
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
    PPD6 = []
    PPD7 = []
    PPD8 = []
    PPD9 = []
    PPD10 = []
    PPD11 = []
    PPD12 = []
    PPD13 = []
    PPD14 = []
    PPD15 = []
    PPD16 = []
    PPD17 = []
    PPD18 = []
    PPD19 = []
    PPD20 = []
    PPD21 = []
    PPD22 = []
    PPD23 = []
    PPD24 = []
    PPD25 = []
    PPD26 = []
    PPD27 = []
    PPD_total = []
    # PPD6 = []
    PPD1_10_violate_count = 0
    PPD2_10_violate_count = 0
    PPD3_10_violate_count = 0
    PPD4_10_violate_count = 0
    PPD5_10_violate_count = 0
    PPD6_10_violate_count = 0
    PPD7_10_violate_count = 0
    PPD8_10_violate_count = 0
    PPD9_10_violate_count = 0
    PPD10_10_violate_count = 0
    PPD11_10_violate_count = 0
    PPD12_10_violate_count = 0
    PPD13_10_violate_count = 0
    PPD14_10_violate_count = 0
    PPD15_10_violate_count = 0
    PPD16_10_violate_count = 0
    PPD17_10_violate_count = 0
    PPD18_10_violate_count = 0
    PPD19_10_violate_count = 0
    PPD20_10_violate_count = 0
    PPD21_10_violate_count = 0
    PPD22_10_violate_count = 0
    PPD23_10_violate_count = 0
    PPD24_10_violate_count = 0
    PPD25_10_violate_count = 0
    PPD26_10_violate_count = 0
    PPD27_10_violate_count = 0
    # PPD6_10_violate_count = 0
    PPD1_15_violate_count = 0
    PPD2_15_violate_count = 0
    PPD3_15_violate_count = 0
    PPD4_15_violate_count = 0
    PPD5_15_violate_count = 0
    PPD6_15_violate_count = 0
    PPD7_15_violate_count = 0
    PPD8_15_violate_count = 0
    PPD9_15_violate_count = 0
    PPD10_15_violate_count = 0
    PPD11_15_violate_count = 0
    PPD12_15_violate_count = 0
    PPD13_15_violate_count = 0
    PPD14_15_violate_count = 0
    PPD15_15_violate_count = 0
    PPD16_15_violate_count = 0
    PPD17_15_violate_count = 0
    PPD18_15_violate_count = 0
    PPD19_15_violate_count = 0
    PPD20_15_violate_count = 0
    PPD21_15_violate_count = 0
    PPD22_15_violate_count = 0
    PPD23_15_violate_count = 0
    PPD24_15_violate_count = 0
    PPD25_15_violate_count = 0
    PPD26_15_violate_count = 0
    PPD27_15_violate_count = 0
    # PPD6_15_violate_count = 0
    occupy_count = 0
    agent_reward_copy = [0] * num_agent
    max_reward_copy = 100000
    hist_copy = []
    done_flag = 0
    while not done_flag:
        if week == 7 or week == 1:
            action = [[30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15], [30, 15], [30, 15], [30, 15],
              [30, 15], [30, 15]
              ]
        else:
            if 32 <= Day_count <= 83:#32
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
            if 32 <= Day_count <= 83:
                action6 = [25, 24]
            else:
                action6 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action7 = [25, 24]
            else:
                action7 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action8 = [25, 24]
            else:
                action8 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action9 = [25, 24]
            else:
                action9 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action10 = [25, 24]
            else:
                action10 = [30.0, 15.0]

            if 32 <= Day_count <= 83:
                action11 = [25, 24]
            else:
                action11 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action12 = [25, 24]
            else:
                action12 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action13 = [25, 24]
            else:
                action13 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action14 = [25, 24]
            else:
                action14 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action15 = [25, 24]
            else:
                action15 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action16 = [25, 24]
            else:
                action16 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action17 = [25, 24]
            else:
                action17 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action18 = [25, 24]
            else:
                action18 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action19 = [25, 24]
            else:
                action19 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action20 = [25, 24]
            else:
                action20 = [30.0, 15.0]


            if 32 <= Day_count <= 83:
                action21 = [25, 24]
            else:
                action21 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action22 = [25, 24]
            else:
                action22 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action23 = [25, 24]
            else:
                action23 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action24 = [25, 24]
            else:
                action24 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action25 = [25, 24]
            else:
                action25 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action26 = [25, 24]
            else:
                action26 = [30.0, 15.0]
            if 32 <= Day_count <= 83:
                action27 = [25, 24]
            else:
                action27 = [30.0, 15.0]
            





            action = [action1, action2, action3, action4, action5,
                      action6, action7, action8, action9, action10,
                      action11, action12, action13, action14, action15,
                      action16, action17, action18, action19, action20,
                      action21, action22, action23, action24, action25,
                      action26, action27
                      ]
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

        if s[0][9] != 0:
            occupy_count = occupy_count + 1
            PPD1.append(PPD[0].astype(np.float64))
        if s[1][9] != 0:
            PPD2.append(PPD[1].astype(np.float64))
        if s[2][9] != 0:
            PPD3.append(PPD[2].astype(np.float64))
        if s[3][9] != 0:
            PPD4.append(PPD[3].astype(np.float64))
        if s[4][9] != 0:
            PPD5.append(PPD[4].astype(np.float64))

        if s[5][9] != 0:
            PPD6.append(PPD[5].astype(np.float64))
        if s[6][9] != 0:
            PPD7.append(PPD[6].astype(np.float64))
        if s[7][9] != 0:
            PPD8.append(PPD[7].astype(np.float64))
        if s[8][9] != 0:
            PPD9.append(PPD[8].astype(np.float64))
        if s[9][9] != 0:
            PPD10.append(PPD[9].astype(np.float64))

        if s[10][9] != 0:
            PPD11.append(PPD[10].astype(np.float64))
        if s[11][9] != 0:
            PPD12.append(PPD[11].astype(np.float64))
        if s[12][9] != 0:
            PPD13.append(PPD[12].astype(np.float64))
        if s[13][9] != 0:
            PPD14.append(PPD[13].astype(np.float64))
        if s[14][9] != 0:
            PPD15.append(PPD[14].astype(np.float64))

        if s[15][9] != 0:
            PPD16.append(PPD[15].astype(np.float64))
        if s[16][9] != 0:
            PPD17.append(PPD[16].astype(np.float64))
        if s[17][9] != 0:
            PPD18.append(PPD[17].astype(np.float64))
        if s[18][9] != 0:
            PPD19.append(PPD[18].astype(np.float64))
        if s[19][9] != 0:
            PPD20.append(PPD[19].astype(np.float64))

        if s[20][9] != 0:
            PPD21.append(PPD[20].astype(np.float64))
        if s[21][9] != 0:
            PPD22.append(PPD[21].astype(np.float64))
        if s[22][9] != 0:
            PPD23.append(PPD[22].astype(np.float64))
        if s[23][9] != 0:
            PPD24.append(PPD[23].astype(np.float64))
        if s[24][9] != 0:
            PPD25.append(PPD[24].astype(np.float64))

        if s[25][9] != 0:
            PPD26.append(PPD[25].astype(np.float64))
        if s[26][9] != 0:
            PPD27.append(PPD[26].astype(np.float64))





            
        reward = np.sum(reward_list) / 96 * 1
        reward_hist.append(reward)

    ave_r = [0] * num_agent
    for i in range(num_agent):
        ave_r[i] = agent_reward[i]
    # print("r1:%.2f r2:%.2f r3:%.2f r4:%.2f r5:%.2f " %
    #       (agent_reward[0], agent_reward[1], agent_reward[2], agent_reward[3], agent_reward[4]))
    # print("r_local_1:%.2f r_local_2:%.2f r_local_3:%.2f r_local_4:%.2f r_local_5:%.2f" %
    #       (agent_reward_local[0], agent_reward_local[1], agent_reward_local[2], agent_reward_local[3],
    #        agent_reward_local[4]))
    # print("r_global_1:%.2f r_global_2:%.2f r_global_3:%.2f r_global_4:%.2f r_global_5:%.2f " %
    #       (agent_reward_global[0], agent_reward_global[1], agent_reward_global[2], agent_reward_global[3],
    #        agent_reward_global[4]))

    st = time.time()
    hist.append(agent_reward)
    total_reward = (agent_reward[0] + agent_reward[1] + agent_reward[2] + agent_reward[3] +
                            agent_reward[4] + agent_reward[5] + agent_reward[6] + agent_reward[7] + agent_reward[8] +
                            agent_reward[9] + agent_reward[10] + agent_reward[11] + agent_reward[12] + agent_reward[13] +
                            agent_reward[14] + agent_reward[15] + agent_reward[16] + agent_reward[17] + agent_reward[18] +
                            agent_reward[19] + agent_reward[20] + agent_reward[21] + agent_reward[22] + agent_reward[23] +
                            agent_reward[24] + agent_reward[25] + agent_reward[26] )
    # total_reward_local = (
    #         agent_reward_local[0] + agent_reward_local[1] + agent_reward_local[2] + agent_reward_local[3] +
    #         agent_reward_local[4])
    # total_reward_global = (
    #         agent_reward_global[0] + agent_reward_global[1] + agent_reward_global[2] + agent_reward_global[3] +
    #         agent_reward_global[4])
    PPD1_reward = sum(PPD1)
    PPD2_reward = sum(PPD2)
    PPD3_reward = sum(PPD3)
    PPD4_reward = sum(PPD4)
    PPD5_reward = sum(PPD5)
    PPD6_reward = sum(PPD6)
    PPD7_reward = sum(PPD7)
    PPD8_reward = sum(PPD8)
    PPD9_reward = sum(PPD9)
    PPD10_reward = sum(PPD10)
    PPD11_reward = sum(PPD11)
    PPD12_reward = sum(PPD12)
    PPD13_reward = sum(PPD13)
    PPD14_reward = sum(PPD14)
    PPD15_reward = sum(PPD15)
    PPD16_reward = sum(PPD16)
    PPD17_reward = sum(PPD17)
    PPD18_reward = sum(PPD18)
    PPD19_reward = sum(PPD19)
    PPD20_reward = sum(PPD20)
    PPD21_reward = sum(PPD21)
    PPD22_reward = sum(PPD22)
    PPD23_reward = sum(PPD23)
    PPD24_reward = sum(PPD24)
    PPD25_reward = sum(PPD25)
    PPD26_reward = sum(PPD26)
    PPD27_reward = sum(PPD27)
    # PPD6_reward = sum(PPD6)
    print("*******************************")
    print("total reward:", total_reward)
    # print("total local reward:", total_reward_local)
    # print("total global reward:", total_reward_global)
    # print("*******************************")
    print("PPD1_reward_average:", PPD1_reward / occupy_count)
    print("PPD2_reward_average:", PPD2_reward / occupy_count)
    print("PPD3_reward_average:", PPD3_reward / occupy_count)
    print("PPD4_reward_average:", PPD4_reward / occupy_count)
    print("PPD5_reward_average:", PPD5_reward / occupy_count)
    print("PPD6_reward_average:", PPD6_reward / occupy_count)
    print("PPD7_reward_average:", PPD7_reward / occupy_count)
    print("PPD8_reward_average:", PPD8_reward / occupy_count)
    print("PPD9_reward_average:", PPD9_reward / occupy_count)
    print("PPD10_reward_average:", PPD10_reward / occupy_count)
    print("PPD11_reward_average:", PPD11_reward / occupy_count)
    print("PPD12_reward_average:", PPD12_reward / occupy_count)
    print("PPD13_reward_average:", PPD13_reward / occupy_count)
    print("PPD14_reward_average:", PPD14_reward / occupy_count)
    print("PPD15_reward_average:", PPD15_reward / occupy_count)
    print("PPD16_reward_average:", PPD16_reward / occupy_count)
    print("PPD17_reward_average:", PPD17_reward / occupy_count)
    print("PPD18_reward_average:", PPD18_reward / occupy_count)
    print("PPD19_reward_average:", PPD19_reward / occupy_count)
    print("PPD20_reward_average:", PPD20_reward / occupy_count)
    print("PPD21_reward_average:", PPD21_reward / occupy_count)
    print("PPD22_reward_average:", PPD22_reward / occupy_count)
    print("PPD23_reward_average:", PPD23_reward / occupy_count)
    print("PPD24_reward_average:", PPD24_reward / occupy_count)
    print("PPD25_reward_average:", PPD25_reward / occupy_count)
    print("PPD26_reward_average:", PPD26_reward / occupy_count)
    print("PPD27_reward_average:", PPD27_reward / occupy_count)



    print("27_Zone_PPD_average", (PPD1_reward / occupy_count + PPD2_reward / occupy_count + PPD3_reward / occupy_count+
                                   PPD4_reward / occupy_count + PPD5_reward / occupy_count + PPD6_reward / occupy_count + PPD7_reward / occupy_count+
                                   
                                   PPD8_reward / occupy_count + PPD9_reward / occupy_count + PPD10_reward / occupy_count + PPD11_reward / occupy_count+

                                   PPD12_reward / occupy_count + PPD13_reward / occupy_count + PPD14_reward / occupy_count + PPD15_reward / occupy_count
                                   
                                   +PPD16_reward / occupy_count + PPD17_reward / occupy_count + PPD18_reward / occupy_count + PPD19_reward / occupy_count

                                   +PPD20_reward / occupy_count + PPD21_reward / occupy_count + PPD22_reward / occupy_count + PPD23_reward / occupy_count

                                   +PPD24_reward / occupy_count + PPD25_reward / occupy_count + PPD26_reward / occupy_count + PPD27_reward / occupy_count
                                   ) / 27*100)
    print("occupy_count: ", occupy_count)
    print("**************************************************************")
    # to_log1 = {
    #     "agent_reward_1": agent_reward[0],
    #     "agent_reward_2": agent_reward[1],
    #     "agent_reward_3": agent_reward[2],
    #     "agent_reward_4": agent_reward[3],
    #     "agent_reward_5": agent_reward[4],
    #     # "agent_reward_6": agent_reward[5],
    #     "total_reward": total_reward,
    #     'PPD1': PPD1_reward,
    #     'PPD2': PPD2_reward,
    #     'PPD3': PPD3_reward,
    #     'PPD4': PPD4_reward,
    #     'PPD5': PPD5_reward,
    #     # 'PPD6': PPD6_reward,
    # }
    # 将列表转换为 NumPy 数组
    PPD1_array = np.array(PPD1)
    PPD2_array = np.array(PPD2)
    PPD3_array = np.array(PPD3)
    PPD4_array = np.array(PPD4)
    PPD5_array = np.array(PPD5)
    PPD6_array = np.array(PPD6)
    PPD7_array = np.array(PPD7)
    PPD8_array = np.array(PPD8)
    PPD9_array = np.array(PPD9)
    PPD10_array = np.array(PPD10)
    PPD11_array = np.array(PPD11)
    PPD12_array = np.array(PPD12)
    PPD13_array = np.array(PPD13)
    PPD14_array = np.array(PPD14)
    PPD15_array = np.array(PPD15)
    PPD16_array = np.array(PPD16)
    PPD17_array = np.array(PPD17)
    PPD18_array = np.array(PPD18)
    PPD19_array = np.array(PPD19)
    PPD20_array = np.array(PPD20)
    PPD21_array = np.array(PPD21)
    PPD22_array = np.array(PPD22)
    PPD23_array = np.array(PPD23)
    PPD24_array = np.array(PPD24)
    PPD25_array = np.array(PPD25)
    PPD26_array = np.array(PPD26)
    PPD27_array = np.array(PPD27)
    # 计算均值
    mean_value1 = np.mean(PPD1_array)
    mean_value2 = np.mean(PPD2_array)
    mean_value3 = np.mean(PPD3_array)
    mean_value4 = np.mean(PPD4_array)
    mean_value5 = np.mean(PPD5_array)
    mean_value6 = np.mean(PPD6_array)
    mean_value7 = np.mean(PPD7_array)
    mean_value8 = np.mean(PPD8_array)
    mean_value9 = np.mean(PPD9_array)
    mean_value10 = np.mean(PPD10_array)
    mean_value11 = np.mean(PPD11_array)
    mean_value12 = np.mean(PPD12_array)
    mean_value13 = np.mean(PPD13_array)
    mean_value14 = np.mean(PPD14_array)
    mean_value15 = np.mean(PPD15_array)
    mean_value16 = np.mean(PPD16_array)
    mean_value17 = np.mean(PPD17_array)
    mean_value18 = np.mean(PPD18_array)
    mean_value19 = np.mean(PPD19_array)
    mean_value20 = np.mean(PPD20_array)
    mean_value21 = np.mean(PPD21_array)
    mean_value22 = np.mean(PPD22_array)
    mean_value23 = np.mean(PPD23_array)
    mean_value24 = np.mean(PPD24_array)
    mean_value25 = np.mean(PPD25_array)
    mean_value26 = np.mean(PPD26_array)
    mean_value27 = np.mean(PPD27_array)
    # mean_value_total = np.mean(PPD_total_array)
    # 计算标准差
    # 计算标准差
    std_dev1 = np.std(PPD1_array)
    std_dev2 = np.std(PPD2_array)
    std_dev3 = np.std(PPD3_array)
    std_dev4 = np.std(PPD4_array)
    std_dev5 = np.std(PPD5_array)
    std_dev6 = np.std(PPD6_array)
    std_dev7 = np.std(PPD7_array)
    std_dev8 = np.std(PPD8_array)
    std_dev9 = np.std(PPD9_array)
    std_dev10 = np.std(PPD10_array)
    std_dev11 = np.std(PPD11_array)
    std_dev12 = np.std(PPD12_array)
    std_dev13 = np.std(PPD13_array)
    std_dev14 = np.std(PPD14_array)
    std_dev15 = np.std(PPD15_array)
    std_dev16 = np.std(PPD16_array)
    std_dev17 = np.std(PPD17_array)
    std_dev18 = np.std(PPD18_array)
    std_dev19 = np.std(PPD19_array)
    std_dev20 = np.std(PPD20_array)
    std_dev21 = np.std(PPD21_array)
    std_dev22 = np.std(PPD22_array)
    std_dev23 = np.std(PPD23_array)
    std_dev24 = np.std(PPD24_array)
    std_dev25 = np.std(PPD25_array)
    std_dev26 = np.std(PPD26_array)
    std_dev27 = np.std(PPD27_array)

            
    print("PPD1均值:", mean_value1*100)
    print("PPD1标准差:", std_dev1*100)
    print("PPD2均值:", mean_value2*100)
    print("PPD2标准差:", std_dev2*100)
    print("PPD3均值:", mean_value3*100)
    print("PPD3标准差:", std_dev3*100)
    print("PPD4均值:", mean_value4*100)
    print("PPD4标准差:", std_dev4*100)
    print("PPD5均值:", mean_value5*100)
    print("PPD5标准差:", std_dev5*100)
    print("PPD6均值:", mean_value6*100)
    print("PPD6标准差:", std_dev6*100)
    print("PPD7均值:", mean_value7*100)
    print("PPD7标准差:", std_dev7*100)
    print("PPD8均值:", mean_value8*100)
    print("PPD8标准差:", std_dev8*100)
    print("PPD9均值:", mean_value9*100)
    print("PPD9标准差:", std_dev9*100)
    print("PPD10均值:", mean_value10*100)
    print("PPD10标准差:", std_dev10*100)

    print("PPD11均值:", mean_value11*100)
    print("PPD11标准差:", std_dev11*100)
    print("PPD12均值:", mean_value12*100)
    print("PPD12标准差:", std_dev12*100)
    print("PPD13均值:", mean_value13*100)
    print("PPD13标准差:", std_dev13*100)
    print("PPD14均值:", mean_value14*100)
    print("PPD14标准差:", std_dev14*100)
    print("PPD15均值:", mean_value15*100)
    print("PPD15标准差:", std_dev15*100)
    print("PPD16均值:", mean_value16*100)
    print("PPD16标准差:", std_dev16*100)
    print("PPD17均值:", mean_value17*100)
    print("PPD17标准差:", std_dev17*100)
    print("PPD18均值:", mean_value18*100)
    print("PPD18标准差:", std_dev18*100)
    print("PPD19均值:", mean_value19*100)
    print("PPD19标准差:", std_dev19*100)
    print("PPD20均值:", mean_value20*100)
    print("PPD20标准差:", std_dev20*100)


    print("PPD21均值:", mean_value21*100)
    print("PPD21标准差:", std_dev21*100)
    print("PPD22均值:", mean_value22*100)
    print("PPD22标准差:", std_dev22*100)
    print("PPD23均值:", mean_value23*100)
    print("PPD23标准差:", std_dev23*100)
    print("PPD24均值:", mean_value24*100)
    print("PPD24标准差:", std_dev24*100)
    print("PPD25均值:", mean_value25*100)
    print("PPD25标准差:", std_dev25*100)
    print("PPD26均值:", mean_value26*100)
    print("PPD26标准差:", std_dev26*100)
    print("PPD27均值:", mean_value27*100)
    print("PPD27标准差:", std_dev27*100)

    # print("PPD_total均值:", mean_value_total*100)
    # print("PPD_total标准差:", std_dev__total*100)
    # 创建一个字典来存储数据
    Zone_PPD_mean_value_average = (mean_value1 * 100 + mean_value2 * 100 + mean_value3 * 100 + mean_value4 * 100 + mean_value5 * 100 +
                                           mean_value6 * 100 + mean_value7 * 100 + mean_value8 * 100 + mean_value9 * 100 + mean_value10 * 100 +
                                           mean_value11 * 100 + mean_value12 * 100 + mean_value13 * 100 + mean_value14 * 100 + mean_value15 * 100 +
                                           mean_value16 * 100 + mean_value17 * 100 + mean_value18 * 100 + mean_value19 * 100 + mean_value20 * 100 +
                                           mean_value21 * 100 + mean_value22 * 100 + mean_value23 * 100 + mean_value24 * 100 + mean_value25 * 100 +
                                           mean_value26 * 100 + mean_value27 * 100 ) / 27
            
    Zone_PPD_std_dev_average = (std_dev1 * 100 + std_dev2 * 100 + std_dev3 * 100 + std_dev4 * 100 + std_dev5 * 100 +
                                        
                                        std_dev6 * 100 + std_dev7 * 100 + std_dev8 * 100 + std_dev9 * 100 + std_dev10 * 100 +

                                        std_dev11 * 100 + std_dev12 * 100 + std_dev13 * 100 + std_dev14 * 100 + std_dev15 * 100 +

                                        std_dev16 * 100 + std_dev17 * 100 + std_dev18 * 100 + std_dev19 * 100 + std_dev20 * 100 +

                                        std_dev21 * 100 + std_dev22 * 100 + std_dev23 * 100 + std_dev24 * 100 + std_dev25 * 100 +

                                        std_dev26 * 100 + std_dev27 * 100 
                                        ) / 27
            
    data_PPD_statistics = {
                'PPD': ['PPD1', 'PPD2', 'PPD3', 'PPD4', 'PPD5',
                        'PPD6', 'PPD7', 'PPD8', 'PPD9', 'PPD10',
                        'PPD11', 'PPD12', 'PPD13', 'PPD14', 'PPD15',
                        'PPD16', 'PPD17', 'PPD18', 'PPD19', 'PPD20',
                        'PPD21', 'PPD22', 'PPD23', 'PPD24', 'PPD25',
                        'PPD26', 'PPD27',
                         '27Zone_PPD_average'],


                '均值': [mean_value1 * 100, mean_value2 * 100, mean_value3 * 100, mean_value4 * 100, mean_value5 * 100,
                         mean_value6 * 100, mean_value7 * 100, mean_value8 * 100, mean_value9 * 100, mean_value10 * 100,

                         mean_value11 * 100, mean_value12 * 100, mean_value13 * 100, mean_value14 * 100, mean_value15 * 100,
                         mean_value16 * 100, mean_value17 * 100, mean_value18 * 100, mean_value19 * 100, mean_value20 * 100,
                         
                         mean_value21 * 100, mean_value22 * 100, mean_value23 * 100, mean_value24 * 100, mean_value25 * 100,

                        mean_value26 * 100, mean_value27 * 100,

                        Zone_PPD_mean_value_average],

                
                '标准差': [std_dev1 * 100, std_dev2 * 100, std_dev3 * 100, std_dev4 * 100, std_dev5 * 100,
                           
                        std_dev6 * 100, std_dev7 * 100, std_dev8 * 100, std_dev9 * 100, std_dev10 * 100,

                        std_dev11 * 100, std_dev12 * 100, std_dev13 * 100, std_dev14 * 100, std_dev15 * 100,
                        
                        std_dev16 * 100, std_dev17 * 100, std_dev18 * 100, std_dev19 * 100, std_dev20 * 100,

                        std_dev21 * 100, std_dev22 * 100, std_dev23 * 100, std_dev24 * 100, std_dev25 * 100,

                        std_dev26 * 100, std_dev27 * 100,

                        Zone_PPD_std_dev_average]
            }
    # 将字典转换为 DataFrame
    df = pd.DataFrame(data_PPD_statistics)

    # 导出 DataFrame 到 Excel 文件
    df.to_excel('Rule_PPD_statistics.xlsx', index=False)

    # Energyplus_env.wandb.log(to_log1, step=1)
    df2 = pd.DataFrame(hist)
    df2.to_excel('reward.xlsx', index=False)


