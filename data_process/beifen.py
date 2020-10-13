# Author: CHEN, Jiaxing
# Function: Data Process
# Date: 2020.10.03
# Modified by:
# Changes:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取原始数据
yuanshi_shujv = pd.read_table('hld_and_rxd.fzp', encoding = 'gbk')
# print(yuanshi_shujv.head(50))

# 提取有用数据
shuzhi_shujv_col = yuanshi_shujv.iloc[13:]
# print(shuzhi_shujv_col)

shuzhi_shujv_fenge = pd.DataFrame([jj.split(';') for jj in shuzhi_shujv_col['Vehicle Record']])
# print(shuzhi_shujv_fenge)

shuzhi_shujv_fenge.columns = shuzhi_shujv_fenge.iloc[0]
# print(shuzhi_shujv_fenge.columns)
shuzhi_shujv_fenge = shuzhi_shujv_fenge.iloc[1:]
# print(shuzhi_shujv_fenge)

# 类型转换
shuzhi_shujv_fenge['Lane'] = shuzhi_shujv_fenge['Lane'].astype(int)
shuzhi_shujv_fenge['    WorldX'] = shuzhi_shujv_fenge['    WorldX'].astype(float)
shuzhi_shujv_fenge['    WorldY'] = shuzhi_shujv_fenge['    WorldY'].astype(float)
shuzhi_shujv_fenge['     a'] = shuzhi_shujv_fenge['     a'].astype(float)
shuzhi_shujv_fenge['     VehNr'] = shuzhi_shujv_fenge['     VehNr'].astype(int)
shuzhi_shujv_fenge['       t'] = shuzhi_shujv_fenge['       t'].astype(float)
shuzhi_shujv_fenge['    vMS'] = shuzhi_shujv_fenge['    vMS'].astype(float)
shuzhi_shujv_fenge[' Power'] = shuzhi_shujv_fenge[' Power'].astype(float)
shuzhi_shujv_fenge['  LVeh'] = shuzhi_shujv_fenge['  LVeh'].astype(int)
shuzhi_shujv_fenge['  Head'] = shuzhi_shujv_fenge['  Head'].astype(float)

# print(shuzhi_shujv_fenge)

'''
# 提取队列信息测试
temp = shuzhi_shujv_fenge.iloc[0:25]
print(temp)

print(temp.shape[0])
print(temp.iloc[24][1])
print(list(temp.iloc[24]))
print(list(temp.iloc[24][0:10]))

count = 0 # 一段时间内的记录数
temp_dict = [] # 全局记录
for i in range(temp.shape[0]):
    if (count > 0): # 跳过当前已经记录的数据个数
        count = count - 1
        continue
    # 提取原则 前方有车并且距离小于250
    if (temp.iloc[i][8] > 0 and temp.iloc[i][9] < 250):
        count = 0
        t = temp.iloc[i][5]
        tt_dict = []
        # 记录当前时间片的所有记录
        for j in range(i, temp.shape[0]):
            if(temp.iloc[j][5] == t):
                tt_dict.append(list(temp.iloc[j][0:10]))
            else:
                count = j - i - 1
                break
        if(len(tt_dict) > 2):
            print(tt_dict)
            temp_dict.append(tt_dict)

print(temp_dict)
'''

'''
# 1号车的信息
v1_all_list = []
v1_t_list = []
v1_v_list = []

# 2号车的信息
v2_all_list = []
v2_t_list = []
v2_v_list = []

# 3号车的信息
v3_all_list = []
v3_t_list = []
v3_v_list = []

for i in range (shuzhi_shujv_fenge.shape[0]):
    if(shuzhi_shujv_fenge.iloc[i][4] == 1):
        v1_all_list.append(list(shuzhi_shujv_fenge.iloc[i]))
        v1_t_list.append(shuzhi_shujv_fenge.iloc[i][5])
        v1_v_list.append(shuzhi_shujv_fenge.iloc[i][6])
    elif(shuzhi_shujv_fenge.iloc[i][4] == 2):
        v2_all_list.append(list(shuzhi_shujv_fenge.iloc[i]))
        v2_t_list.append(shuzhi_shujv_fenge.iloc[i][5])
        v2_v_list.append(shuzhi_shujv_fenge.iloc[i][6])
    elif(shuzhi_shujv_fenge.iloc[i][4] == 3):
        v3_all_list.append(list(shuzhi_shujv_fenge.iloc[i]))
        v3_t_list.append(shuzhi_shujv_fenge.iloc[i][5])
        v3_v_list.append(shuzhi_shujv_fenge.iloc[i][6])


v1_v_np = np.array(v1_v_list)
v1_t_np = np.array(v1_t_list)

v2_v_np = np.array(v2_v_list)
v2_t_np = np.array(v2_t_list)

v3_v_np = np.array(v3_v_list)
v3_t_np = np.array(v3_t_list)

plt.plot(v1_t_np, v1_v_np, 'b')
plt.plot(v2_t_np, v2_v_np, 'r')
plt.plot(v3_t_np, v3_v_np, 'g')
plt.show()
'''





        
