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

# 提取有用数据
shuzhi_shujv_col = yuanshi_shujv.iloc[13:]
shuzhi_shujv_fenge = pd.DataFrame([jj.split(';') for jj in shuzhi_shujv_col['Vehicle Record']])
shuzhi_shujv_fenge.columns = shuzhi_shujv_fenge.iloc[0]
shuzhi_shujv_fenge = shuzhi_shujv_fenge.iloc[1:]
shuzhi_shujv_fenge = shuzhi_shujv_fenge.drop([' '], axis =1)

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

# ！！！需要更改的 VehNr 值

veh_nr_1 = 1
veh_nr_2 = 1
veh_nr_3 = 1
time_jieduan_kaishi = 0
time_jieduan_jieshu = 0

# 获取截取片段 的 开始时刻
for i in range(shuzhi_shujv_fenge.shape[0]):
    if(shuzhi_shujv_fenge.iloc[i][8] == veh_nr_1):
        veh_nr_2 = int(shuzhi_shujv_fenge.iloc[i][4])
        break
for j in range(i,shuzhi_shujv_fenge.shape[0]):
    if(shuzhi_shujv_fenge.iloc[j][8] == veh_nr_2):
        veh_nr_3 = int(shuzhi_shujv_fenge.iloc[j][4])
        time_jieduan_kaishi = shuzhi_shujv_fenge.iloc[j][5]
        time_jieduan_jieshu = shuzhi_shujv_fenge.iloc[j][5]
        break

# print(veh_nr_2)
# print(veh_nr_3)

# 获取截取片段 的 结束时刻
for k in range(j,shuzhi_shujv_fenge.shape[0]):
    if(shuzhi_shujv_fenge.iloc[k][4] == veh_nr_1):
        time_jieduan_jieshu = shuzhi_shujv_fenge.iloc[k][5]
    if(shuzhi_shujv_fenge.iloc[k][5] - time_jieduan_jieshu > 1):
        break

# print(time_jieduan_kaishi)
# print(time_jieduan_jieshu)

xuhao_jieduan_kaishi = 0
xuhao_jieduan_jieshu = 0

# 获取 截取 开始 序号
for i in range(shuzhi_shujv_fenge.shape[0]):
 if(shuzhi_shujv_fenge.iloc[i][5] == time_jieduan_kaishi):
     xuhao_jieduan_kaishi = i
     break

# 获取 截取 结束 序号
for i in range(shuzhi_shujv_fenge.shape[0]):
 if(shuzhi_shujv_fenge.iloc[i][5] == time_jieduan_jieshu + 0.2):
     xuhao_jieduan_jieshu = i - 1
     break

# print(xuhao_jieduan_kaishi)
# print(xuhao_jieduan_jieshu)

# 提取包含当前队列的信息
shuzhi_shujv_veh_nr = shuzhi_shujv_fenge[xuhao_jieduan_kaishi:xuhao_jieduan_jieshu+1]
shuzhi_shujv_veh_nr = shuzhi_shujv_veh_nr.reset_index()
shuzhi_shujv_veh_nr.drop(['index'], axis=1, inplace=True)


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

for i in range (shuzhi_shujv_veh_nr.shape[0]):
    if(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_1):
        v1_all_list.append(list(shuzhi_shujv_veh_nr.iloc[i]))
        v1_t_list.append(shuzhi_shujv_veh_nr.iloc[i][5])
        v1_v_list.append(shuzhi_shujv_veh_nr.iloc[i][6])
    elif(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_2):
        v2_all_list.append(list(shuzhi_shujv_veh_nr.iloc[i]))
        v2_t_list.append(shuzhi_shujv_veh_nr.iloc[i][5])
        v2_v_list.append(shuzhi_shujv_veh_nr.iloc[i][6])
    elif(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_3):
        v3_all_list.append(list(shuzhi_shujv_veh_nr.iloc[i]))
        v3_t_list.append(shuzhi_shujv_veh_nr.iloc[i][5])
        v3_v_list.append(shuzhi_shujv_veh_nr.iloc[i][6])


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


shujv_veh_lst = [] 

for i in range(shuzhi_shujv_veh_nr.shape[0]):
    if(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_1 or\
       shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_2 or\
       shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_3):
        shujv_veh_lst.append(list(shuzhi_shujv_veh_nr.iloc[i]))

shujv_veh_np = np.array(shujv_veh_lst)
print(shujv_veh_np)
print(shujv_veh_np.shape)
        
duilie_list = []

# 分别对1，2，3号车提取信息
for i_hang in range(shujv_hangshu):
    if(i_hang %3 != 0):
        continue
    
    info_veh_nr_1 = []
    info_veh_nr_2 = []
    info_veh_nr_3 = []
    for i in range(i_hang, i_hang + 3):
        if(shujv_veh_np[i][4] == veh_nr_1):
            if(shujv_veh_np[i][1] < 2700):
                info_veh_nr_1.append(float(2700 - shujv_veh_np[i][1]))
            elif(shujv_veh_np[i][1] < 3100):
                info_veh_nr_1.append(float(3100 - shujv_veh_np[i][1]))
            elif(shujv_veh_np[i][1] < 3500):
                info_veh_nr_1.append(float(3500 - shujv_veh_np[i][1]))
            elif(shujv_veh_np[i][1] < 4000):
                info_veh_nr_1.append(float(4000 - shujv_veh_np[i][1]))
            else:
                info_veh_nr_1.append(float(1000))
            info_veh_nr_1.append(shujv_veh_np[i][6])
            info_veh_nr_1.append(shujv_veh_np[i][3])
        elif(shujv_veh_np[i][4] == veh_nr_2):
            info_veh_nr_2.append(shujv_veh_np[i][6])
            info_veh_nr_2.append(shujv_veh_np[i][3])
        elif(shujv_veh_np[i][4] == veh_nr_3):
            info_veh_nr_3.append(shujv_veh_np[i][6])
            info_veh_nr_3.append(shujv_veh_np[i][3])
    duilie_list.append(info_veh_nr_1 + info_veh_nr_2 + info_veh_nr_3)

print(duilie_list)

duilie_np = np.array(duilie_list)
print(duilie_np.shape)
