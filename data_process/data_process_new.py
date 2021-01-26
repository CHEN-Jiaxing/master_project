#* **********************
#* Author: CHEN, Jiaxing
#* Function: Data Process
#* Date: 2021.01.15
#* Modified by:
#* Changes:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取原始数据
shujv = pd.read_table('hld_001.fzp', encoding = 'gbk')

# 提取有用数据
shujv = shujv.iloc[15:]
shujv = pd.DataFrame([jj.split(';') for jj in shujv['$VISION']])
shujv.columns = shujv.iloc[0]
shujv = shujv.iloc[1:]
rear_x = []
front_x = []
y = []
for i in range(len(shujv)):
    shujv.iloc[i][5] = shujv.iloc[i][5].split(' ')
    shujv.iloc[i][6] = shujv.iloc[i][6].split(' ')
    rear_x.append(shujv.iloc[i][5][0])
    y.append(shujv.iloc[i][5][1])
    front_x.append(shujv.iloc[i][6][0])
shujv = shujv.reset_index()
shujv.insert(8,'rear_x',rear_x)
shujv.insert(9,'y',y)
shujv.insert(10,'front_x',front_x)
for i in range(len(shujv)):
    if (float(shujv.loc[i][7][0]) >= 0.0 and float(shujv.loc[i][7][0]) <= 900.0) != True:
        shujv.drop(i,axis=0,inplace=True)
shujv.drop(['COORDREAR'],axis=1,inplace=True)
shujv.drop(['COORDFRONT'],axis=1,inplace=True)
shujv.drop(['index'],axis=1,inplace=True)
shujv = shujv.reset_index(drop=True)

# 数据类型转换
shujv['$VEHICLE:SIMSEC'] = shujv['$VEHICLE:SIMSEC'].astype(float)
shujv['NO'] = shujv['NO'].astype(int)
shujv['SPEED'] = shujv['SPEED'].astype(float)
shujv['ACCELERATION'] = shujv['ACCELERATION'].astype(float)
shujv['FOLLOWDIST'] = shujv['FOLLOWDIST'].astype(float)
shujv['rear_x'] = shujv['rear_x'].astype(float)
shujv['y'] = shujv['y'].astype(float)
shujv['front_x'] = shujv['front_x'].astype(float)

# 时间行数对应
t = list(range(1,int(shujv.iloc[len(shujv)-1][0]+1)))
idx = []
idx_tmp = -1
for i in range(len(t)):
    for j in range(idx_tmp, len(shujv)):
        if int(shujv.iloc[j][0]) == t[i]:
            idx_tmp = j
            idx.append(idx_tmp)
            break
    if idx_tmp == -1:
        idx.append(idx_tmp)

# 计算跟车距离和前车序号
def dst_idx_veh_nr_calc(dangqian_shike, veh_nr):
    dst = -1
    veh_tbf_nr = -2
    idx_tbf = -2
    if int(dangqian_shike) == t[-1]:
        print("此表没有序号车后续记录-1")
        return [dst, idx_tbf, veh_tbf_nr]
    for i in range(idx[int(dangqian_shike)], idx[int(dangqian_shike)+1]):
        if shujv.iloc[i][1] == veh_nr:
            dst =  shujv.iloc[i][4]
    if dst == -1:
        print("此表没有序号车后续记录-2")
        return [dst, idx_tbf, veh_tbf_nr]
    for i in range(idx[int(dangqian_shike) - 1], idx[int(dangqian_shike)]):
        if shujv.iloc[i][1] == veh_nr:
            qianche_front_x = shujv.iloc[i][7] + dst
            qianche_y = shujv.iloc[i][6]
            break
    for i in range(idx[int(dangqian_shike) - 1], idx[int(dangqian_shike)]):
        if abs(shujv.iloc[i][7] - qianche_front_x) < 0.1 and abs(shujv.iloc[i][6] - qianche_y) < 0.1:
            veh_tbf_nr = shujv.iloc[i][1]
            idx_tbf = i
            break
    if veh_tbf_nr == -2:
        print("此表没有要寻找序号车后续记录-3")
    return [dst, idx_tbf, veh_tbf_nr]

# !!!输入想要的当前车编号
veh_cur_nr = 3
# !!!输入想要的当前车编号
zuhe_input_neur = []

for i in range(len(idx) - 2):
    idx_cur = -1
    dangqian_shike = -2
    if idx[i] == -1 or idx[i+1] == -1:
        continue
    for j in range(idx[i], idx[i+1]):
        if shujv.iloc[j][1] == veh_cur_nr:
            idx_cur = j
            dangqian_shike = shujv.iloc[j][0]
            break
    if idx_cur == -1:
        continue
    dst_2_3,idx_mid,veh_mid_nr = dst_idx_veh_nr_calc(dangqian_shike, veh_cur_nr)
    dst_1_2,idx_frnt,veh_frnt_nr = dst_idx_veh_nr_calc(dangqian_shike, veh_mid_nr)
    if idx_cur != -1 and idx_mid != -2 and idx_frnt!= -2:
        zuhe_input_neur.append([idx_frnt, idx_mid, idx_cur])        

# 整理信息以便输入神经网络
hld_x = np.array((range(11)))
hld_x = (hld_x)*10+5
for i in range(len(zuhe_input_neur)):
    t = shujv.iloc[zuhe_input_neur[i][0]][0]
    v1 = shujv.iloc[zuhe_input_neur[i][0]][2]
    a1 = shujv.iloc[zuhe_input_neur[i][0]][3]
    y1 = shujv.iloc[zuhe_input_neur[i][0]][6]
    x1 = shujv.iloc[zuhe_input_neur[i][0]][7]
    v2 = shujv.iloc[zuhe_input_neur[i][1]][2]
    a2 = shujv.iloc[zuhe_input_neur[i][1]][3]
    v3 = shujv.iloc[zuhe_input_neur[i][2]][2]
    a3 = shujv.iloc[zuhe_input_neur[i][2]][3]
    for j in range(len(hld_x)-1):
        if y1 < hld_x[j]:
            break
    if j%2 == 0:
        x1 = 900.0 - x1
    zuhe_input_neur[i] = [t, x1, v1, a1, v2, a2, v3, a3,y1]
# print(zuhe_input_neur)
zuhe_input_neur = np.array(zuhe_input_neur)
plt.plot(zuhe_input_neur[:,0],zuhe_input_neur[:,2],'r')
plt.plot(zuhe_input_neur[:,0],zuhe_input_neur[:,4],'b')
plt.plot(zuhe_input_neur[:,0],zuhe_input_neur[:,6],'g')
plt.show()
