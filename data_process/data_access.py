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

# 删除无效列
shuzhi_shujv_fenge = shuzhi_shujv_fenge.drop([' '], axis =1)

print(shuzhi_shujv_fenge)

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

'''
# 类型转换
shuzhi_shujv_veh_nr['Lane'] = shuzhi_shujv_veh_nr['Lane'].astype(int)
shuzhi_shujv_veh_nr['    WorldX'] = shuzhi_shujv_veh_nr['    WorldX'].astype(float)
shuzhi_shujv_veh_nr['    WorldY'] = shuzhi_shujv_veh_nr['    WorldY'].astype(float)
shuzhi_shujv_veh_nr['     a'] = shuzhi_shujv_veh_nr['     a'].astype(float)
shuzhi_shujv_veh_nr['     VehNr'] = shuzhi_shujv_veh_nr['     VehNr'].astype(int)
shuzhi_shujv_veh_nr['       t'] = shuzhi_shujv_veh_nr['       t'].astype(float)
shuzhi_shujv_veh_nr['    vMS'] = shuzhi_shujv_veh_nr['    vMS'].astype(float)
shuzhi_shujv_veh_nr[' Power'] = shuzhi_shujv_veh_nr[' Power'].astype(float)
shuzhi_shujv_veh_nr['  LVeh'] = shuzhi_shujv_veh_nr['  LVeh'].astype(int)
shuzhi_shujv_veh_nr['  Head'] = shuzhi_shujv_veh_nr['  Head'].astype(float)
'''
shujv_veh_lst = [] 

for i in range(shuzhi_shujv_veh_nr.shape[0]):
    if(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_1 or\
       shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_2 or\
       shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_3):
        shujv_veh_lst.append(list(shuzhi_shujv_veh_nr.iloc[i]))

shujv_veh_np = np.array(shujv_veh_lst)
print(shujv_veh_np)
print(shujv_veh_np.shape)
