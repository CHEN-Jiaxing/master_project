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
print(shuzhi_shujv_col)

'''
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

# print(shuzhi_shujv_fenge)

data_sampled = []

# ！！！需要修改的起始时间
time = 2.0
time_flag = 0

for i in range(shuzhi_shujv_fenge.shape[0]):
    if (shuzhi_shujv_fenge.iloc[i][5] != time ):
        if (time_flag == 0):
            time = time + 1
            time_flag = 1
    else:
        time_flag = 0
        data_sampled.append(list(shuzhi_shujv_fenge.iloc[i]))
        

print(data_sampled)

'''
