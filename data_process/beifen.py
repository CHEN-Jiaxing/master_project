# Author: CHEN, Jiaxing
# Function: Data Process
# Date: 2020.10.03
# Modified by:
# Changes:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading

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

# 多线程选取 前车 序号
veh_nr_1_ls = []

class myThread(threading.Thread):
    def __init__(self, start_nr, end_nr, veh_nr_1_st):
        threading.Thread.__init__(self)
        self.start_nr = int(start_nr)
        self.end_nr = int(end_nr)
        self.veh_nr_1_st = veh_nr_1_st

    def run(self):
        for i in range(self.start_nr, self.end_nr):
            if(shuzhi_shujv_fenge.iloc[i][8] <= 0):
                self.veh_nr_1_st.add(shuzhi_shujv_fenge.iloc[i][4])

ls1 = ls2 = ls3 = ls4 = ls5 = ls6 = ls7 = ls8 = {0}
len_shujv = shuzhi_shujv_fenge.shape[0] - shuzhi_shujv_fenge.shape[0] % 8

thread1 = myThread(len_shujv * 0 / 8, len_shujv * 1 / 8, ls1)
thread2 = myThread(len_shujv * 1 / 8, len_shujv * 2 / 8, ls2)
thread3 = myThread(len_shujv * 2 / 8, len_shujv * 3 / 8, ls3)
thread4 = myThread(len_shujv * 3 / 8, len_shujv * 4 / 8, ls4)
thread5 = myThread(len_shujv * 4 / 8, len_shujv * 5 / 8, ls5)
thread6 = myThread(len_shujv * 5 / 8, len_shujv * 6 / 8, ls6)
thread7 = myThread(len_shujv * 6 / 8, len_shujv * 7 / 8, ls7)
thread8 = myThread(len_shujv * 7 / 8, len_shujv * 8 / 8, ls8)

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()

veh_nr_1_st = thread1.veh_nr_1_st.union(thread2.veh_nr_1_st)\
              .union(thread3.veh_nr_1_st).union(thread4.veh_nr_1_st)\
              .union(thread5.veh_nr_1_st).union(thread6.veh_nr_1_st)\
              .union(thread7.veh_nr_1_st).union(thread8.veh_nr_1_st)

veh_nr_1_ls = list(veh_nr_1_st)
veh_nr_1_ls.sort()
print(veh_nr_1_ls)

# 对 前车 优化选取
def veh_nr_1_ls_shaixuan(ls):
    ls_temp = ls
    ls = []
    jiasu = 0
    for i_ls in range(len(ls_temp)):
        veh_nr_1 = ls_temp[i_ls]
        veh_nr_2 = 1
        veh_nr_3 = 1
        
        for i in range(jiasu, shuzhi_shujv_fenge.shape[0]):
            if(shuzhi_shujv_fenge.iloc[i][8] == veh_nr_1):
                veh_nr_2 = int(shuzhi_shujv_fenge.iloc[i][4])
                break
        for j in range(i,shuzhi_shujv_fenge.shape[0]):
            if(shuzhi_shujv_fenge.iloc[j][8] == veh_nr_2):
                veh_nr_3 = int(shuzhi_shujv_fenge.iloc[j][4])
                jiasu = j
                break
        if(veh_nr_2 != 1 and veh_nr_3 != 1):
            ls.append(veh_nr_1)
    return ls

veh_nr_1_ls = veh_nr_1_ls_shaixuan(veh_nr_1_ls)
# print(veh_nr_1_ls)

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
 if(abs(shuzhi_shujv_fenge.iloc[i][5] - time_jieduan_kaishi) < 1e-3):
     xuhao_jieduan_kaishi = i
     break

# 获取 截取 结束 序号
for j in range(i,shuzhi_shujv_fenge.shape[0]):
 if(abs(shuzhi_shujv_fenge.iloc[j][5] - (time_jieduan_jieshu + 0.2)) < 1e-3 ):
     xuhao_jieduan_jieshu = j - 1 
     break
 elif(abs(time_jieduan_jieshu - 3600) < 1e-3):
        xuhao_jieduan_jieshu = shuzhi_shujv_fenge.shape[0]

# print(xuhao_jieduan_kaishi)
# print(xuhao_jieduan_jieshu)

# 提取包含当前队列的信息
shuzhi_shujv_veh_nr = shuzhi_shujv_fenge[xuhao_jieduan_kaishi:xuhao_jieduan_jieshu+1]
shuzhi_shujv_veh_nr = shuzhi_shujv_veh_nr.reset_index()
shuzhi_shujv_veh_nr.drop(['index'], axis=1, inplace=True)


# 前车的信息
v1_t_list = []
v1_v_list = []

# 中车的信息
v2_t_list = []
v2_v_list = []

# 后车的信息
v3_t_list = []
v3_v_list = []

for i in range (shuzhi_shujv_veh_nr.shape[0]):
    if(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_1):
        v1_t_list.append(shuzhi_shujv_veh_nr.iloc[i][5])
        v1_v_list.append(shuzhi_shujv_veh_nr.iloc[i][6])
        
    elif(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_2):
        v2_t_list.append(shuzhi_shujv_veh_nr.iloc[i][5])
        v2_v_list.append(shuzhi_shujv_veh_nr.iloc[i][6])
        
    elif(shuzhi_shujv_veh_nr.iloc[i][4] == veh_nr_3):
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

# 提取1，2，3号车信息

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
