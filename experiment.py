#* **********************
#* Author: CHEN, Jiaxing
#* Function: test and experiment
#* Date: 2021.03.11
#* Modified by:
#* Changes:
#* **********************

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

'''
for i in range(50):
    # print("data_"+str(i)+" = sub.time_shift(data_" + str(i)+", 60)")
    print("data_"+str(i), end=',')
'''

'''
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

data = np.load('./data_master/input_dp/input_dp_6.npy')
veh_velc_list = data[:,6]
veh_acc_list = data[:,7]
veh_velc_list= savgol_filter(veh_velc_list, 51, 3, mode= 'nearest')
veh_acc_list= savgol_filter(veh_acc_list, 51, 3, mode= 'nearest')

plt.figure()
plt.subplot(2,1,1)
plt.plot(veh_velc_list, 'r')
plt.subplot(2,1,2)
plt.plot(veh_acc_list, 'g')
plt.show()
'''
print("===========End==========")


