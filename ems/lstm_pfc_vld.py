#* **********************
#* Author: CHEN, Jiaxing
#* Function: validation for velocity prediction
#* Date: 2021.03.08
#* Modified by:
#* Changes:
#* **********************

# 画图中文显示
import matplotlib
matplotlib.rc("font",family='DengXian')
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.models import load_model
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# ! 读取数据
data = np.load('./data_master/input_pfc_12.9/input_pfc_vld.npy')
data = np.concatenate((data,data,data,data,data),axis=0)
data[:,0] = savgol_filter(data[:,0], 31, 3, mode= 'nearest')
data[:,1] = savgol_filter(data[:,1], 31, 3, mode= 'nearest')
data[:,2] = savgol_filter(data[:,2], 31, 3, mode= 'nearest')
data[:,3] = savgol_filter(data[:,3], 31, 3, mode= 'nearest')
data[:,4] = savgol_filter(data[:,4], 31, 3, mode= 'nearest')
data[:,5] = savgol_filter(data[:,5], 31, 3, mode= 'nearest')
data[:,6] = savgol_filter(data[:,6], 31, 3, mode= 'nearest')

# 网络参数设置
model = load_model("lstm_pfc.h5")
BATCH_START = 0
TIME_STEPS = 15
BATCH_SIZE = 4
INPUT_SIZE = 6
OUTPUT_SIZE = 1

# 数据长度取整
BATCH_NUM = int(len(data) / (BATCH_SIZE * TIME_STEPS))
data = data[0:BATCH_NUM * BATCH_SIZE * TIME_STEPS]
print("当前数据总数:", len(data))

# 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
def batch():
    global BATCH_START, TIME_STEPS
    t = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 0:6]
    X_batch = X_batch.reshape(BATCH_SIZE, TIME_STEPS,INPUT_SIZE)
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:7]
    Y_batch = Y_batch.reshape(BATCH_SIZE, TIME_STEPS,OUTPUT_SIZE)
    BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

loss = []
# *实际功率输出y_p和预测功率输出p_p
y_p = np.array([])
p_p = np.array([])

for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
    X_batch, Y_batch, t = batch()
    
    pred = model.predict(X_batch, BATCH_SIZE)
    '''
    plt.title("蓄电池输出功率预测情况")
    plt.plot(t[0, :], Y_batch[0, : , 0], 'r',label="动态规划")
    plt.plot( t[0, :], pred[0, : , 0], 'g--',label="LSTM")
    plt.xlabel("批次")
    plt.ylabel("功率")
    if step == 0:
        plt.legend()
    
    # plt.plot(t[0, :], Y_batch[0, : , 0], 'r', t[0, :], pred[0, : , 0], 'g--')
    plt.draw()
    plt.pause(1e-4)
    '''
    # y_p = np.append(y_p, Y_batch[0, : , 0])
    p_p = np.append(p_p, pred[0, : , 0])
    '''
    results = model.evaluate(X_batch, Y_batch, BATCH_SIZE)
    print("test loss, test acc:", results)
    loss.append(results)
    '''
'''
plt.figure()
plt.plot(np.array(loss))
# plt.xlim((0,30))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('蓄电池输出功率验证集下的误差变化图')
plt.show()
plt.ylim((0,1000))

# 计算均方根误差
def rmse_calc(x, y):
    l = len(x)
    s = 0.0
    for i in range(l):
        s += (x[i]- y[i])**2
    return math.sqrt(s/l)

# 均方根误差
rmse_v = rmse_calc(p_p, y_p)
print("输出功率均方根误差：", rmse_v)
'''
np.save("p_lstm", p_p)
# np.save("p_dp", y_p)

print("=====END=====")