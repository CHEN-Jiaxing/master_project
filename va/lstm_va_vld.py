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
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.models import load_model
from subsidiary import rmse_calc
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

# ! 读取数据
data = np.load('./data_master/input_va_time_delay/input_2_60_va_vld.npy')
'''
data[:,1]= savgol_filter(data[:,1], 31, 3, mode= 'nearest')
data[:,2]= savgol_filter(data[:,2], 31, 3, mode= 'nearest')
data[:,3]= savgol_filter(data[:,3], 31, 3, mode= 'nearest')
data[:,4]= savgol_filter(data[:,4], 31, 3, mode= 'nearest')
data[:,5]= savgol_filter(data[:,5], 31, 3, mode= 'nearest')
data[:,6]= savgol_filter(data[:,6], 31, 3, mode= 'nearest')
data[:,7]= savgol_filter(data[:,7], 31, 3, mode= 'nearest')
data[:,8]= savgol_filter(data[:,8], 31, 3, mode= 'nearest')
'''
model = load_model("./va/lstm_va_60.h5")

# 网络参数设置
BATCH_START = 0
TIME_STEPS = 6
BATCH_SIZE = 20
INPUT_SIZE = 5
OUTPUT_SIZE = 2

# 数据长度取整
BATCH_NUM = int(len(data) / BATCH_SIZE)
data = data[0:BATCH_NUM * BATCH_SIZE]
print("当前数据总数:", len(data))

# 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
def batch():
    global BATCH_START, TIME_STEPS
    t = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 1:6]
    X_batch = X_batch.reshape(BATCH_SIZE, TIME_STEPS,INPUT_SIZE)
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:8]
    Y_batch = Y_batch.reshape(BATCH_SIZE, TIME_STEPS,OUTPUT_SIZE)
    BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

loss = []

# 实际速度和预测输出速度
y_v = np.array([])
p_v = np.array([])

# 实际加速度和预测输出加速度
y_a = np.array([])
p_a = np.array([])

for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
    X_batch, Y_batch, t = batch()
    
    pred = model.predict(X_batch, BATCH_SIZE)
    '''
    plt.subplot(2, 1, 1)
    plt.title("1s预测步长下的预测情况", fontsize=12)
    plt.plot(t[0, :], Y_batch[0, : , 0], 'r',label="实际速度")
    plt.plot( t[0, :], pred[0, : , 0], 'g--',label="预测速度")
    plt.xlabel("时间(s)"+'\n'+"a 速度预测情况", fontsize=12)
    plt.ylabel("速度(km/h)", fontsize=12)
    if step == 0:
        plt.legend()
    '''
    # plt.plot(t[0, :], Y_batch[0, : , 0], 'r', t[0, :], pred[0, : , 0], 'g--')
    
    y_v = np.append(y_v, Y_batch[0, : , 0])
    p_v = np.append(p_v, pred[0, : , 0])
    '''
    plt.subplot(2,1, 2)
    plt.plot(t[0, :], Y_batch[0, : , 1], 'm',label="实际加速度")
    plt.plot( t[0, :], pred[0, : , 1], 'b--',label="预测加速度")
    plt.xlabel("时间(s)"+'\n'+"b 加速度预测情况", fontsize=12)
    plt.ylabel("加速度(m/(s^2))", fontsize=12)
    if step == 0:
        plt.legend()
    
    # plt.plot(t[0, :], Y_batch[0, : , 1], 'm', t[0, :], pred[0, : , 1], 'b-.')

    plt.draw()
    plt.pause(1e-6)
    '''
    y_a = np.append(y_a, Y_batch[0, : , 1])
    p_a = np.append(p_a, pred[0, : , 1])

    results = model.evaluate(X_batch, Y_batch, BATCH_SIZE)
    # print("test loss, test acc:", results)
    loss.append(results)
'''
plt.figure()
plt.plot(np.array(loss))
plt.xlim((0,30))
plt.ylim((0,100))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('1s预测步长下的误差变化图——验证集')
plt.show()
'''

'''
p_v= savgol_filter(p_v, 31, 3, mode= 'nearest')
y_v= savgol_filter(y_v, 31, 3, mode= 'nearest')
p_a= savgol_filter(p_a, 31, 3, mode= 'nearest')
y_a= savgol_filter(y_a, 31, 3, mode= 'nearest')
'''

plt.subplot(2, 1, 1)
plt.title("60s预测步长下的预测情况——工况二", fontsize=12)
plt.plot(y_v, 'r',label="实际速度")
plt.plot(p_v, 'g--',label="预测速度")
plt.xlabel("时间(s)"+'\n'+"a 速度预测情况", fontsize=12)
plt.ylabel("速度(km/h)", fontsize=12)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y_a, 'r',label="实际加速度")
plt.plot(p_a, 'g--',label="预加测速度")
plt.xlabel("时间(s)"+'\n'+"b 加速度预测情况", fontsize=12)
plt.ylabel("加速度(m/(s^2))", fontsize=12)
plt.legend()
plt.show()


# 均方根误差
rmse_v = rmse_calc(p_v, y_v)
print("速度均方根误差：", rmse_v)
rmse_a = rmse_calc(p_a, y_a)
print("加速度均方根误差：", rmse_a)

print("=====END=====")
