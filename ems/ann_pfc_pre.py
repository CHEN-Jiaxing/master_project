#* **********************
#* Author: CHEN, Jiaxing
#* Function: ANN as the compared model
#* Date: 2021.12.02
#* Modified by:
#* Changes:
#* **********************

# 画图中文显示
import matplotlib
matplotlib.rc("font",family='DengXian')
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import adam, rmsprop, adadelta
import numpy as np
import matplotlib.pyplot as plt
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

# ! 读取数据
data = np.load('./data_master/input_pfc_12.9/input_pfc_pre.npy')
data[:,0] = savgol_filter(data[:,0], 31, 3, mode= 'nearest')
data[:,1] = savgol_filter(data[:,1], 31, 3, mode= 'nearest')
data[:,2] = savgol_filter(data[:,2], 31, 3, mode= 'nearest')
data[:,3] = savgol_filter(data[:,3], 31, 3, mode= 'nearest')
data[:,4] = savgol_filter(data[:,4], 31, 3, mode= 'nearest')
data[:,5] = savgol_filter(data[:,5], 31, 3, mode= 'nearest')
data[:,6] = savgol_filter(data[:,6], 31, 3, mode= 'nearest')

# 网络参数设置
BATCH_START = 0
TIME_STEPS = 15
BATCH_SIZE = 4
INPUT_SIZE = 6
OUTPUT_SIZE = 1
LR = 1e-3

# 数据长度取整
BATCH_NUM = int(len(data) / (BATCH_SIZE * TIME_STEPS))
data = data[0:BATCH_NUM * BATCH_SIZE * TIME_STEPS]
print("当前数据总数:", len(data))

# 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
def batch():
    global BATCH_START, TIME_STEPS
    t = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE)
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 0:6]
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:7]
    if BATCH_START >= len(data) - BATCH_SIZE * TIME_STEPS:
        BATCH_START = 0
    else:
        BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

#construct model
model = Sequential()
model.add(Dense(16, init='uniform',activation='linear',input_dim=INPUT_SIZE))
# model.add(Dense(32, activation='linear'))
model.add(Dense(OUTPUT_SIZE,activation='linear'))
adamoptimizer = adam(LR, beta_1=0.9, beta_2=0.999, decay=0.00001)
model.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )

loss = []
print('Training ------------')
# for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
for step in range(5001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, t = batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)

    '''
    plt.plot(t[:], Y_batch[: , 0], 'r', t[:], pred[: , 0], 'g--')
    plt.draw()
    plt.pause(1e-3)
    '''
    if step % 10 == 0:
        print('train cost: ', cost)

    loss.append(cost)

loss = np.array(loss)
loss = savgol_filter(loss, 11, 3, mode= 'nearest')
plt.plot(loss[:,0])
# plt.xlim((0,450))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('蓄电池输出功率训练集下的误差变化图')
plt.show()

# ! pfc预测模型保存
model.save('ann_pfc.h5')
print("========END========")