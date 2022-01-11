#* **********************
#* Author: CHEN, Jiaxing
#* Function: train for pfc prediction
#* Date: 2021.03.23
#* Modified by:
#* Changes:
#* **********************

# 画图中文显示
import matplotlib
matplotlib.rc("font",family='DengXian')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Dropout
from keras.optimizers import Adam,rmsprop
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
CELL_SIZE = 16
LR = 1e-3

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
    if BATCH_START >= len(data) - BATCH_SIZE * TIME_STEPS:
        BATCH_START = 0
    else:
        BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,activation='tanh',
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# model.add(LSTM(32, return_sequences=True, stateful=True))
# model.add(Dense(54,activation='linear'))
# model.add(Dense(32,activation='linear'))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE,activation='linear')))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

loss = []
print('Training ------------')
# for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
for step in range(5001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, t = batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    '''
    plt.plot(t[0, :], Y_batch[0, : , 0], 'r', t[0, :], pred[0, : , 0], 'g--')
    plt.draw()
    plt.pause(1e-3)
    '''
    if step % 10 == 0:
        print('train cost: ', cost)

    loss.append(cost)

loss = np.array(loss)
loss = savgol_filter(loss, 11, 3, mode= 'nearest')
plt.plot(loss)
# plt.xlim((0,450))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('蓄电池输出功率训练集下的误差变化图')
plt.show()

# ! pfc预测模型保存
model.save('lstm_pfc.h5')
print("========END========")