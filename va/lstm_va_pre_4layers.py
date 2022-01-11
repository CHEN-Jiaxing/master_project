#* **********************
#* Author: CHEN, Jiaxing
#* Function: train for velocity prediction
#* Date: 2021.03.05
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

# ! 读取数据
data = np.load('./data_master/input_va_time_delay/input_1_va_pre.npy')

# 网络参数设置
BATCH_START = 0
TIME_STEPS = 10
BATCH_SIZE = 20
INPUT_SIZE = 5
OUTPUT_SIZE = 2
LR = 0.1

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
    if BATCH_START == len(data) - BATCH_SIZE * TIME_STEPS:
        BATCH_START = 0
    else:
        BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(20,activation='tanh',
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
model.add(Dense(10,activation='linear')) 
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

loss = []
print('Training ------------')
# for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
for step in range(500):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, t = batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    '''
    plt.subplot(2, 1, 1)
    plt.plot(t[0, :], Y_batch[0, : , 0], 'r', t[0, :], pred[0, : , 0], 'g--')
    plt.subplot(2,1, 2)
    plt.plot(t[0, :], Y_batch[0, : , 1], 'm', t[0, :], pred[0, : , 1], 'b-.')
    plt.draw()
    plt.pause(0.1)
    '''
    loss.append(cost)
    if step % 10 == 0:
        print('train cost: ', cost)

plt.figure(2)
plt.plot(np.array(loss))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('30s预测步长下的误差变化图——训练集')
plt.show()
# plt.xlim((0,450))
plt.ylim((0,2000))

# !   
model.save('lstm_va.h5')
print("========END========")
