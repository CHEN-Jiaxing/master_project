#* **********************
#* Author: CHEN, Jiaxing
#* Function: ANN as the compared model
#* Date: 2021.11.04
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

# ! 读取数据
data = np.load('./data_master/input_va_time_delay/input_60_va_pre.npy')

# 网络参数设置
BATCH_START = 0
TIME_STEPS = 10
BATCH_SIZE = 20
INPUT_SIZE = 5
OUTPUT_SIZE = 2
LR = 0.001

# 数据长度取整
BATCH_NUM = int(len(data) / BATCH_SIZE)
data = data[0:BATCH_NUM * BATCH_SIZE]
print("当前数据总数:", len(data))

#train data
# 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
def batch():
    global BATCH_START, TIME_STEPS
    t = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE)
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 1:6]
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:8]

    if BATCH_START == len(data) - BATCH_SIZE * TIME_STEPS:
        BATCH_START = 0
    else:
        BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

#construct model
model = Sequential()
model.add(Dense(9, init='uniform',activation='linear' ,input_dim=INPUT_SIZE))
# model.add(Dense(10, activation='linear'))
model.add(Dense(OUTPUT_SIZE,activation='linear'))
adamoptimizer = adam(LR, beta_1=0.9, beta_2=0.999, decay=0.00001)
model.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )

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
    plt.plot(t[:], Y_batch[: , 0], 'r', t[:], pred[: , 0], 'g--')
    plt.subplot(2,1, 2)
    plt.plot(t[:], Y_batch[: , 1], 'm', t[:], pred[: , 1], 'b-.')
    plt.draw()
    plt.pause(0.1)
    '''
    loss.append(cost)
    if step % 10 == 0:
        print('train cost: ', cost)

plt.figure(2)
print(loss)
plt.plot(np.array(loss))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('30s预测步长下的误差变化图——训练集')
plt.show()
# plt.xlim((0,450))
plt.ylim((0,2000))

# !   
model.save('ann_va.h5')
print("========END========")
