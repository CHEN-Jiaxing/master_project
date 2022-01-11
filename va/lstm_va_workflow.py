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
from keras.models import load_model
from subsidiary import rmse_calc
import pandas as pd

# 网络参数设置
BATCH_START = 0
TIME_STEPS = 10
BATCH_SIZE = 20
INPUT_SIZE = 5
OUTPUT_SIZE = 2
CELL_SIZE = 5
LR = 0.1

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

def deal(company_name_list):
	# list转dataframe
    df = pd.DataFrame(company_name_list, columns=['1', '2'])

	# 保存到本地excel
    df.to_excel("temp.xlsx", index=False)

res = []
ts_l=[5,6,7,8,9,10,11,12]
for CELL_SIZE in ts_l:
    # ! 读取数据
    data = np.load('./data_master/input_va_time_delay/input_60_va_pre.npy')
    BATCH_START = 0
    # 数据长度取整
    BATCH_NUM = int(len(data) / BATCH_SIZE)
    data = data[0:BATCH_NUM * BATCH_SIZE]
    print("当前数据总数:", len(data))

    # 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
    
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(
        batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=CELL_SIZE,
        return_sequences=True,      # True: output at all steps. False: output as last step.
        stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
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
        
    # !   
    model.save('lstm_va.h5')
    print("===111=====END====111====")
    del(data)


    BATCH_START = 0

    # ! 读取数据
    data = np.load('./data_master/input_va_time_delay/input_2_60_va_vld.npy')
    model = load_model("lstm_va.h5")

    # 数据长度取整
    BATCH_NUM = int(len(data) / BATCH_SIZE)
    data = data[0:BATCH_NUM * BATCH_SIZE]
    print("当前数据总数:", len(data))


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
        
        y_v = np.append(y_v, Y_batch[0, : , 0])
        p_v = np.append(p_v, pred[0, : , 0])
        
        y_a = np.append(y_a, Y_batch[0, : , 1])
        p_a = np.append(p_a, pred[0, : , 1])

        results = model.evaluate(X_batch, Y_batch, BATCH_SIZE)
        # print("test loss, test acc:", results)
        loss.append(results)

    # 均方根误差
    rmse_v = rmse_calc(p_v, y_v)
    print("速度均方根误差：", rmse_v)
    rmse_a = rmse_calc(p_a, y_a)
    print("加速度均方根误差：", rmse_a)

    print("==222===END===222==")

    res.append([rmse_v,rmse_a])
deal(res)
print(res)